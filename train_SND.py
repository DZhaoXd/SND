import argparse
import os
import copy
import datetime
import logging
import time
import math
import numpy as np
from collections import OrderedDict

import torch
import random
import pickle
import torch.nn.functional as F
from PIL import Image
from core.datasets.transform import Compose

from core.configs import cfg
from core.datasets import build_dataset, rand_mixer_v2
from core.models import build_model, build_feature_extractor, build_classifier 
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU 
from core.utils.misc import get_color_pallete, strip_prefix_if_present, WeightEMA, denormalizeimage
from core.utils.losses import BinaryCrossEntropy, pseudo_labels_probs, update_running_conf, full2weak

from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.apis.inference import inference, multi_scale_inference, sample_val_images
from core.apis.inference import evel_stu, run_test, run_candidate
from core.apis.inference import soft_ND_measure, entropy_measure
import pandas as pd
from datasets.generate_city_label_info import gen_lb_info
import higher

      
def train(cfg, local_rank, distributed):
    logger = logging.getLogger("DTST.trainer")
    logger.info("Start training")

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)
    
    classifier = build_classifier(cfg)
    classifier.to(device)

    ## distributed training
    batch_size = cfg.SOLVER.BATCH_SIZE
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size())//2
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg2
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()
    
    ## optimizer 
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()
    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    
    output_dir = cfg.OUTPUT_DIR
    local_rank = 0
    start_epoch = 0
    iteration = 0
    
    ####  resume the ckpt 
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
    
    ####  loss define
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    binary_ce = BinaryCrossEntropy(ignore_index=255)
    
    ####  model define
    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    classifier.train()
    classifier_his = copy.deepcopy(classifier).cuda()
    feature_extractor_his = copy.deepcopy(feature_extractor).cuda()
    classifier_his.eval()
    feature_extractor_his.eval()
    
    
    ###### Mixup and  rsc init 
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'CTR')) and \
        os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'CTR_O')):
        start_from_None = False
        print('+++++++++++++++++++++++  mixup from stable')
    else:
        print('+++++++++++++++++++++++  mixup from None')
        start_from_None = True
        run_candidate(cfg, feature_extractor_his, classifier_his, local_rank, distributed, init_candidate=True, update_meta=True)
        gen_lb_info(cfg, 'CTR')  ## for mixup 
        gen_lb_info(cfg, 'CTR_O')  ## for rsc 
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False, rsc='CTR_O.p')
    print('len(tgt_train_data)', len(tgt_train_data))
    if cfg.TCR.OPEN:
        tgt_train_data.mixer.label_to_file, _ = pickle.load(open(os.path.join(cfg.OUTPUT_DIR, 'CTR.p'), "rb"))

    if distributed:
        tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    else:
        tgt_train_sampler = None
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True, sampler=tgt_train_sampler, drop_last=True)

    ###### confident  init 
    #default param in SAC (https://github.com/visinf/da-sac)
    THRESHOLD_BETA = 0.001
    running_conf = torch.zeros(cfg.MODEL.NUM_CLASSES).cuda()
    running_conf.fill_(THRESHOLD_BETA)

    ###### Dynamic teacher init
    if cfg.DTU.DYNAMIC:
        stu_eval_list = []
        stu_score_buffer = []
        res_dict = {'stu_ori':[], 'stu_now':[], 'update_iter':[]}
        
        
    cls_his_optimizer = WeightEMA(
        list(classifier_his.parameters()), 
        list(classifier.parameters()),
        alpha= cfg.DTU.EMA_WEIGHT,
    )  
    feature_extractor_his = copy.deepcopy(feature_extractor).cuda()
    fea_his_optimizer = WeightEMA(
        list(feature_extractor_his.parameters()), 
        list(feature_extractor.parameters()),
        alpha= cfg.DTU.EMA_WEIGHT,
    )      
    
    
    if cfg.METAPL.OPEN:
        cls_meta = build_classifier(cfg).to(device)
        gen_lb_info(cfg, 'meta_val')
        gen_lb_info(cfg, 'meta_val_mixup')
        

    start_training_time = time.time()
    end = time.time()
    for rebuild_id in range(255):
        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
        for i, (tgt_input, y, names, tgt_trans_param, tgt_img_full, mix_label) in enumerate(tgt_train_loader):
            
            data_time = time.time() - end
            
            current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
            for index in range(len(optimizer_fea.param_groups)):
                optimizer_fea.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_cls.param_groups)):
                optimizer_cls.param_groups[index]['lr'] = current_lr*10

            optimizer_fea.zero_grad()
            optimizer_cls.zero_grad()

            tgt_input = tgt_input.cuda(non_blocking=True)
            tgt_size = tgt_input.shape[-2:]
            tgt_img_full = tgt_img_full.cuda(non_blocking=True)
            mix_label = mix_label.cuda()
            tgt_full_size = tgt_img_full.shape[-2:]
            
            ### stu forward
            tgt_pred = classifier(feature_extractor(tgt_input)[1])
            tgt_pred = F.interpolate(tgt_pred, size=tgt_size, mode='bilinear', align_corners=True)

            ######### dy update
            if cfg.DTU.DYNAMIC:
                with torch.no_grad():
                    tgt_pred_full = classifier(feature_extractor(tgt_img_full.clone().detach())[1])
                    output = F.softmax(tgt_pred_full.clone().detach(), dim=1).detach()
                    if cfg.DTU.PROXY_METRIC == 'ENT': 
                        entropy_val = entropy_measure(output)
                        stu_score_buffer.append(entropy_val)
                        stu_eval_list.append([tgt_img_full.clone().detach().cpu()])
                    elif cfg.DTU.PROXY_METRIC == 'Soft_ND':
                        soft_ND_val, soft_ND_state = soft_ND_measure(output, select_point=100)
                        stu_score_buffer.append(soft_ND_val)
                        stu_eval_list.append([tgt_img_full.clone().detach().cpu(), soft_ND_state.cpu()])
                    else:
                        print('no support')
                        return
            ###########
                    
            #### history model 
            with torch.no_grad():   
                
                size = tgt_img_full.shape[-2:]
                train_feas = feature_extractor_his(tgt_img_full)[1]
                tgt_pred_his_full = classifier_his(train_feas, tgt_img_full.shape[-2:])
                tgt_prob_his = F.softmax(full2weak(tgt_pred_his_full, tgt_trans_param), dim=1)
                train_feas = full2weak(train_feas, tgt_trans_param, down_ratio=8)
                
                # pos label
                running_conf = update_running_conf(F.softmax(tgt_pred_his_full, dim=1), running_conf, THRESHOLD_BETA)
                #psd_label, _, _ = pseudo_labels_probs(tgt_prob_his, running_conf, THRESHOLD_BETA)
                if cfg.METAPL.OPEN:
                    psd_label = tgt_prob_his.max(1)[1].detach()
                else:
                    psd_label, _, _ = pseudo_labels_probs(tgt_prob_his, running_conf, THRESHOLD_BETA)
               
            
            if cfg.METAPL.OPEN:
                # and (iteration+1) > cfg.METAPL.VAL_UPDATE 
                # meta-pseudo labeling
                cls_meta.load_state_dict(classifier_his.state_dict())
                cls_meta.train()
                inner_optimiser  = torch.optim.SGD(cls_meta.parameters(), lr= 0.1)
                para_w = torch.ones_like(psd_label).float().cuda()
                para_w = torch.nn.Parameter(para_w, requires_grad=True)
                optimiser_para_w = torch.optim.Adam([para_w], lr=9.0e-1) 

                
                psd_label = psd_label * (mix_label==255) + mix_label * ((mix_label!=255))
                psd_label = psd_label.long()
                
                ## start inner-loop for calculate w
                optimize_cnt = 1
                for optimize_id in range(optimize_cnt):
                    inner_optimiser.zero_grad()
                    optimiser_para_w.zero_grad()
                    with higher.innerloop_ctx(cls_meta, inner_optimiser, copy_initial_weights=False) as (fcls, diffopt):
                        ####### pesudo optimize
                        virtual_pred = fcls(train_feas.detach(), psd_label.shape[-2:])
                        s_ce_loss = criterion(virtual_pred, psd_label.long()) * para_w 
                        s_ce_loss = s_ce_loss.mean()
                        diffopt.step(s_ce_loss)
                        virtual_psd = virtual_pred.max(1)[1]
                        #uni_psd = torch.unique(virtual_psd)
                        
                        ##### updating the w by unbaised data and meta cls
                        val_cnt = 1
                        val_loss = 0

                        #### mixup resampling 
                        for cnt in range(val_cnt):
                            val_input, val_label = sample_val_images(cfg, psd_label, meta_val='meta_val')
                            val_input = val_input.cuda(non_blocking=True)
                            val_label = val_label.cuda(non_blocking=True).long()              
                            
                            v_size = val_label.shape[-2:]
                            with torch.no_grad():
                                val_feas = feature_extractor(val_input)[1]
                            val_pred = fcls(val_feas.detach(), v_size)
                            val_loss += criterion(val_pred, val_label).mean()
                        (val_loss/val_cnt).backward()
                            
                    optimiser_para_w.step()            
                ## end inner-loop for calculate w
                uc_map_eln = para_w.clone().detach()
                if cfg.TCR.OPEN:
                    uc_map_eln[mix_label!=255] = 1
            else:
                uc_map_eln = torch.ones_like(psd_label).float()

            psd_label = psd_label * (mix_label==255) + mix_label * ((mix_label!=255))
            st_loss = criterion(tgt_pred, psd_label.long())
            pesudo_p_loss = (st_loss * (0.5 + 0.5*(uc_map_eln-1).exp()) ).mean()
            #pesudo_n_loss = binary_ce(tgt_pred.view(m_batchsize*C, 1, height, width), t_neg_label) * 0.5
            st_loss = pesudo_p_loss
            st_loss.backward() 
            
            ### update current model
            optimizer_fea.step()
            optimizer_cls.step()  
        
            ### update history model
            ### eval student perfromance
            if cfg.DTU.DYNAMIC:
                if len(stu_score_buffer) >= cfg.DTU.Query_START and int(len(stu_score_buffer)-cfg.DTU.Query_START) % cfg.DTU.Query_STEP ==0:   
                    all_score = evel_stu(cfg, feature_extractor, classifier, stu_eval_list)
                    compare_res = np.array(all_score) - np.array(stu_score_buffer)
                    if np.mean(compare_res > 0) > 0.5 or len(stu_score_buffer) > cfg.DTU.META_MAX_UPDATE:
                        update_iter = len(stu_score_buffer)

                        cls_his_optimizer.step()
                        fea_his_optimizer.step()
                        
                        res_dict['stu_ori'].append(np.array(stu_score_buffer).mean())
                        res_dict['stu_now'].append(np.array(all_score).mean())
                        res_dict['update_iter'].append(update_iter)
                        
                        df = pd.DataFrame(res_dict)
                        df.to_csv('dyIter_FN.csv')

                        ## reset
                        stu_eval_list = []
                        stu_score_buffer = []

            else:
                if iteration % cfg.DTU.FIX_ITERATION == 0:
                    cls_his_optimizer.step()
                    fea_his_optimizer.step()
           
            ## update
            if cfg.TCR.OPEN:
                tgt_train_data.mix_p = (1-running_conf[tgt_train_data.mix_classes]) / ((1-running_conf[tgt_train_data.mix_classes]).sum() )
            
            meters.update(loss_p_loss=pesudo_p_loss.item())
            
            iteration = iteration + 1

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iters:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters),
                            lr=optimizer_fea.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
                    
            if (iteration == cfg.SOLVER.MAX_ITER or (iteration+1) % (cfg.SOLVER.CHECKPOINT_PERIOD)==0):
                filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(int(iteration)))
                torch.save({'iteration': iteration, 'feature_extractor': feature_extractor_his.state_dict(), 'classifier':classifier_his.state_dict()}, filename)
                run_test(cfg, feature_extractor_his, classifier_his, local_rank, distributed)

            
            ### re-build candidate and dataloader
            if (cfg.TCR.OPEN and (iteration+1) % cfg.TCR.UPDATE_FREQUENCY == 0) or \
                (cfg.METAPL.OPEN and (iteration+1) % cfg.METAPL.VAL_UPDATE == 0):
                run_candidate(cfg, feature_extractor_his, classifier_his, local_rank, distributed, update_meta=start_from_None)
                gen_lb_info(cfg, 'CTR')  ## for mixup 
                gen_lb_info(cfg, 'CTR_O')  ## for rsc 
                tgt_train_data = build_dataset(cfg, mode='train', is_source=False, rsc='CTR_O.p')
                if distributed:
                    tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
                else:
                    tgt_train_sampler = None
                tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True, sampler=tgt_train_sampler, drop_last=True)
                tgt_train_data.mixer.label_to_file, _ = pickle.load(open(os.path.join(cfg.OUTPUT_DIR, 'CTR.p'), "rb"))
                if cfg.METAPL.OPEN:
                    ## meta_val
                    gen_lb_info(cfg, 'meta_val')
                    ## meta_val_mixup
                    gen_lb_info(cfg, 'meta_val_mixup')
        
                break
            
            if iteration == cfg.SOLVER.STOP_ITER:
                break
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (cfg.SOLVER.MAX_ITER)
        )
    )

    return feature_extractor_his, classifier_his          

            


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        RANK = int(os.environ["RANK"])
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            NGPUS_PER_NODE = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        else:
            NGPUS_PER_NODE = torch.cuda.device_count()
        assert NGPUS_PER_NODE > 0, "CUDA is not supported"
        GPU = RANK % NGPUS_PER_NODE
        torch.cuda.set_device(GPU)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='tcp://{}:{}'.format(
                                                 master_address, master_port),
                                             rank=RANK, world_size=WORLD_SIZE)
        NUM_GPUS = WORLD_SIZE
        print(f"RANK and WORLD_SIZE in environ: {RANK}/{WORLD_SIZE}")
        # torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(
        #     backend="nccl", init_method="env://"
        # )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("DTST", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    fea, cls = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, fea, cls, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
