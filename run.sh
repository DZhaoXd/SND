
# G2C
# warm
CUDA_VISIBLE_DEVICES=0 nohup python train_meta_query.py -cfg configs/deeplabv2_r101_dtst_G2C.yaml OUTPUT_DIR results/G2C_SND_WARM/ resume pretrain/G2C_model_iter020000.pth > logs/G2C_SND_WARM.file 2>&1 &
cp -r results/G2C_SND_WARM/ results/G2C_SND/
# adaptation
CUDA_VISIBLE_DEVICES=0 nohup python train_meta_query.py -cfg configs/deeplabv2_r101_SND_G2C.yaml OUTPUT_DIR results/G2C_SND/ resume pretrain/G2C_model_iter020000.pth > logs/G2C_SND.file 2>&1 &
# DTU
CUDA_VISIBLE_DEVICES=3 nohup python train_SND_DTU.py -cfg configs/deeplabv2_r101_SND_G2C_Full.yaml OUTPUT_DIR results/G2C_SND_DTU/ resume pretrain/G2C_model_iter020000.pth > logs/G2C_SND_DTU.file 2>&1 &



## G2S

# warm
CUDA_VISIBLE_DEVICES=3 nohup python train_meta_query.py -cfg configs/deeplabv2_r101_dtst_G2S.yaml OUTPUT_DIR results/G2S_SND_WARM_8K resume ../repo/DT-ST-main/results/gta+syn_//model_iter008000.pth > logs/G2S_SND_WARM_8K.file 2>&1 &
cp -r results/G2S_SND_WARM_8K/ results/G2S_SND
# adaptation
CUDA_VISIBLE_DEVICES=1 nohup python train_meta_query.py -cfg configs/deeplabv2_r101_SND_G2S.yaml OUTPUT_DIR results/G2S_SND resume ../repo/DT-ST-main/results/gta+syn_//model_iter008000.pth > logs/G2S_SND.file 2>&1 &
 CUDA_VISIBLE_DEVICES=0 nohup python train_meta_query.py -cfg configs/deeplabv2_r101_SND_G2S_Full.yaml OUTPUT_DIR results/G2S_SND_DTU/ resume ../repo/DT-ST-main/results/gta+syn_//model_iter008000.pth > logs/G2S_SND_DTU 2>&1 &



 CUDA_VISIBLE_DEVICES=5 nohup  python train_static.py -cfg configs/deeplabv2_r101_dtst_G2S_static_12.yaml OUTPUT_DIR results/synthia_SND_static_12/ resume ../repo/DT-ST-main/results/gta+syn_//model_iter008000.pth > logs/BDD_SND_static_12.file 2>&1 &
 CUDA_VISIBLE_DEVICES=5 nohup  python train_static.py -cfg configs/deeplabv2_r101_dtst_G2S_static_24.yaml OUTPUT_DIR results/synthia_SND_static_24/ resume ../repo/DT-ST-main/results/gta+syn_//model_iter008000.pth > logs/BDD_SND_static_24.file 2>&1 &

 CUDA_VISIBLE_DEVICES=1 nohup  python train_static.py -cfg configs/deeplabv2_r101_dtst_G2S_static_32.yaml OUTPUT_DIR results/synthia_SND_static_32/ resume ../repo/DT-ST-main/results/gta+syn_//model_iter008000.pth > logs/BDD_SND_static_32.file 2>&1 &
 CUDA_VISIBLE_DEVICES=1 nohup  python train_static.py -cfg configs/deeplabv2_r101_dtst_G2S_static_48.yaml OUTPUT_DIR results/synthia_SND_static_48/ resume ../repo/DT-ST-main/results/gta+syn_//model_iter008000.pth > logs/BDD_SND_static_48.file 2>&1 &




## BDD
CUDA_VISIBLE_DEVICES=6 nohup  python train_meta_query.py -cfg configs/deeplabv2_r101_dtst_BDD.yaml OUTPUT_DIR results/BDD_SND_WARM/ resume pretrain/G2C_model_iter020000.pth > logs/BDD_SND_WARM.file 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup  python train_static.py -cfg configs/deeplabv2_r101_dtst_BDD_static.yaml OUTPUT_DIR results/BDD_SND_static_12/ resume pretrain/G2C_model_iter020000.pth > logs/BDD_SND_static_12.file 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup  python train_static.py -cfg configs/deeplabv2_r101_dtst_BDD_static.yaml OUTPUT_DIR results/BDD_SND_static_24/ resume pretrain/G2C_model_iter020000.pth > logs/BDD_SND_static_24.file 2>&1 &


# test
CUDA_VISIBLE_DEVICES=1 python test.py -cfg configs/deeplabv2_r101_SND_G2S.yaml resume ./pretrain/S2C_Pretrain_DG.pth 


# test synthia
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_synthia_16.yaml resume ../DTST/results/synthia_HARD_PL_DTST/model_iter010999.pth > logs/eval_synthia 2>&1 &
# test gta
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_gta_19.yaml resume ./results/G2C_SND/model_iter006499.pth > logs/eval_gta5 2>&1 &
# test gta pretrain
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_gta_19.yaml resume ./pretrain/G2C_model_iter020000.pth > logs/eval_gta5_pretrain 2>&1 &
# test synthia pretrain
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_synthia_16.yaml resume ./pretrain/S2C_Pretrain_NO_DG.pth > logs/eval_synthia_pretrain 2>&1 &


#test sd
     CUDA_VISIBLE_DEVICES=1 python test.py --saveres -cfg configs/deeplabv2_r101_SDllama70b.yaml resume ./pretrain/G2C_model_iter020000.pth   

