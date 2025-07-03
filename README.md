python main.py --cfg_path configs/realsr_swinunet_material256.yaml --save_dir ckpts/task1_try

python inference_resshift.py -i /nfs5/xfy/test_images/lq -o samples_image/pred --task realsr --scale 4 --version v1 --ckptn 300K
