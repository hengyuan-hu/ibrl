python train_rl.py --config_path release/cfgs/robomimic_rl/can_rft.yaml --pretrain_only 1 --pretrain_num_epoch 5 --load_pretrained_agent None --save_dir release/model/robomimic/can_pretrain

python train_rl.py --config_path release/cfgs/robomimic_rl/square_rft.yaml --pretrain_only 1 --pretrain_num_epoch 10 --load_pretrained_agent None --save_dir release/model/robomimic/square_pretrain
