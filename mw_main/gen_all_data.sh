################ for modem ################
python generate_metaworld_dataset.py \
    --num_episodes 3 \
    --save_gifs 5 \
    --output_path ../data/metaworld/Assembly_frame_stack_1_224x224 \
    --env_cfg.env_name Assembly \
    --env_cfg.frame_stack 2 \
    --env_cfg.rl_image_size 224 \
    --env_cfg.end_on_success false \
    --add_modem_format true

python generate_metaworld_dataset.py \
    --num_episodes 3 \
    --save_gifs 5 \
    --output_path ../data/metaworld/StickPull_frame_stack_1_224x224 \
    --env_cfg.env_name StickPull \
    --env_cfg.frame_stack 2 \
    --env_cfg.rl_image_size 224 \
    --env_cfg.end_on_success false \
    --add_modem_format true

python generate_metaworld_dataset.py \
    --num_episodes 3 \
    --save_gifs 5 \
    --output_path ../data/metaworld/BoxClose_frame_stack_1_224x224 \
    --env_cfg.env_name BoxClose \
    --env_cfg.frame_stack 2 \
    --env_cfg.rl_image_size 224 \
    --env_cfg.end_on_success false \
    --add_modem_format true

python generate_metaworld_dataset.py \
    --num_episodes 3 \
    --save_gifs 5 \
    --output_path ../data/metaworld/CoffeePush_frame_stack_1_224x224 \
    --env_cfg.env_name CoffeePush \
    --env_cfg.frame_stack 2 \
    --env_cfg.rl_image_size 224 \
    --env_cfg.end_on_success false \
    --add_modem_format true


################ for our method ################
python generate_metaworld_dataset.py \
    --num_episodes 5 \
    --save_gifs 5 \
    --output_path ../data/metaworld/CoffeePush_frame_stack_1_96x96_end_on_success \
    --env_cfg.env_name CoffeePush \
    --env_cfg.frame_stack 1 \
    --env_cfg.rl_image_size 96 \
    --env_cfg.end_on_success true

python generate_metaworld_dataset.py \
    --num_episodes 5 \
    --save_gifs 5 \
    --output_path ../data/metaworld/Assembly_frame_stack_1_96x96_end_on_success \
    --env_cfg.env_name Assembly \
    --env_cfg.frame_stack 1 \
    --env_cfg.rl_image_size 96 \
    --env_cfg.end_on_success true

python generate_metaworld_dataset.py \
    --num_episodes 5 \
    --save_gifs 5 \
    --output_path ../data/metaworld/StickPull_frame_stack_1_96x96_end_on_success \
    --env_cfg.env_name StickPull \
    --env_cfg.frame_stack 1 \
    --env_cfg.rl_image_size 96 \
    --env_cfg.end_on_success true

python generate_metaworld_dataset.py \
    --num_episodes 5 \
    --save_gifs 5 \
    --output_path ../data/metaworld/BoxClose_frame_stack_1_96x96_end_on_success \
    --env_cfg.env_name BoxClose \
    --env_cfg.frame_stack 1 \
    --env_cfg.rl_image_size 96 \
    --env_cfg.end_on_success true
