pretrain_path="./weights/ft_model.50.pth"


CUDA_VISIBLE_DEVICES=7 \
python sliding_window_infer.py \
    --csv_file_path \
    --save_csv_path \
    --pretrain_path ${pretrain_path} \

