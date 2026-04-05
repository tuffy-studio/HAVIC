finetune_path="./weights/ft_model.50.pth"
mode=evalution

CUDA_VISIBLE_DEVICES=7 \
python sliding_window_infer.py \
    --csv_file_path \
    --save_csv_path \
    --finetune_path ${finetune_path} \
    --mode ${mode}

