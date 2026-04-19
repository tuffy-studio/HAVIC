finetune_path="../weights/ft_model.50.pth"
csv_file_path=" " # TODO
save_csv_path="./swi_result.csv"
mode="evaluation" # Set to "evaluation" for evaluation mode, or "inference" for inference mode. In evaluation mode, the script will compute metrics. In inference mode, it will only generate predictions without computing metrics.
verbose=True # Set to True to print detailed logs during processing. Adjust based on your needs.

window_size_frames=16 # Number of frames per window. 5fps * 3.2 seconds = 16 frames per window. 
window_stride_frames=2 # Number of frames to stride the window. 5fps * 0.4 seconds = 2 frames stride.
max_time=10 # "Max processing time (seconds) for each video. Adjust based on your needs and computational resources."
max_workers=2 # "Number of processes to use for parallel processing. Adjust based on your CPU cores and memory."

cmd="python sliding_window_infer.py \
    --csv_file_path ${csv_file_path} \
    --save_csv_path ${save_csv_path} \
    --finetune_path ${finetune_path} \
    --mode ${mode} \
    --window_size_frames ${window_size_frames} \
    --window_stride_frames ${window_stride_frames} \
    --max_time ${max_time} \
    --max_workers ${max_workers}"

# 控制 verbose
if [ "$verbose" = "True" ]; then
    cmd="${cmd} --verbose"
fi

CUDA_VISIBLE_DEVICES=5 $cmd