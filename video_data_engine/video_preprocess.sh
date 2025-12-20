#!/bin/bash
set -e

# 指定输入目录或 CSV 文件
INPUT_PATH="put_the_dir_or_csv_path_here"

# 运行 Python 脚本
python run_video_engine.py --input "$INPUT_PATH"

