import os
import argparse
from video_engine import collect_videos_to_csv, pipeline_process

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Data Engine Pipeline")
    parser.add_argument('--train_set_dir', type=str, required=True, help='the root directory of training set')
    parser.add_argument('--test_set_dir', type=str, required=True, help='the root directory of test set')
    parser.add_argument('--save_dir', type=str, default="", help='the root directory to save processed videos and csvs')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # collect videos from the specified directories and save to CSV files
    print(f"Collecting videos from directory: {args.train_set_dir} and {args.test_set_dir}")
    if not os.path.exists(args.train_set_dir) or not os.path.exists(args.test_set_dir):
        print(f"Error: One or both directories do not exist.")
        exit(1)

    train_csv = collect_videos_to_csv(root_dir=args.train_set_dir, output_csv=os.path.join(args.save_dir, f"training_set.csv"))
    test_csv = collect_videos_to_csv(root_dir=args.test_set_dir, output_csv=os.path.join(args.save_dir, f"test_set.csv"))
    print(f"Video collection finished, csv saved at: {train_csv}, {test_csv}")

    print(f"Processing training set: {train_csv}")
    matched_train_csv = pipeline_process(csv_path=train_csv)
    print(f"Training Video preprocess pipeline finished, the final csv saved at: {matched_train_csv}")

    print(f"Processing test set: {test_csv}")
    matched_test_csv = pipeline_process(csv_path=test_csv)
    print(f"Test Video preprocess pipeline finished, the final csv saved at: {matched_test_csv}")