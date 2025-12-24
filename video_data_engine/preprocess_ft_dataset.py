import argparse
import os
from video_engine import pipeline_process  # 确保 pipeline_process 在 video_engine 模块中

parser = argparse.ArgumentParser(description="Video Data Engine Pipeline")
parser.add_argument(
    '--training_set_csv', type=str, required=True,
    help='Path to the training set CSV file'
)
parser.add_argument(
    '--test_set_csv', type=str, required=True,
    help='Path to the test set CSV file'
)
args = parser.parse_args()

train_csv = args.training_set_csv
test_csv = args.test_set_csv

if not os.path.exists(train_csv):
    raise FileNotFoundError(f"Training CSV not found: {train_csv}")
if not os.path.exists(test_csv):
    raise FileNotFoundError(f"Test CSV not found: {test_csv}")

print(f"[INFO] Processing training set: {train_csv}")
processed_train_csv = pipeline_process(train_csv)
final_train_csv = os.path.join(os.path.dirname(processed_train_csv), "processed_training_set.csv")
os.rename(processed_train_csv, final_train_csv)
print(f"[INFO] Training set finished: {final_train_csv}")

print(f"[INFO] Processing test set: {test_csv}")
processed_test_csv = pipeline_process(test_csv)
final_test_csv = os.path.join(os.path.dirname(processed_test_csv), "processed_test_set.csv")
os.rename(processed_test_csv, final_test_csv)
print(f"[INFO] Test set finished: {final_test_csv}")
