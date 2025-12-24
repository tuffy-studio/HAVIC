"""
Split FakeAVCeleb dataset into training and test sets
and generate audio/visual/overall labels.

The dataset is randomly split into:
    - 70% training
    - 30% test

Outputs:
    - training_set.csv
    - test_set.csv

Each CSV contains:
    video_path, audio_label, visual_label, overall_label
"""

import argparse
import random
import csv
from pathlib import Path


# =========================
# Fixed configuration
# =========================
TRAIN_RATIO = 0.7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split FakeAVCeleb dataset into training and test sets"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the FakeAVCeleb dataset root directory",
    )
    return parser.parse_args()


def collect_video_paths(dataset_root):
    """
    Recursively collect all .mp4 video paths under dataset_root.
    """
    dataset_root = Path(dataset_root)
    return sorted(dataset_root.rglob("*.mp4"))


def infer_labels_from_path(video_path: Path):
    """
    Infer audio, visual, and overall labels from the video path.

    Label rule:
        RealVideo-RealAudio  -> (0, 0, 0)
        RealVideo-FakeAudio  -> (0, 1, 1)
        FakeVideo-RealAudio  -> (1, 0, 1)
        FakeVideo-FakeAudio  -> (1, 1, 1)
    """
    path_str = str(video_path)

    if "FakeVideo-FakeAudio" in path_str:
        visual_label = 1
        audio_label = 1
    elif "FakeVideo-RealAudio" in path_str:
        visual_label = 1
        audio_label = 0
    elif "RealVideo-FakeAudio" in path_str:
        visual_label = 0
        audio_label = 1
    elif "RealVideo-RealAudio" in path_str:
        visual_label = 0
        audio_label = 0
    else:
        raise ValueError(f"Cannot infer labels from path: {video_path}")

    overall_label = int(visual_label or audio_label)
    return audio_label, visual_label, overall_label


def save_csv(video_paths, save_path):
    """
    Save video paths and labels to CSV.
    """
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["video_path", "audio_label", "visual_label", "overall_label"]
        )

        for vp in video_paths:
            audio_label, visual_label, overall_label = infer_labels_from_path(vp)
            writer.writerow(
                [str(vp), audio_label, visual_label, overall_label]
            )


def main():
    args = parse_args()

    print(f"[INFO] Scanning dataset root: {args.dataset_root}")
    video_paths = collect_video_paths(args.dataset_root)
    num_videos = len(video_paths)

    if num_videos == 0:
        raise RuntimeError("No video files (.mp4) found in dataset_root!")

    print(f"[INFO] Found {num_videos} videos")

    # Shuffle and split
    random.shuffle(video_paths)
    train_size = int(num_videos * TRAIN_RATIO)

    train_videos = video_paths[:train_size]
    test_videos = video_paths[train_size:]

    print(f"[INFO] Training videos: {len(train_videos)}")
    print(f"[INFO] Test videos: {len(test_videos)}")

    save_csv(train_videos, "training_set.csv")
    save_csv(test_videos, "test_set.csv")

    print("[INFO] Saved training_set.csv and test_set.csv")
    print("[INFO] Dataset split completed successfully.")


if __name__ == "__main__":
    main()
