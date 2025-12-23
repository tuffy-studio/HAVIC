"""
make_video_csv.py

Usage:
    # Evaluation mode with multiple labels (only overall_label is required)
    python make_video_csv.py \
        --input_dir /path/to/videos_real \
        --output_csv /path/to/eval_real.csv \
        --mode evaluation \
        --overall_label 1 \
        --audio_label 1 \
        --visual_label 1

    # Inference mode (no labels required)
    python make_video_csv.py \
        --input_dir /path/to/videos \
        --output_csv /path/to/infer.csv \
        --mode inference

Notes:
    1. `input_dir` should contain videos of the same type, as evaluation assumes
       all videos in one folder share the same labels.
    2. For datasets with multiple types or labels, run the script separately for each folder and merge them manually.
    3. Supported video formats: .mp4
    4. All paths in the generated CSV will be absolute paths.
"""

import os
import csv
import argparse

def gather_videos(input_dir):
    """Gather all video files under the input directory."""
    videos = []
    input_dir = os.path.abspath(input_dir)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mp4')):
                videos.append(os.path.abspath(os.path.join(root, file)))
    return videos

def write_csv(output_csv, videos, mode='inference', overall_label=None, audio_label=None, visual_label=None):
    """Write CSV file based on mode and labels."""
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        if mode == 'evaluation':
            if overall_label is None:
                raise ValueError("For evaluation mode, --overall_label must be specified")
            # header depends on whether audio/video labels are provided
            headers = ['video_path', 'overall_label']
            if audio_label is not None:
                headers.append('audio_label')
            if visual_label is not None:
                headers.append('visual_label')
            writer.writerow(headers)

            for video in videos:
                row = [video, overall_label]
                if audio_label is not None:
                    row.append(audio_label)
                if visual_label is not None:
                    row.append(visual_label)
                writer.writerow(row)
        else:  # inference
            writer.writerow(['visual_path'])
            for video in videos:
                writer.writerow([video])
    print(f"CSV file saved to {output_csv}. Mode: {mode}. Total videos: {len(videos)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV for evaluation or inference")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing videos")
    parser.add_argument('--output_csv', type=str, required=True, help="Output CSV file path")
    parser.add_argument('--mode', type=str, required=True, choices=['evaluation','inference'], default='inference',
                        help="Mode: 'evaluation' or 'inference'")
    parser.add_argument('--overall_label', type=str, required=True, choices=['0','1'], default=None,
                        help="Overall label (0: real, 1: fake). Required in evaluation mode.")
    parser.add_argument('--audio_label', type=str, choices=['0','1'], default=None,
                        help="Optional audio label for evaluation")
    parser.add_argument('--visual_label', type=str, choices=['0','1'], default=None,
                        help="Optional visual label for evaluation")
    args = parser.parse_args()

    videos = gather_videos(args.input_dir)
    write_csv(
        args.output_csv, videos,
        mode=args.mode,
        overall_label=args.overall_label,
        audio_label=args.audio_label,
        visual_label=args.visual_label
    )
