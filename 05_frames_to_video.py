"""Build a video from chronologically numbered labeled frame images.

This script loads all supported image files from an input folder, sorts them by
numeric index found in the filename, and writes them to a video file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2

FRAME_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input folder, output file, and frame rate."""
    parser = argparse.ArgumentParser(
        description='Build a video from chronologically numbered labeled frames.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'data' / 'labeled_frames_a13',
        help='Root folder containing labeled frame images.',
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path(__file__).resolve().parent / 'results' / 'labeled_frames_a13.mp4',
        help='Output video file path.',
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=20.0,
        help='Frame rate for the output video.',
    )
    return parser.parse_args()


def numeric_key(path: Path) -> tuple[int, str]:
    """Generate a sort key for frame names with numeric suffixes."""
    name = path.stem
    digits = re.findall(r'(\d+)', name)
    if digits:
        return int(digits[-1]), name
    return float('inf'), name


def get_image_paths(input_dir: Path) -> list[Path]:
    """Return all supported image paths from the input directory sorted by filename."""
    image_paths = [
        path
        for path in sorted(input_dir.rglob('*'))
        if path.suffix.lower() in FRAME_EXTENSIONS and path.is_file()
    ]
    image_paths.sort(key=numeric_key)
    return image_paths


def get_fourcc(output_file: Path) -> int:
    """Choose an appropriate codec based on the output file extension."""
    ext = output_file.suffix.lower()
    if ext == '.mp4':
        return cv2.VideoWriter_fourcc(*'mp4v')
    if ext == '.avi':
        return cv2.VideoWriter_fourcc(*'XVID')
    if ext == '.mov':
        return cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter_fourcc(*'mp4v')


def main() -> None:
    """Assemble frames into a video file and write it to disk."""
    args = parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f'Input directory not found: {args.input_dir}')

    image_paths = get_image_paths(args.input_dir)
    if not image_paths:
        raise SystemExit(f'No image frames found under {args.input_dir}')

    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise SystemExit(f'Could not read first frame: {image_paths[0]}')

    height, width = first_image.shape[:2]
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = get_fourcc(args.output_file)
    video = cv2.VideoWriter(str(args.output_file), fourcc, args.fps, (width, height))
    if not video.isOpened():
        raise SystemExit(f'Failed to open video writer for {args.output_file}')

    print(f'Writing {len(image_paths)} frames to {args.output_file} @ {args.fps} FPS')

    for idx, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise SystemExit(f'Could not read image: {image_path}')
        if frame.shape[:2] != (height, width):
            raise SystemExit(
                f'Inconsistent frame size at {image_path}: expected {(width, height)}, got {frame.shape[1], frame.shape[0]}'
            )
        video.write(frame)
        if idx % 100 == 0:
            print(f'  wrote {idx} frames...')

    video.release()
    print(f'Video successfully created: {args.output_file}')


if __name__ == '__main__':
    main()
