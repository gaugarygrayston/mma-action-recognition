# Video Movement Classification using Pose-Based Features

This repository demonstrates a pose-based video movement classification workflow. It includes labeled frame datasets, feature extraction utilities, model training notebooks, and scripts for annotating frames and building output videos.



https://github.com/user-attachments/assets/0a3c2d56-03e2-4228-9b04-a0057fe16a6d



## Repository structure

- `research_paper` — full research paper documenting methodology, results, and findings.
- `01_feature_extraction.ipynb` — extract pose features from labeled frame folders and create a clean CSV dataset.
- `02_random_forest_baseline.ipynb` — train and evaluate a Random Forest baseline on extracted pose features.
- `03_move_classifier_lstm_vs_mlp.ipynb` — train and compare an MLP and an LSTM sequence model.
- `04_annotate_frames.py` — annotate frame images with predicted movement labels using a saved model.
- `05_frames_to_video.py` — assemble ordered image frames into a video file.
- `src/extraction.py` — shared utilities for SSD person detection, MediaPipe pose extraction, and feature row assembly.
- `data/` — labeled frame datasets, extracted CSV datasets, and label mapping.
- `models/` — saved TensorFlow/Keras models used by the annotation script.
- `results/` — saved model reports, confusion matrices, annotated frames, and generated videos.
- `ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/` — SSD person detection model files.
- `config/pipeline.config` — model configuration used by the detection pipeline.

## Key files

- `research_paper` — academic paper: "From Octagon to Algorithm: A Knockout Approach to Building the Ultimate Fighting Classifier for Combat Sports Action Recognition"
- `data/extracted_video_data_clean.csv` — cleaned feature dataset used for model training.
- `data/class_dictionary.txt` — class label mapping for simplifying movement labels.
- `models/lstm_model.h5` — saved LSTM sequence model (93% accuracy on test set).
- `models/mlp_model.h5` — saved feedforward MLP model (41% accuracy on test set).

## Setup

1. Create a new virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Recommended Python version: `3.12.x`.

## Workflow

1. Run `01_feature_extraction.ipynb` to extract pose-based features from `data/sorted_frames_*` and save a cleaned CSV dataset.
2. Run `02_random_forest_baseline.ipynb` to train and validate a Random Forest baseline model.
3. Run `03_move_classifier_lstm_vs_mlp.ipynb` to train and compare MLP and LSTM models, then generate classification reports and plots.
4. Use `04_annotate_frames.py` to label and save annotated frames from a frame dataset.
5. Use `05_frames_to_video.py` to convert annotated image frames into a playable video.

## Running the annotation and video scripts

Annotate frames with the saved LSTM model:
```bash
python 04_annotate_frames.py
```

Build a video from a directory of labeled frames:
```bash
python 05_frames_to_video.py --input-dir data/labeled_frames_a13 --output-file results/labeled_frames_a13.mp4 --fps 20
```

## Offsite data & sample outputs

The full training dataset and large sample output videos are hosted offsite and are not included in this repository to keep the repo size small. 

```
Training dataset (frames): (https://www.kaggle.com/datasets/garygau/mma-sparring-training-data)
```

The repository `.gitignore` excludes `data/` and `results/` by default so that large artifacts are not pushed. If you prefer to keep a small curated sample dataset in the repo, create `data/sample_frames/` and commit only that folder.

## Results

- `results/classification_report_mlp.txt`
- `results/classification_report_lstm.txt`
- `results/confusion_matrix_mlp.png`
- `results/confusion_matrix_lstm.png`
- `results/loss_plot_mlp.png`
- `results/accuracy_plot_lstm.png`
