# Data included in updated_workflow

This folder contains cleaned feature datasets, labeled frame folders, and supporting metadata used by the notebooks.

- `extracted_video_data.csv` — extracted pose and movement features used for classifier training.
- `extracted_static_data.csv` — static feature dataset used for the random forest baseline.
- `class_dictionary.txt` — raw-to-simplified label mapping used by the feature pipeline.
- `sorted_frames_a13/`, `sorted_frames_note9/`, `sorted_frames_s10e/` — labeled frame folders used by the extraction notebook.

## Notes

The extraction notebook reads image folders from `data/sorted_frames_*` and writes feature rows to CSV. Raw videos are not required for the current workflow.
