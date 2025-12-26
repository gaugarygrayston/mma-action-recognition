# MMA Action Recognition with Pose Estimation and LSTMs

## Overview
This project is a master's capstone focused on classifying MMA actions from sparring footage using pose estimation and sequence modeling. A custom annotated dataset was created from MMA sparring videos and used to train an LSTM-based classifier.

The final model achieves **97% classification accuracy** on held-out test data.

*I currently only have my research paper and an AI generated slide deck of my research paper on here so far, the code for this project can be found in the appendix of my research paper. I am still working on putting the code on here too.

## Technical Summary
- Pose estimation on video frames
- Temporal sequence modeling using LSTMs
- Custom-built annotated MMA dataset
- End-to-end training and evaluation pipeline

## Documents
- 📄 Research Paper: docs/research_paper.pdf
- 📊 Slides (AI generated): docs/slide_deck_ai_generated.pdf
- 📊 Slides (not AI generated): https://drive.google.com/file/d/1lfKZhEtxtDc1Aw6Yw0uBXWvJZ7I_OJcZ/view?usp=drive_link

## Technologies
- Python
- NumPy / Pandas
- Pose Estimation (e.g. MediaPipe / OpenPose)
- TensorFlow or PyTorch
- LSTM Networks
