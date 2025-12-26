# MMA Action Recognition with Pose Estimation and LSTMs

## Overview
This project is a master's capstone focused on classifying MMA actions from sparring footage using pose estimation and sequence modeling. A custom annotated dataset was created from MMA sparring videos and used to train an LSTM-based classifier.

The final model achieves **97% classification accuracy** on held-out test data.

## Project Status

⚠️ **Work in Progress**

At this time, this repository contains:
- The full research paper for this project
- An AI-generated slide deck summarizing the research paper

The full implementation code for this project is currently documented in the **appendix of the research paper**. I am actively working on organizing and migrating the code into this repository for public release.

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
