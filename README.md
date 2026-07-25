# DefectLens AI

DefectLens AI is an industrial visual anomaly detection system designed to inspect manufactured products (such as bottles, cables, or screws) and identify defects without having seen labeled defective training examples.

This project implements Student-Teacher Feature Pyramid Matching (STFPM), featuring a frozen pretrained teacher network and a trainable student network that learns to mimic the teacher on normal images. On a defective image, the student and teacher disagree. This disagreement generates both the anomaly score and a visual heatmap showing the defect location.

## Project Structure

- `backend/`: FastAPI application, PyTorch model definitions, training/inference pipelines, and tests.
- `frontend/`: React and Vite-based single-page application using custom styling design tokens.
- `docs/`: Screenshots and system documentation.

## Getting Started

Initial setup begins with the `bottle` category from the MVTec AD dataset.

Detailed setup, training, and running instructions will be added in subsequent modules.
