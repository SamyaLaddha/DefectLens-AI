# DefectLens AI

DefectLens AI is an industrial visual anomaly detection system designed to inspect manufactured products (such as bottles, cables, or screws) and identify defects without having seen labeled defective training examples.

This project implements Student-Teacher Feature Pyramid Matching (STFPM), featuring a frozen pretrained teacher network and a trainable student network that learns to mimic the teacher on normal images. On a defective image, the student and teacher disagree. This disagreement generates both the anomaly score and a visual heatmap showing the defect location.

## Project Structure

- `backend/`: FastAPI application, PyTorch model definitions, training/inference pipelines, and tests.
- `frontend/`: React and Vite-based single-page application using custom styling design tokens.
- `docs/`: Screenshots and system documentation.

## Getting Started

Initial setup begins with the `bottle` category from the MVTec AD dataset.

### 1. Backend Setup
1. Navigate to the `backend/` directory and copy the environment template file to create your local `.env`:
   ```powershell
   cd backend
   copy .env.example .env
   ```
2. Start the FastAPI development server:
   ```powershell
   ..\.venv\Scripts\python -m uvicorn app.main:app --reload
   ```

### 2. Frontend Setup
1. Navigate to the `frontend/` directory and copy the environment template to create a local, untracked `.env` configuration (Vite requires a `.env` file to load environment variables, as it does not read `.env.example` directly):
   ```powershell
   cd frontend
   copy .env.example .env
   ```
2. Start the Vite React development server:
   ```powershell
   npm run dev
   ```
