# Review 1 Completion & Master Evidence Summary

This document summarizes the complete Review 1 deliverables and verification evidence for all 5 team roles in the **Real-Time Indian Sign Language Recognition Project**.

---

## 1. Member 1: Static Sign Recognition

### Tasks Completed
- **Dataset Selection**: Confirmed RealSign static ISL dataset with A–Z alphabet structure.
- **Dataset Analysis**: Analyzed 25,975 initial images across 26 classes.
- **Data Quality & Cleaning**:
  - Removed 2 corrupted image files from Testing set.
  - Performed hash-based cross-split leakage checks (`isl_leakage_check.py`).
  - Removed 685 train-leak images (`isl_remove_train_leaks.py`).
  - Removed 52 validation-leak images (`isl_remove_val_leaks.py`).
  - **Verified Final Dataset**: 17,513 Training images, 2,527 Validation images, 5,198 Testing images (**Total: 25,238 clean images, 0 cross-split leakage**).
- **Model Selection & Preprocessing Prototype**: Documented selection of MobileNetV2 transfer learning model and verified 224x224 RGB image resizing and [0.0, 1.0] pixel normalization.

---

## 2. Member 2: Dynamic Sign Recognition

### Tasks Completed
- **Dataset Selection & Video Analysis**: Analyzed 61 dynamic sign video samples (`Sample Videos/`).
- **Metadata Breakdown**:
  - All videos recorded at **30.0 FPS**.
  - Resolutions: `640x720`, `880x720`, `760x720`.
  - Frame counts: Range **111 – 298 frames** (Mean: 166.4 frames).
  - Video Durations: **3.7s – 9.93s** (Mean: 5.55s).
- **Sequence Sampling Strategy**: Established uniform temporal sampling of **16 frames per video sequence**, producing a sequence tensor contract of `(16, 224, 224, 3)`.
- **Report & Sample Extraction**: Generated `dynamic_dataset_report.md` and saved sample extracted sequence frames in `analysis_output/`.

---

## 3. Member 3: Data Pipeline & Preprocessing

### Tasks Completed
- **Data Contracts (`data_pipeline/contracts.py`)**: Defined validation functions for image tensors `(H, W, C)` and sequence tensors `(T, H, W, C)`.
- **Static Preprocessing (`data_pipeline/static_preprocessing.py`)**: Implemented `preprocess_image(image) -> tensor` handling file path or array, resizing to 224x224, and normalizing to [0.0, 1.0].
- **Dynamic Preprocessing (`data_pipeline/dynamic_preprocessing.py`)**: Implemented `prepare_sequence(video_or_frames) -> sequence_tensor` with video decoding, uniform 16-frame sampling, resizing, and normalization.

---

## 4. Member 4: Model Architecture & RAG Module

### Tasks Completed
- **Architecture Documentation (`docs/architecture_spec.md`)**:
  - **Static Branch**: MobileNetV2 backbone (224x224x3 input -> Global Average Pooling -> Dropout(0.2) -> Dense 256 -> 26 Softmax classes).
  - **Dynamic Branch**: Spatial CNN backbone (frame -> 512-dim vector) + 2-layer Bidirectional LSTM / GRU (128 hidden units) -> Dynamic Sign Softmax.
- **RAG / LLM Module Design**: Defined RAG pipeline specification ingesting recognized ISL keyword sequences, performing vector search over ISL sign-phrase pairs, and passing context to an LLM generator to output natural, grammatically correct English.

---

## 5. Member 5: Integration & Backend

### Tasks Completed
- **Repository Organization**: Established modular structure (`data_pipeline/`, `docs/`, `integration/`).
- **Dependencies & Git Setup**: Created `requirements.txt` and `.gitignore`.
- **Mock Recognizers**: Implemented `MockStaticRecognizer` (`integration/mock_static.py`) and `MockDynamicRecognizer` (`integration/mock_dynamic.py`) implementing `run_static(frame)` and `run_dynamic(frame_buffer)` interface contracts.
- **Integration Test Suite**: Created and passed `integration_tests.py` testing static preprocessing, dynamic sequence sampling, mock static prediction, mock dynamic prediction, and real video file execution.

---

## 🏁 Acceptance Checkpoints Verification Summary

| Checkpoint | Status |
| :--- | :---: |
| Each member has a clearly defined scope & SRS handoff | **PASS** |
| Research / dataset statistics / prototype evidence available | **PASS** |
| Project structure, requirements, and .gitignore ready | **PASS** |
| Architecture & module interface contracts documented | **PASS** |
| Automated integration tests run and pass cleanly | **PASS** |
