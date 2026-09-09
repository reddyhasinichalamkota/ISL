# Real-Time Indian Sign Language Recognition System

A deep learning platform for **real-time Indian Sign Language (ISL) recognition** supporting both **static sign language (A–Z alphabets)** and **dynamic sign language (words and phrases)** with spatial-temporal deep learning and LLM/RAG natural language post-processing.

---

## 👥 Team Roles & Responsibilities

| Role | Responsibility | Review 1 Deliverable |
| :--- | :--- | :--- |
| **Member 1** | Static Sign Recognition | Dataset analysis, data quality audit, leakage removal (25,238 clean images), MobileNetV2 selection. |
| **Member 2** | Dynamic Sign Recognition | Video dataset analysis (61 videos, 30 FPS, frame counts), sampling strategy (16 frames), sequence contract. |
| **Member 3** | Data Pipeline & Preprocessing | Reusable data contracts (`preprocess_image`, `prepare_sequence`), uniform sampling, tensor normalization. |
| **Member 4** | Model Architecture & RAG | Spatial CNN backbone, temporal LSTM/GRU architecture design, RAG grammar & translation pipeline spec. |
| **Member 5** | Integration & Backend | Repository setup, modular directory structure, `.gitignore`, `requirements.txt`, mock modules, integration test suite. |

---

## 📁 Repository Structure

```
d:/ISL/
├── Planner/                       # SRS handoffs, review workflows & master plan
├── Dataset/                       # Static ISL image dataset (Training, Testing, Validation)
├── Sample Videos/                 # Dynamic ISL video dataset (61 MP4 videos)
├── analysis_output/               # Output visual inspection frame samples
├── data_pipeline/                 # Reusable Data Preprocessing Package
│   ├── contracts.py               # Data shape & type validation contracts
│   ├── static_preprocessing.py    # Static image resizing (224x224) & normalization
│   └── dynamic_preprocessing.py   # Dynamic video decoding & uniform sequence sampling (T=16)
├── docs/                          # Architecture & system specifications
│   └── architecture_spec.md       # CNN, LSTM/GRU, and RAG architectural specification
├── integration/                   # Integration & Mock Module Package
│   ├── mock_static.py             # Mock static recognizer (run_static)
│   └── mock_dynamic.py            # Mock dynamic recognizer (run_dynamic)
├── isl_dataset_analysis.py        # Static dataset statistics script
├── isl_leakage_check.py           # Cross-split duplicate check script
├── isl_remove_train_leaks.py      # Training set leak removal script
├── isl_remove_val_leaks.py        # Validation set leak removal script
├── dynamic_analysis.py            # Dynamic video dataset analysis script
├── integration_tests.py           # Review 1 Automated Integration Test Suite
├── read.md                        # Member 1 detailed static audit log
├── writeup.md                     # Data cleaning & leakage verification writeup
├── dynamic_dataset_report.md      # Member 2 dynamic video dataset report
├── review1_summary.md             # Review 1 master evidence summary
├── requirements.txt               # Dependency configuration file
└── .gitignore                     # Git tracking exclusion filters
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites & Environment Setup
Activate the virtual environment:
```powershell
.\tensorflow-env\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Review 1 Dynamic Analysis
```powershell
python dynamic_analysis.py
```

### 4. Run Review 1 Integration Tests
```powershell
python integration_tests.py
```

---

## ✅ Review 1 Verification Status

- [x] Static ISL dataset cleaned (0 cross-split leakage verified)
- [x] Dynamic video dataset analyzed (61 videos, 30 FPS, frame counts documented)
- [x] Preprocessing pipelines built (`preprocess_image` & `prepare_sequence`)
- [x] Architecture & RAG documentation completed (`docs/architecture_spec.md`)
- [x] Modular repository & integration test suite passing (`integration_tests.py`)
