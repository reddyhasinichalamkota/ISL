# Real-Time Indian Sign Language Recognition Using Spatial-Temporal Deep Learning and Natural Language Integration

**Abstract**  
Communication barriers between hearing-impaired individuals who use Indian Sign Language (ISL) and the non-signing public pose significant societal challenges. Existing ISL recognition solutions often suffer from data leakage between training and evaluation splits, lack modular real-time architectures, or are restricted solely to static gestures. This research paper presents a unified, spatial-temporal deep learning framework for real-time ISL recognition capable of handling both static alphabet signs (A–Z) and dynamic phrase-level video gestures. We establish a rigorous data hygiene protocol that detected and eliminated 737 cross-split duplicate images from the static dataset, yielding a leak-free benchmark of 25,238 images across 26 classes. For dynamic gestures, we present a 30.0 FPS video dataset characterization and a uniform temporal sampling algorithm ($T = 16$ frames). We detail the multi-branch spatial-temporal architecture combining a MobileNetV2 spatial backbone, a 2-layer Bidirectional LSTM/GRU temporal classifier, and a Retrieval-Augmented Generation (RAG) LLM module for grammar translation. Finally, we provide a modular software engineering architecture with mock interface contracts (`run_static`, `run_dynamic`) and an automated test suite achieving 100% pass rates across all integration modules.

---

## 1. Introduction

### 1.1 Problem Statement
Indian Sign Language (ISL) is a rich, visually complex language spoken by millions of deaf and hard-of-hearing individuals across India. Unlike spoken languages, ISL relies on manual features (hand shape, orientation, movement trajectory) and non-manual features (facial expressions, body posture). The absence of widespread ISL literacy among the general public creates communication barriers in healthcare, education, and governance.

Computer vision and deep learning offer a promising path toward automated real-time ISL recognition. However, practical deployment faces three major hurdles:
1. **Static vs. Dynamic Dichotomy**: Static signs (e.g., individual letters A–Z) require high-spatial-resolution feature extraction, whereas dynamic signs (e.g., words like *"Hello"*, *"Thank you"*, *"Water"*) require spatial-temporal modeling over temporal frame sequences.
2. **Data Leakage in Benchmarks**: Naive random splitting of sign language datasets often places near-identical frame captures from the same subject into both training and validation/testing sets, leading to artificially inflated accuracy metrics.
3. **Unstructured Output & Grammar Disconnect**: Direct classification models output isolated sign tokens (e.g., `["YOU", "NAME", "WHAT"]`), which lack standard English grammatical structure.

### 1.2 Project Scope & Objectives
This project develops a modular, real-time ISL recognition application that accepts live video or frame streams, classifies both static and dynamic gestures, and translates predicted sign sequences into coherent English sentences. 

The core objectives fulfilled in this phase include:
- **Clean Static Benchmark**: Elimination of all cross-split data leakage across 26 alphabet classes.
- **Dynamic Sequence Characterization**: Standardized frame-sequence sampling protocol ($T=16$) for dynamic video processing.
- **Modular Data Pipeline**: Production-grade preprocessing contracts (`preprocess_image`, `prepare_sequence`).
- **Architectural Design**: Dual-branch spatial (MobileNetV2) and spatial-temporal (CNN + Bi-LSTM/GRU) models paired with a RAG/LLM post-processing layer.
- **Integration Framework**: Production-ready repository structure, mock execution interfaces, and automated unit/integration testing.

---

## 2. Team Work Distribution & Review Roadmap

To ensure parallel development and systematic verification, the project is structured around a 5-member team workflow across 3 review milestones:

```
+-----------------------------------------------------------------------------------+
|                            ISL PROJECT TEAM WORKFLOW                              |
+-------------------+---------------------------------------------------------------+
| Member / Role     | Core Responsibility & Scope                                   |
+-------------------+---------------------------------------------------------------+
| Member 1: Static  | Static dataset analysis, data quality audit, leakage removal,  |
|                   | MobileNetV2 baseline CNN training & evaluation.                |
| Member 2: Dynamic | Dynamic video dataset analysis, frame extraction, uniform     |
|                   | temporal sequence sampling, CNN + LSTM/GRU modeling.          |
| Member 3: Pipeline| Reusable data preprocessing contracts (`preprocess_image`,    |
|                   | `prepare_sequence`), dataset loaders, tensor normalization.   |
| Member 4: Arch/RAG| Spatial & temporal model architecture design, loss functions, |
|                   | RAG / LLM contextual grammar translation pipeline.             |
| Member 5: Backend | Repository structure, requirements, `.gitignore`, mock        |
|                   | contracts (`run_static`, `run_dynamic`), integration tests.   |
+-------------------+---------------------------------------------------------------+
```

---

## 3. Dataset Analysis & Data Hygiene Engineering

### 3.1 Static ISL Dataset Analysis (Member 1)
The static recognition task uses the **RealSign ISL Dataset**, representing 26 alphabet classes (A–Z). The initial raw dataset distribution comprised 25,975 total images:

$$\text{Initial Dataset}: \quad \text{Train} = 18,198, \quad \text{Val} = 2,579, \quad \text{Test} = 5,198$$

#### Data Quality Audit & Corrupted File Removal
During initial inspection, 2 corrupted image files were identified in the test partition and removed. Pixel intensity range was confirmed to be standard 8-bit RGB ($[0, 255]$), with variable spatial resolutions requiring standardization to $224 \times 224 \times 3$.

#### Cross-Split Data Leakage Detection & Removal
Data leakage occurs when identical or duplicate samples appear across training and test/validation splits, corrupting model evaluation. A cryptographic MD5 image hashing pipeline (`isl_leakage_check.py`) was executed across all splits.

```
                  +-----------------------------------+
                  |   Raw Dataset (25,975 Images)     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  MD5 Cross-Split Leak Check 1     |
                  |  Identified 648 leaked image groups|
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Train Cleanup: Removed 685 leaks |
                  |  (Train: 18,198 -> 17,513)        |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  MD5 Cross-Split Leak Check 2     |
                  |  Identified 51 Test-Val leaks     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Val Cleanup: Removed 52 leaks    |
                  |  (Val: 2,579 -> 2,527)            |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Final Clean Dataset: 25,238      |
                  |  VERIFIED 0 DATA LEAKAGE          |
                  +-----------------------------------+
```

#### Final Clean Static Dataset Statistics

| Split Name | Initial Count | Leakage Removed | Final Clean Count | Class Representation | Leakage Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training** | 18,198 | 685 | **17,513** | 26 (A–Z) | Clean |
| **Validation** | 2,579 | 52 | **2,527** | 26 (A–Z) | Clean |
| **Testing** | 5,198 | 0 | **5,198** | 26 (A–Z) | Untouched Holdout |
| **Total** | **25,975** | **737** | **25,238** | **26 Classes** | **0 Leaks Verified** |

### 3.2 Dynamic ISL Video Dataset Characterization (Member 2)
Dynamic gesture recognition requires evaluating spatial movements over time. An empirical analysis of 61 dynamic ISL gesture video samples (`Sample Videos/`) was executed using `dynamic_analysis.py`.

#### Empirical Video Metadata Statistics

```
+-------------------------------------------------------------------+
|               DYNAMIC VIDEO DATASET CHARACTERISTICS               |
+-----------------------------------+-------------------------------+
| Metric                            | Empirical Value               |
+-----------------------------------+-------------------------------+
| Total Analyzed Videos             | 61 MP4 Files                  |
| Unique Gesture Classes            | 61 Distinct Dynamic Signs     |
| Video Frame Rate (FPS)            | Constant 30.0 FPS             |
| Spatial Resolutions               | 640x720, 880x720, 760x720     |
| Frame Count per Video (Min/Max)   | 111 / 298 Frames              |
| Mean Frame Count                  | 166.4 Frames                  |
| Video Duration (Min/Max)          | 3.70s / 9.93s                 |
| Mean Video Duration               | 5.55 Seconds                  |
+-----------------------------------+-------------------------------+
```

---

## 4. Preprocessing & Data Pipeline Architecture (Member 3)

The data pipeline package (`data_pipeline/`) establishes strict, reusable input/output mathematical contracts for both static and dynamic recognition modules.

```
                    +------------------------------------+
                    |        Raw Input Source            |
                    | (Image File / BGR Array / MP4)     |
                    +-----------------+------------------+
                                      |
             +------------------------+------------------------+
             |                                                 |
             v                                                 v
+--------------------------+                      +--------------------------+
| static_preprocessing.py  |                      | dynamic_preprocessing.py |
+--------------------------+                      +--------------------------+
| 1. BGR to RGB Conversion |                      | 1. OpenCV Video Capture  |
| 2. Resize to (224, 224)  |                      | 2. Uniform Sampling T=16 |
| 3. Rescale [0.0, 1.0]    |                      | 3. Per-frame Resize 224  |
+------------+-------------+                      | 4. Rescale [0.0, 1.0]    |
             |                                    +------------+-------------+
             v                                                 |
+--------------------------+                                   v
| Tensor: (224, 224, 3)    |                      +--------------------------+
+--------------------------+                      | Tensor: (16, 224, 224, 3)|
                                                  +--------------------------+
```

### 4.1 Static Preprocessing Contract
The function `preprocess_image(image, target_size=(224, 224), normalize=True)` performs:
1. **Format Validation**: Converts file path or raw array to 3-channel RGB image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$.
2. **Bilinear Spatial Resizing**: Maps spatial dimensions to $224 \times 224$.
3. **Min-Max Normalization**: Scales pixel intensities:

$$\mathbf{I}_{\text{norm}} = \frac{\mathbf{I}_{\text{resized}}}{255.0} \in [0.0, 1.0]^{224 \times 224 \times 3}$$

### 4.2 Dynamic Uniform Sequence Sampling Algorithm
Because video durations vary between 3.7s and 9.9s ($111$ to $298$ frames), feeding variable-length frame sequences directly into deep learning models causes batching inefficiency. 

We implement a **Uniform Temporal Sampling Algorithm** that maps any video of $N$ total frames to a fixed sequence length $T = 16$:

$$\text{Index}(i) = \left\lfloor i \cdot \frac{N - 1}{T - 1} \right\rfloor \quad \text{for } i = 0, 1, \dots, T-1$$

This produces a standardized sequence tensor:

$$\mathbf{S} \in \mathbb{R}^{16 \times 224 \times 224 \times 3}$$

---

## 5. Model Architecture & RAG Specification (Member 4)

### 5.1 Static Recognition Architecture (MobileNetV2 Backbone)
For static sign classification (26 classes), we employ a MobileNetV2 backbone pre-trained on ImageNet. MobileNetV2 uses depthwise separable convolutions to achieve high spatial feature representation with low parameter latency.

```
Input Frame (224x224x3)
       │
       ▼
┌───────────────────────────────┐
│ MobileNetV2 Feature Extractor │  ──► Inverted Residual Blocks (Depthwise Conv)
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ Global Average Pooling (1280) │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ Dropout Rate (0.2)            │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ Dense Layer (256, ReLU)       │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ Output Dense (26, Softmax)    │  ──► P(Class_i | Input Frame)
└───────────────────────────────┘
```

### 5.2 Dynamic Spatial-Temporal Architecture (CNN + Bi-LSTM/GRU)
Dynamic gesture recognition uses a hybrid spatial-temporal architecture. A 2D CNN extracts spatial feature vectors $\mathbf{f}_t \in \mathbb{R}^{512}$ for each frame $t \in [1, 16]$. The sequence of feature vectors $[\mathbf{f}_1, \mathbf{f}_2, \dots, \mathbf{f}_{16}]$ is processed by a 2-layer Bidirectional LSTM or GRU temporal model.

$$\mathbf{h}_t = \text{Bi-LSTM}(\mathbf{f}_t, \mathbf{h}_{t-1})$$

$$\mathbf{y}_{\text{pred}} = \text{Softmax}(\mathbf{W}_o \mathbf{h}_{16} + \mathbf{b}_o)$$

```
Frame Sequence (16 x 224 x 224 x 3)
       │
       ▼
┌───────────────────────────────────────┐
│ Frame-wise CNN Feature Extractor      │  ──► Generates (16 x 512) Matrix
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 2-Layer Bidirectional LSTM / GRU      │  ──► Captures Forward & Backward Temporal Context
│ (128 Hidden Units, Dropout = 0.3)     │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ Classification Layer (Softmax)        │  ──► Dynamic Sign Label & Confidence
└───────────────────────────────────────┘
```

### 5.3 RAG / LLM Contextual Translation Architecture
Raw sign language prediction outputs consist of ungrammatical sign token sequences (e.g., `["HELLO", "YOU", "NAME", "WHAT"]`). The RAG (Retrieval-Augmented Generation) module bridges the gap between sign labels and natural language.

```
Sign Recognition Stream: ["HELLO", "YOU", "NAME", "WHAT"]
                        │
                        ▼
       ┌─────────────────────────────────┐
       │ Vector Similarity Search        │ ──► Searches Sign Grammar Dictionary
       │ (FAISS / Chromadb Index)        │
       └────────────────┬────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────┐
       │ Prompt & Context Injector       │ ──► Context: "ISL Question Template"
       └────────────────┬────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────┐
       │ LLM Generator (Flan-T5 / Llama) │
       └────────────────┬────────────────┘
                        │
                        ▼
Natural English Output: "Hello! What is your name?"
```

---

## 6. Software Integration & Testing Framework (Member 5)

### 6.1 Modular Repository Structure
The project repository is structured to enforce clean component isolation:

```
d:/ISL/
├── Dataset/                       # Static ISL image dataset (Training, Testing, Validation)
├── Sample Videos/                 # Dynamic ISL video dataset (61 MP4 files)
├── analysis_output/               # Sample frame sequence outputs
├── data_pipeline/                 # Reusable Preprocessing Package
│   ├── contracts.py               # Shape validation contracts
│   ├── static_preprocessing.py    # Static image resizing & normalization
│   └── dynamic_preprocessing.py   # Dynamic video decoding & uniform sampling
├── docs/                          # Architectural specifications
│   └── architecture_spec.md       # CNN, LSTM/GRU, and RAG technical spec
├── integration/                   # Interface Contracts & Mocks
│   ├── mock_static.py             # Mock static recognizer (run_static)
│   └── mock_dynamic.py            # Mock dynamic recognizer (run_dynamic)
├── isl_dataset_analysis.py        # Static dataset statistics script
├── isl_leakage_check.py           # Cross-split duplicate check script
├── isl_remove_train_leaks.py      # Train leak removal script
├── isl_remove_val_leaks.py        # Validation leak removal script
├── dynamic_analysis.py            # Dynamic video analysis script
├── integration_tests.py           # Automated Test Suite
├── read.md                        # Static dataset audit documentation
├── writeup.md                     # Data cleaning writeup
├── dynamic_dataset_report.md      # Dynamic video dataset report
├── review1_summary.md             # Review 1 master evidence report
├── ISL_Research_Paper.md          # Master Project Research Paper
├── requirements.txt               # Dependencies file
└── .gitignore                     # Exclusion filters
```

### 6.2 Interface Contracts & Mock Implementation
To enable application development prior to final model training, Member 5 established standalone contract interfaces:
- `run_static(frame: np.ndarray) -> Tuple[str, float]`
- `run_dynamic(frame_buffer: Union[List[np.ndarray], np.ndarray]) -> Tuple[str, float]`

### 6.3 Automated Integration Test Verification
An automated test suite (`integration_tests.py`) was constructed using Python's `unittest` framework to validate all pipeline contracts and mock modules.

```
==============================================
  RUNNING REVIEW 1 INTEGRATION TEST SUITE     
==============================================

 [PASS] Real Video Integration Test ('Hello.mp4'): Label='Name', Conf=0.92
 [PASS] Dynamic Preprocessing Contract
 [PASS] Mock Dynamic Recognizer: Label='Food', Conf=0.92
 [PASS] Mock Static Recognizer: Label='G', Conf=0.95
 [PASS] Static Preprocessing Contract

----------------------------------------------------------------------
Ran 5 tests in 0.607s

OK (100% Pass Rate)
```

---

## 7. Project Progress & Review Roadmap

```
+-----------------------------------------------------------------------------------+
|                            3-REVIEW PROGRESS ROADMAP                              |
+-------------------+---------------------------------------------------------------+
| Review Phase      | Status & Milestones                                           |
+-------------------+---------------------------------------------------------------+
| **Review 1**      | **100% COMPLETE & VERIFIED**                                  |
|                   | • Cleaned 25,238 static dataset (0 leakage).                  |
|                   | • Analyzed 61 dynamic gesture videos (30 FPS, T=16 sampling). |
|                   | • Built reusable `data_pipeline` contracts.                   |
|                   | • Documented `architecture_spec.md` (CNN, LSTM/GRU, RAG).     |
|                   | • Passed `integration_tests.py` with 100% success rate.       |
+-------------------+---------------------------------------------------------------+
| **Review 2**      | **PLANNED NEXT STEP**                                         |
|                   | • Train baseline CNN & MobileNetV2 transfer learning model.   |
|                   | • Train Spatial CNN + LSTM/GRU dynamic model.                 |
|                   | • Evaluate accuracy, precision, recall, F1 & confusion matrix.|
|                   | • Build initial RAG prototype for token-to-sentence mapping.  |
+-------------------+---------------------------------------------------------------+
| **Review 3**      | **FINAL PHASE**                                               |
|                   | • Integrate trained models into live OpenCV webcam application.|
|                   | • Implement sliding window frame buffer & prediction smoothing.|
|                   | • End-to-end responsiveness and latency benchmark.            |
|                   | • Complete final system evaluation and demo workflow.         |
+-------------------+---------------------------------------------------------------+
```

---

## 8. Conclusion

This research paper provides a complete technical and empirical foundation for the Real-Time Indian Sign Language Recognition System. By implementing cryptographic hashing data hygiene, we eliminated 737 cross-split duplicate images, establishing a zero-leakage benchmark of 25,238 static images. We established a uniform 16-frame sequence sampling algorithm for dynamic video processing across 61 gesture classes. Furthermore, we designed a multi-branch spatial-temporal deep learning architecture paired with a RAG/LLM natural language translation pipeline. The complete software system has been modularized, documented, and verified via automated integration tests, making the project 100% prepared for model training in Review 2 and live webcam deployment in Review 3.

---

## 9. References & Key Project Documents
1. **Project Master Plan**: `Planner/ISL_plan.docx`
2. **Team Workflow & Review Requirements**: `Planner/ISL_Project_Team_Workflow_3_Reviews.docx`
3. **Static Audit Log**: `read.md` & `writeup.md`
4. **Dynamic Video Report**: `dynamic_dataset_report.md`
5. **Architecture Specification**: `docs/architecture_spec.md`
6. **Review 1 Master Evidence Report**: `review1_summary.md`
