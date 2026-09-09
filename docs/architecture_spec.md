# Indian Sign Language Recognition Architecture & RAG Specification (Review 1)

This document provides the Review 1 technical architecture specification for **Member 4 (Model Architecture & RAG Module)** and **Member 1/2**.

---

## 1. System Overview

The Real-Time ISL System consists of three main operational modules:
1. **Static Recognition Branch**: Single-frame spatial gesture classification (A–Z alphabets).
2. **Dynamic Recognition Branch**: Multi-frame spatial-temporal sign recognition (Words/Sentences).
3. **RAG / LLM Enhancement Module**: Post-processing layer converting recognized sign streams into natural, grammatically correct text.

```
                     +---------------------------+
                     |    Webcam / Video Input   |
                     +-------------+-------------+
                                   |
           +-----------------------+-----------------------+
           |                                               |
           v                                               v
+-----------------------+                       +-----------------------+
|  Static Image Branch  |                       | Dynamic Video Branch  |
|  Frame: (224, 224, 3) |                       | Sequence: (T, 224,224,3)|
+-----------+-----------+                       +-----------+-----------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
| MobileNetV2 / CNN     |                       | Spatial CNN Feature   |
| Spatial Feature Ext.  |                       | Extractor (16 x 512)  |
+-----------+-----------+                       +-----------+-----------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
| Dense Classifier Head |                       | Temporal Classifier   |
| (26 Alphabet Classes) |                       | (LSTM / GRU Layer)    |
+-----------+-----------+                       +-----------+-----------+
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v
                        +-----------------------+
                        |  Predicted Sign Text  |
                        +-----------+-----------+
                                    |
                                    v
                        +-----------------------+
                        | RAG / LLM Post-Proc.  |
                        | Context & Grammar     |
                        +-----------------------+
```

---

## 2. Static Model Architecture (Member 1 & 4)

- **Input Dimension**: `(224, 224, 3)` RGB float tensor normalized to `[0.0, 1.0]`.
- **Backbone**: MobileNetV2 (Pre-trained on ImageNet / Transfer Learning).
- **Spatial Feature Representation**: Global Average Pooling layer extracting a 1280-dimensional feature vector.
- **Classification Head**:
  - `Dropout(0.2)`
  - `Dense(256, activation='relu')`
  - `Dense(26, activation='softmax')` (Alphabet classes A–Z)

---

## 3. Dynamic Spatial-Temporal Architecture (Member 2 & 4)

- **Input Dimension**: Sequence tensor `(T=16, H=224, W=224, C=3)`.
- **Spatial Feature Extractor**: Frame-level CNN backbone producing frame embedding vector `F_t` of size 512.
- **Sequence Representation**: Tensor shape `(Batch, 16, 512)`.
- **Temporal Classifier Options**:
  - **Option A (LSTM)**: 2-layer Bidirectional LSTM (Hidden Units: 128, Dropout: 0.3).
  - **Option B (GRU)**: 2-layer Bidirectional GRU (Hidden Units: 128, Dropout: 0.3).
- **Classification Head**: `Dense(Num_Dynamic_Classes, activation='softmax')`.

---

## 4. LLM / RAG Module Specification (Member 4)

### Purpose & Use Case
ISL sign recognizers output isolated sign labels (e.g., `["hello", "you", "name", "what"]`). The RAG / LLM module ingests these recognized keyword streams, retrieves contextual sentence templates or domain sign dictionary definitions, and outputs natural English: `"Hello! What is your name?"`.

### Retrieval & Generation Architecture
1. **Input**: Sequence of predicted sign tokens + confidence scores.
2. **Context Retrieval**: Search vector index of ISL phrase pairs & grammar rules.
3. **LLM Generator**: Small footprint LLM (e.g. Flan-T5 / Llama prompt template) instructed to format the output string without hallucinating facts.
4. **Output**: Natural language sentence.

---

## 5. Review 1 Acceptance Checkpoints
- [x] Static spatial architecture diagram and tensor shapes defined
- [x] Dynamic temporal (CNN + LSTM/GRU) architecture and contracts defined
- [x] RAG / LLM use-case purpose and retrieval pipeline specified
- [x] Input / output tensor specifications documented for all members
