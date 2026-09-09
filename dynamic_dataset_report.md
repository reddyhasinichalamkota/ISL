# Dynamic ISL Video Dataset Analysis Report (Review 1)

## Executive Summary
This document provides the Review 1 dataset analysis and spatial-temporal pre-processing evaluation for Member 2 (Dynamic Sign Recognition).

- **Total Analyzed Videos**: 61
- **Unique Dynamic Signs**: 61
- **Video Resolution**: 640x720, 880x720, 760x720
- **FPS Range**: 30.0 FPS – 30.0 FPS (Average: 30.00 FPS)
- **Frame Count Range**: 111 – 298 frames (Average: 166.4 frames)
- **Video Duration**: 3.7s – 9.93s (Average: 5.55 seconds)

---

## Sampling & Preprocessing Strategy
1. **Sequence Sampling**: Fixed uniform sampling of **T = 16 frames** per video to balance spatial detail and temporal context.
2. **Spatial Scaling**: Resized each extracted frame to **224 × 224 × 3** matching spatial CNN backbone standards.
3. **Pixel Normalization**: Re-scaled RGB values to `[0.0, 1.0]`.

---

## Sample Extracted Sequences
- Sample 'Bear.mp4': Preprocessed sequence tensor shape = (16, 224, 224, 3), pixel range = [27, 255]
- Sample 'Break.mp4': Preprocessed sequence tensor shape = (16, 224, 224, 3), pixel range = [35, 255]
- Sample 'Brinjal.mp4': Preprocessed sequence tensor shape = (16, 224, 224, 3), pixel range = [29, 255]
- Sample 'Budget.mp4': Preprocessed sequence tensor shape = (16, 224, 224, 3), pixel range = [18, 255]

---
## Review 1 Checkpoints Checklist
- [x] Dynamic dataset selected and analyzed
- [x] Resolution, FPS, and frame count statistics documented
- [x] Frame extraction sequence pipeline verified
- [x] Temporal sequence shape contract confirmed: `(16, 224, 224, 3)`
- [x] Initial sequence sampling strategy documented
