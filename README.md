# SHMoE-Dehaze
SHMoE-Dehaze: Semantic and Haze-Aware Mixture-of-Experts Framework for Spatially Adaptive Image Dehazing.
# SHMoE-Dehaze

Semantic and Haze-Aware Mixture-of-Experts Framework for Spatially Adaptive Image Dehazing

## Overview

SHMoE-Dehaze is a semantic-guided image dehazing framework designed to address spatially varying haze degradation. Unlike conventional dehazing methods that apply a single restoration strategy across an entire image, SHMoE-Dehaze dynamically allocates restoration capacity according to local haze severity and scene semantics.

The framework integrates:

- Atmospheric Scattering Model (ASM) expert for light haze restoration
- MH-Former (Transformer-based expert) for medium haze restoration
- HazeDiff-CPG (Diffusion-based expert) for severe haze restoration
- Semantic-haze routing module for adaptive expert allocation
- Soft Mixture-of-Experts (MoE) fusion for spatially consistent restoration

## Framework

<p align="center">
  docs/architecture.png
</p>

The proposed framework consists of:

1. Semantic feature extraction using SAM-3
2. Haze-aware feature encoding
3. Cross-attention semantic-haze routing
4. Adaptive expert selection
5. Soft expert fusion

## Features

- Semantic-aware dehazing
- Spatially adaptive restoration
- Mixture-of-Experts architecture
- Transformer and diffusion integration
- Physics-guided restoration
- Frequency-aware diffusion guidance
- End-to-end trainable framework

## Datasets

The framework was trained and evaluated on:

### Training
- RESIDE ITS
- RESIDE OTS

### Evaluation
- O-HAZE
- I-HAZE
- Dense-HAZE
- NH-HAZE

## Results

SHMoE-Dehaze achieves strong performance across synthetic and real-world dehazing benchmarks while maintaining high structural fidelity and color consistency.

| Model | PSNR | SSIM |
|---------|---------|---------|
| SHMoE-Dehaze | 33.7 | 0.972 |

## Repository Structure

```text
SHMoE-Dehaze/
│
├── datasets/
├── models/
│   ├── asm/
│   ├── mhformer/
│   ├── hazediff_cpg/
│   └── routing/
│
├── configs/
├── scripts/
├── checkpoints/
├── utils/
├── evaluation/
├── docs/
└── README.md
