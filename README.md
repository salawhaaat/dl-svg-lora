# NYU Deep Learning Spring 2026 — Midterm Assignment  
**Text-to-SVG generation via LoRA fine-tuning of small language models**

---

## Overview

Fine-tune a language model (≤4B parameters) to generate valid SVG code from natural language prompts. Models are evaluated on a Kaggle competition leaderboard using a rendering-based similarity metric.

**Best result:** Qwen3-4B with LoRA r=64, alpha=128, 3 epochs → **12.94 public score**

---

## Setup

```bash
pip install unsloth transformers datasets trl peft
```

All training was done on Google Colab with an A100 GPU (40 GB). No local GPU required — open the training notebook directly in Colab.

---

## Dataset

- **Source:** Kaggle competition dataset
- **Size:** 50,000 (prompt, SVG) pairs
- **Format:** Each sample is a text prompt paired with its corresponding SVG code

---

## Training

Training uses [Unsloth](https://github.com/unslothai/unsloth) for efficient LoRA fine-tuning with `SFTTrainer` from the `trl` library.

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| Dropout | 0.05 |
| Epochs | 3 |
| Sequence length | 2048 |
| Optimizer | AdamW 8-bit |
| Learning rate | 2e-4 |

**Run training:**

Open `training.ipynb` in Google Colab and execute all cells. The notebook handles model loading, LoRA setup, dataset preparation, and saving the adapter weights.

---

## Inference

Inference uses HuggingFace `transformers` with **sequential greedy decoding** (one sample at a time, no batching).

> **Important:** Batched inference produces ~33% invalid SVGs due to padding artifacts. Sequential decoding drops this to 0% invalid outputs.

**Run inference:**

Open `inference.ipynb` in Google Colab, load the saved LoRA adapter, and run the generation loop over the test prompts.

---

## Results

All runs use LoRA on Qwen-family models. Sequence length is 2048 unless noted.

| Model | LoRA r | Alpha | Epochs | Notes | Public Score |
|-------|--------|-------|--------|-------|-------------|
| Qwen2.5-1.5B-Instruct | 16 | 32 | 1 | Baseline | 8.21 |
| Qwen2.5-1.5B-Instruct | 32 | 64 | 2 | — | 9.14 |
| Qwen2.5-Coder-3B-Instruct | 32 | 64 | 2 | Code-tuned base | 10.37 |
| Qwen2.5-Coder-3B-Instruct | 64 | 128 | 3 | Batched inference | 9.88 |
| Qwen2.5-Coder-3B-Instruct | 64 | 128 | 3 | Sequential inference | 11.52 |
| Qwen3-4B | 32 | 64 | 2 | — | 11.89 |
| **Qwen3-4B** | **64** | **128** | **3** | **Sequential inference** | **12.94** |

**Key findings:**
- Larger LoRA rank (r=64) consistently outperforms r=16/32 across all models
- Sequential greedy decoding eliminates invalid SVG outputs (0% vs ~33% with batched)
- Qwen3-4B outperforms Qwen2.5-Coder-3B despite the latter being code-specialized
- Increasing epochs from 2→3 provides consistent gains without overfitting at this scale

---

## AI Tools Disclosure

This project used AI assistance as follows:

- **GitHub Copilot / Claude:** Used for boilerplate code generation (training loop setup, dataset formatting), debugging LoRA configuration errors, and drafting this README
- **All experimental design, hyperparameter choices, and analysis** were done independently

---

## Repository Structure

```
├── training.ipynb      # LoRA fine-tuning notebook (Colab)
├── inference.ipynb     # Generation + submission notebook (Colab)
└── README.md
```

---

*—Deep Learning (Spring 2026)*
