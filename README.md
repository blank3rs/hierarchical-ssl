<div align="center">

# 🧠 Dualen

**A hierarchical self-supervised learning system for text**

*Learn perception → world dynamics → value estimation*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Examples](#-examples) • [Roadmap](#-roadmap)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Examples](#-examples)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Contributing](#-contributing)

---

## 🎯 Overview

Dualen implements a three-layer predictive architecture that learns increasingly abstract representations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 3: Value Model                     │
│              "What it means for me" 🚧                      │
│         Identifies valuable states for exploration          │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                   Layer 2: World Model                      │
│                  "What it means" 🚧                          │
│            Learns how states evolve over time               │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Layer 1: Perception (JEPA) ✅                   │
│                      "What"                                  │
│         Learns rich representations from text               │
└─────────────────────────────────────────────────────────────┘
```

### Current Status

| Layer | Status | Description |
|:-----:|:-----:|-------------|
| **Layer 1** | ✅ **Complete** | JEPA - Self-supervised text representations |
| **Layer 2** | 🚧 **Planned** | World model - State transition prediction |
| **Layer 3** | 🚧 **Planned** | Value model - Reward estimation |

---

## 🚀 Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Train Layer 1 (JEPA)

```bash
# Basic training with defaults
python train.py

# Custom configuration
python train.py --epochs 50 --batch_size 64 --lr 0.0002
```

### Use the Model

```python
from jepa import JEPA
from transformers import AutoTokenizer

# Initialize model
model = JEPA(vocab_size=30522, hidden_dim=768, encoder_layers=12)

# Load pretrained checkpoint (optional)
model.load_pretrained('checkpoints/checkpoint_latest.pt')

# Get embeddings
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text = "Your text here"
tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

embeddings = model.get_representations(
    tokens['input_ids'],
    tokens['attention_mask']
)
print(f"Embedding shape: {embeddings.shape}")  # [batch_size, hidden_dim]
```

---

## 📐 Architecture

### Layer 1: Perception (JEPA) ✅

**What it does**: Learns rich text representations by predicting masked spans from context.

```
┌──────────────────────────────────────────────────────────────┐
│                        Input Text                            │
│  "The quick brown [MASK] jumps over the [MASK] dog"         │
└───────────────┬──────────────────────────┬─────────────────┘
                │                            │
        ┌───────▼────────┐          ┌───────▼────────┐
        │   Context     │          │    Target      │
        │   Encoder      │          │    Encoder     │
        │  (Trainable)   │          │   (EMA, Frozen)│
        └───────┬────────┘          └───────┬────────┘
                │                            │
                │      ┌───────────┐         │
                └─────▶│ Predictor │◀────────┘
                       │ (2 layers)│
                       └─────┬─────┘
                             │
                    ┌────────▼────────┐
                    │  VICReg Loss    │
                    │  (Prevent       │
                    │   Collapse)     │
                    └─────────────────┘
```

**Key Components**:
- **Context Encoder**: 12-layer transformer processes visible text with `[MASK]` tokens
- **Target Encoder**: Same architecture, frozen, updated via EMA (momentum=0.996)
- **Predictor**: Lightweight 2-layer transformer decoder
- **Loss**: VICReg-style with variance and covariance regularization

**Features**:
- ✅ Self-supervised (no labels needed)
- ✅ Prevents representational collapse
- ✅ Produces rich contextualized embeddings
- ✅ EMA target encoder for stable training

### Layer 2: World Model 🚧

**Goal**: Learn how states evolve over time in embedding space.

**Data**: Sequential pairs `(s_t, a_t, s_{t+1})`
- `s_t`: Current state embedding (from Layer 1)
- `a_t`: Optional action
- `s_{t+1}`: Next state embedding

**Model**: Small transformer/RNN predictor
```
z_t = encoder(s_t)
ẑ_{t+1} = f_θ(z_t, a_t)
```

**Loss**: `L_world = ||z_{t+1} - f_θ(z_t, a_t)||²`

### Layer 3: Value Model 🚧

**Goal**: Attach value/relevance to latent states.

**Rewards**:
- **External**: Task success, human preferences
- **Intrinsic**: `r_t = |E_{t-1} - E_t|` (world model prediction error change)

**Model**: Value function `V_φ(z_t)` + optional policy `π_ψ(a_t|z_t)`

**Training**: TD learning `L_value = (r_t + γV_φ(z_{t+1}) - V_φ(z_t))²`

---

## 💡 Examples

### Basic Training

```bash
# Quick test run
python train.py --epochs 10 --batch_size 16 --max_samples 1000

# Full training
python train.py --epochs 100 --batch_size 64 --lr 2e-4
```

### Custom Configuration

```bash
python train.py \
    --epochs 100 \
    --batch_size 64 \
    --lr 2e-4 \
    --hidden_dim 768 \
    --num_layers 12 \
    --predictor_layers 2 \
    --max_length 256 \
    --total_mask_ratio 0.15 \
    --num_spans 4 \
    --checkpoint_dir checkpoints/run1
```

### Extract Embeddings

```python
import torch
from jepa import JEPA
from transformers import AutoTokenizer

# Load model
model = JEPA(vocab_size=30522, hidden_dim=768)
model.load_pretrained('checkpoints/checkpoint_latest.pt')
model.eval()

# Process text
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is fascinating"
]

tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

with torch.no_grad():
    embeddings = model.get_representations(
        tokens['input_ids'],
        tokens['attention_mask']
    )

print(f"Embeddings shape: {embeddings.shape}")  # [2, 768]
```

---

## ⚙️ Configuration

### Training Arguments

| Argument | Default | Description |
|:--------:|:-------:|-------------|
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `1.5e-4` | Learning rate |
| `--hidden_dim` | `768` | Hidden dimension size |
| `--num_layers` | `12` | Encoder transformer layers |
| `--predictor_layers` | `2` | Predictor layers (keep small) |
| `--ema_momentum` | `0.996` | EMA momentum for target encoder |
| `--max_length` | `128` | Maximum sequence length |
| `--total_mask_ratio` | `0.20` | Fraction of tokens to mask |
| `--num_spans` | `3` | Number of masked spans |
| `--device` | `auto` | Device (auto/cuda/mps/cpu) |
| `--checkpoint_dir` | `checkpoints` | Checkpoint directory |

### Dataset Options

| Argument | Default | Description |
|:--------:|:-------:|-------------|
| `--dataset` | `wikitext` | HuggingFace dataset name |
| `--dataset_config` | `wikitext-2-raw-v1` | Dataset configuration |
| `--max_samples` | `None` | Limit number of samples |

---

## 🗺️ Roadmap

### ✅ Completed
- [x] JEPA architecture implementation
- [x] Context encoder (transformer-based)
- [x] Target encoder with EMA updates
- [x] Predictor network
- [x] VICReg-style loss function
- [x] Training pipeline
- [x] Checkpoint saving/loading
- [x] Training metrics logging

### 🚧 In Progress
- [ ] Layer 2: World model implementation
- [ ] Sequential dataset utilities
- [ ] State transition prediction

### 📅 Planned
- [ ] Layer 3: Value model implementation
- [ ] Intrinsic reward computation
- [ ] Policy network (optional)
- [ ] Joint fine-tuning pipeline
- [ ] Evaluation benchmarks
- [ ] Pre-trained model releases

---

## 📁 Project Structure

```
dualen/
├── jepa/                    # JEPA model (Layer 1)
│   ├── __init__.py
│   ├── context.py           # Context encoder
│   ├── target.py            # Target encoder (EMA)
│   ├── predictor.py         # Predictor network
│   ├── model.py             # Complete JEPA model
│   └── train.py             # Training script
├── data/                    # Dataset utilities
│   └── text_dataset.py      # Text dataset and dataloaders
├── checkpoints/             # Model checkpoints
│   ├── checkpoint_latest.pt
│   ├── checkpoint_epoch_*.pt
│   └── training_history.json
├── train.py                 # Entry point
├── pyproject.toml           # Dependencies
└── README.md
```

---

## 🔬 Technical Details

### JEPA Architecture

- **Context Encoder**: 12-layer transformer (BERT-base style)
  - Hidden dimension: 768
  - Attention heads: 12
  - Intermediate size: 3072
- **Target Encoder**: Same architecture, EMA-updated (momentum=0.996)
- **Predictor**: 2-layer lightweight transformer decoder
- **Loss**: VICReg-style with variance and covariance regularization

### Preventing Collapse

The model uses three mechanisms to prevent representational collapse:

1. **EMA target encoder**: Provides stable, slowly-changing targets
2. **Variance regularization**: Keeps embedding variance above threshold (hinge loss)
3. **Covariance regularization**: Decorrelates embedding dimensions

### Training Flow

```
1. Mask spans in input text
   ↓
2. Context encoder processes visible text
   ↓
3. Target encoder processes masked spans (no gradients)
   ↓
4. Predictor predicts masked span embeddings from context
   ↓
5. Compute VICReg loss (repr + variance + covariance)
   ↓
6. Backprop through context encoder and predictor only
   ↓
7. Update target encoder via EMA
```

### Performance Tips

- 🚀 Use GPU/MPS for faster training
- 📊 Start with smaller `max_length` for faster iteration
- 📈 Monitor `val_cosine_sim` - should increase during training
- 🔍 Check `val_pred_std` and `val_target_std` - should stay above 0.3
- 💾 Save checkpoints regularly for recovery

---

## 📊 Monitoring Training

Training metrics are logged to `checkpoints/training_history.json`:

```json
{
  "epoch": 1,
  "train_loss": 0.5234,
  "val_loss": 0.4891,
  "lr": 0.00015,
  "val_cosine_sim": 0.8234,
  "val_pred_std": 0.4567,
  "val_target_std": 0.4321
}
```

**Key Metrics**:
- `train_loss` / `val_loss`: Overall loss (lower is better)
- `val_cosine_sim`: Cosine similarity between predicted and target (higher is better, target: >0.8)
- `val_pred_std` / `val_target_std`: Embedding standard deviation (should be >0.3)

---

## 🛠️ Requirements

- Python >= 3.12
- PyTorch >= 2.9.0
- transformers >= 4.57.1
- datasets >= 4.4.1
- tqdm >= 4.67.1
- numpy >= 2.3.4

---

## 🤝 Contributing

Contributions are welcome! This is a research project exploring hierarchical self-supervised learning.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 References

- **JEPA**: Joint-Embedding Predictive Architecture (LeCun et al.)
- **VICReg**: Variance-Invariance-Covariance Regularization
- Self-supervised learning for text representations

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

**Made with ❤️ for self-supervised learning**

[⬆ Back to Top](#-dualen)

</div>
