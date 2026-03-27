# BRENet

BRENet (Boundary-Region Enhancement Network) — a deep learning model repository.

## Repository Structure

```
BRENet/
├── models/
│   ├── BRENet.py      # Main model definition
│   ├── backbone.py    # ResNet backbone
│   └── modules.py     # Core modules (BRE, ChannelAttention, SpatialAttention)
├── utils/
│   └── utils.py       # Training utilities (clip_gradient, adjust_lr)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from models import BRENet

model = BRENet(backbone='resnet50', pretrained=True)
```