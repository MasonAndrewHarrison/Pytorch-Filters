# Pytorch-Filters

A collection of image edge detection filters built with PyTorch — fully vectorized, GPU-compatible, and easy to drop into any vision pipeline.

## Filters

- **Canny** — classic multi-stage edge detector with Gaussian blur, Sobel gradients, non-maximum suppression, and hysteresis
- **DoG** (Difference of Gaussians) — fast approximation of the Laplacian of Gaussian for edge detection
- **XDoG** (Extended Difference of Gaussians) — stylized edge detection with soft thresholding, great for non-photorealistic rendering

## Installation

```bash
pip install pytorch-filters
```

## Quick Start

```python
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pytorch_filters import canny, difference_of_gaussians, ex_difference_of_gaussians

# Load image as tensor [1, 1, H, W]
img = TF.to_tensor(Image.open("photo.jpg").convert("L")).unsqueeze(0)

edges = canny(img)
dog   = difference_of_gaussians(img, sigma=1.4, k=1.6)
xdog  = ex_difference_of_gaussians(img, sigma=1.0, tau=0.99, phi=100)
```

## Demo

```bash
pip install -r requirements.txt
python demo.py images/demo1.jpg
```

This will display the original image alongside Canny, DoG, and XDoG results side by side.

![demo output](images/output.png)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision
- NumPy
