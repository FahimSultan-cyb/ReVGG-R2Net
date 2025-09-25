# ReVGG-R2Net: Residual VGG with Recurrent Blocks for Medical Image Segmentation
A robust implementation of ReVGG-R2Net architecture combining VGG16 encoder with Residual Recurrent Convolutional blocks for binary medical image segmentation.


### Architecture Overview
Encoder: VGG16 with ImageNet pretrained weights (all layers trainable)
Core Innovation: Residual Recurrent Convolutional Blocks (R2 blocks)
Decoder: Transpose convolution with skip connections
Output: Sigmoid activation for binary segmentation

- **Encoder**: VGG16 
- **Core Innovation**: Residual Recurrent Convolutional Blocks (R2 blocks)
- **Decoder**:  Transpose convolution with skip connections
- **Output**: Sigmoid activation for binary segmentation



### Key Features
- **Residual connections with recurrent convolutions**
- **Batch size optimization during training**
- **Comprehensive evaluation metrics (Dice, Jaccard, F1-Score)**
- **Mac MPS GPU support**
- **Robust data preprocessing pipeline**



## Installation
```bash
git clone https://github.com/FahimSultan-cyb/ReVGG-R2Net.git
import os, sys
root_path = os.path.join(os.getcwd(), "ReVGG-R2Net")
os.chdir(root_path)
!pip install -r requirements.txt
```


## Training
```bash
from scripts.train import train_model
model, history = train_model(data_path="path/to/dataset")
```


## Inference
```bash
from scripts.inference import load_model_and_predict
predictions = load_model_and_predict(
    model_path="path/to/model.keras",
    image_path="path/to/test/images"
)
```


## Evaluation
```bash
from scripts.evaluate import evaluate_model
results = evaluate_model(
    model_path="path/to/model.keras",
    test_data_path="path/to/test/data"
)
```









