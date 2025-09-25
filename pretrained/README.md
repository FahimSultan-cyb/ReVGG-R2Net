# Pretrained Models

This directory contains trained ReVGG-R2Net models.

## Model Files

- `ReVGG_R2Net_final.keras` - Main trained model file
- Place your trained models here for easy access

## Model Specifications

- **Input Size**: 512×512×3 (RGB images)
- **Output Size**: 512×512×1 (Binary masks)
- **Architecture**: VGG16 encoder with R2 blocks
- **Activation**: Sigmoid (binary segmentation)

## Loading Models

```python
from utils.metrics import get_custom_objects
import tensorflow as tf

# Load with custom metrics
custom_objects = get_custom_objects()
model = tf.keras.models.load_model('pretrained/ReVGG_R2Net_final.keras', custom_objects=custom_objects)
```

## File Structure

Trained models should follow this naming convention:
- `ReVGG_R2Net_[dataset]_[date].keras` - For different datasets
- `ReVGG_R2Net_final.keras` - Main production model



## Usage Notes

1. Models are saved in Keras format (.keras)
2. Custom metrics are required for loading
3. All models expect 512×512 input images
4. Preprocessing should match training pipeline
