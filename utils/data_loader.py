import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from config import Config

config = Config()
tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_dir = os.path.join(data_path, 'Original')
        self.mask_dir = os.path.join(data_path, 'Mask')
        
    def _load_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                img_resized = img.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
                image_array = np.array(img_resized, dtype=np.float32)
            
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array]*3, axis=-1)
            elif image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            return image_array / 255.0
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.INPUT_CHANNELS), dtype=np.float32)
    
    def _load_mask(self, mask_path):
        try:
            with Image.open(mask_path) as mask_img:
                mask_img = ImageOps.grayscale(mask_img)
                mask_resized = mask_img.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT), Image.NEAREST)
                mask_array = np.array(mask_resized, dtype=np.float32)
            
            mask_array = mask_array / 255.0
            mask_array = np.expand_dims(mask_array, axis=-1)
            return mask_array
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.OUTPUT_CHANNELS), dtype=np.float32)
    
    def load_data(self):
        images = sorted([f for f in os.listdir(self.image_dir) 
                        if not f.startswith('.') and os.path.isfile(os.path.join(self.image_dir, f))])
        masks = sorted([f for f in os.listdir(self.mask_dir) 
                       if not f.startswith('.') and os.path.isfile(os.path.join(self.mask_dir, f))])
        
        x_data = np.zeros((len(images), config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.INPUT_CHANNELS), dtype=np.float32)
        y_data = np.zeros((len(masks), config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.OUTPUT_CHANNELS), dtype=np.float32)
        
        print(f"Loading {len(images)} images...")
        for idx, img_name in enumerate(images):
            img_path = os.path.join(self.image_dir, img_name)
            x_data[idx] = self._load_image(img_path)
        
        print(f"Loading {len(masks)} masks...")
        for idx, mask_name in enumerate(masks):
            mask_path = os.path.join(self.mask_dir, mask_name)
            y_data[idx] = self._load_mask(mask_path)
        
        return x_data, y_data
    
    def split_data(self, x_data, y_data):
        x_train, x_temp, y_train, y_temp = train_test_split(
            x_data, y_data, test_size=(1-config.TRAIN_RATIO), random_state=config.RANDOM_SEED)
        
        val_size = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=(1-val_size), random_state=config.RANDOM_SEED)
        
        return x_train, x_val, x_test, y_train, y_val, y_test

def load_and_preprocess_data(data_path):
    loader = DataLoader(data_path)
    x_data, y_data = loader.load_data()
    return loader.split_data(x_data, y_data)
