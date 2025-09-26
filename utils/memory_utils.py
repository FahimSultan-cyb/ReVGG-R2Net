import os
import gc
import tensorflow as tf
import torch

def configure_tensorflow_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU")

def clear_session():
    tf.keras.backend.clear_session()
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_info():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            if 'device_name' in gpu_details:
                print(f"GPU Device: {gpu_details['device_name']}")
    except:
        pass

def optimize_for_colab():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    
    configure_tensorflow_memory()
    
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

def set_mixed_precision():
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")
    except:
        print("Mixed precision not available")

class MemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_session()
        if epoch % 5 == 0:
            get_memory_info()
