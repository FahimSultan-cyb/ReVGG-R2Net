import os
import tensorflow as tf

class Config:
    RANDOM_SEED = 42
    
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    INITIAL_LEARNING_RATE = 1e-4
    MIN_LEARNING_RATE = 1e-7
    LR_REDUCE_FACTOR = 0.5
    LR_PATIENCE = 10
    EARLY_STOPPING_PATIENCE = 20
    
    MAX_EPOCHS = 50
    BATCH_SIZE_OPTIONS = [1, 2, 4]
    BATCH_SIZE_TEST_EPOCHS = 3
    
    RECURRENT_STEPS = 2
    
    SMOOTH_FACTOR = 1e-6
    
    RESULTS_DIR = "results"
    MODELS_DIR = "pretrained"
    
    @staticmethod
    def create_directories():
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
    
    @staticmethod
    def configure_gpu():
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
