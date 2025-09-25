import os

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
    BATCH_SIZE_OPTIONS = [4, 8, 16]
    BATCH_SIZE_TEST_EPOCHS = 5
    
    RECURRENT_STEPS = 2
    
    SMOOTH_FACTOR = 1e-6
    
    RESULTS_DIR = "results"
    MODELS_DIR = "pretrained"
    
    @staticmethod
    def create_directories():
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
