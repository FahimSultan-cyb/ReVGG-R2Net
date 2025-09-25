import os
import json
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers, callbacks
from sklearn.metrics import accuracy_score

from config import Config
from models.revgg_r2net import create_revgg_r2net
from utils.data_loader import load_and_preprocess_data
from utils.metrics import dice_coef, jaccard_index, precision_metric, recall_metric, f1_metric

physical_devices = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
config = Config()

def optimize_batch_size(x_train, y_train, x_val, y_val):
    best_batch_size = config.BATCH_SIZE_OPTIONS[0]
    best_val_dice = 0
    
    print("Optimizing batch size...")
    for batch_size in config.BATCH_SIZE_OPTIONS:
        print(f"Testing batch size: {batch_size}")
        temp_model = create_revgg_r2net()
        temp_model.compile(optimizer=optimizers.Adam(learning_rate=config.INITIAL_LEARNING_RATE), 
                          loss='binary_crossentropy', 
                          metrics=[dice_coef])
        
        temp_history = temp_model.fit(x_train, y_train,
                                     validation_data=(x_val, y_val),
                                     batch_size=batch_size,
                                     epochs=config.BATCH_SIZE_TEST_EPOCHS,
                                     verbose=0)
        
        val_dice = max(temp_history.history['val_dice_coef'])
        print(f"Batch size {batch_size} - Validation Dice: {val_dice:.4f}")
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_batch_size = batch_size
        
        del temp_model
    
    print(f"Optimal batch size selected: {best_batch_size}")
    return best_batch_size

def train_model(data_path, save_path=None):
    config.create_directories()
    
    if save_path is None:
        save_path = os.path.join(config.MODELS_DIR, "ReVGG_R2Net_final.keras")
    
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_preprocess_data(data_path)
    
    print("Dataset Information:")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Input shape: {x_train.shape[1:]}")
    print(f"Output shape: {y_train.shape[1:]}")
    
    best_batch_size = optimize_batch_size(x_train, y_train, x_val, y_val)
    
    model = create_revgg_r2net()
    optimizer = optimizers.Adam(learning_rate=config.INITIAL_LEARNING_RATE)
    
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=[dice_coef, jaccard_index, precision_metric, recall_metric, f1_metric])
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=config.LR_REDUCE_FACTOR, 
        patience=config.LR_PATIENCE, 
        min_lr=config.MIN_LEARNING_RATE, 
        verbose=1
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_dice_coef', 
        patience=config.EARLY_STOPPING_PATIENCE, 
        restore_best_weights=True, 
        mode='max', 
        verbose=1
    )
    model_checkpoint = callbacks.ModelCheckpoint(
        save_path, 
        monitor='val_dice_coef', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    csv_logger = callbacks.CSVLogger(os.path.join(config.RESULTS_DIR, 'training_log.csv'))
    
    callbacks_list = [reduce_lr, early_stopping, model_checkpoint, csv_logger]
    
    print("Starting main training...")
    history = model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       batch_size=best_batch_size,
                       epochs=config.MAX_EPOCHS,
                       callbacks=callbacks_list,
                       verbose=1)
    
    model.save(save_path)
    print(f"Final model saved at: {save_path}")
    
    plt.figure(figsize=(20, 15))
    metrics_to_plot = ['loss', 'dice_coef', 'jaccard_index', 'precision_metric', 'recall_metric', 'f1_metric']
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        plt.plot(history.history[metric], label=f'Training {metric.replace("_", " ").title()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.replace("_", " ").title()}')
        plt.title(f'{metric.replace("_", " ").title()} Convergence')
        plt.xlabel('Epochs')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'training_convergence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    model.load_weights(save_path)
    
    print("Evaluating on test set...")
    test_predictions = model.predict(x_test, batch_size=1, verbose=1)
    test_predictions_binary = (test_predictions > 0.5).astype(np.float32)
    
    test_accuracy = accuracy_score(y_test.flatten(), test_predictions_binary.flatten())
    test_dice = dice_coef(y_test, test_predictions).numpy()
    test_jaccard = jaccard_index(y_test, test_predictions).numpy()
    test_precision = precision_metric(y_test, test_predictions).numpy()
    test_recall = recall_metric(y_test, test_predictions).numpy()
    test_f1 = f1_metric(y_test, test_predictions).numpy()
    
    results = {
        'Model': 'ReVGG-R2Net',
        'Accuracy': float(test_accuracy),
        'Dice_Coefficient': float(test_dice),
        'Jaccard_Index': float(test_jaccard),
        'Precision': float(test_precision),
        'Recall': float(test_recall),
        'F1_Score': float(test_f1),
        'Training_Epochs': len(history.history['loss']),
        'Best_Batch_Size': best_batch_size,
        'Final_Learning_Rate': float(model.optimizer.learning_rate),
        'Dataset_Size': {
            'Total_Images': len(x_train) + len(x_val) + len(x_test),
            'Training': len(x_train),
            'Validation': len(x_val),
            'Test': len(x_test)
        }
    }
    
    with open(os.path.join(config.RESULTS_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Model: {results['Model']}")
    print(f"Accuracy: {results['Accuracy']:.4f}")
    print(f"Dice Coefficient: {results['Dice_Coefficient']:.4f}")
    print(f"Jaccard Index: {results['Jaccard_Index']:.4f}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"Recall: {results['Recall']:.4f}")
    print(f"F1-Score: {results['F1_Score']:.4f}")
    print(f"Training Epochs: {results['Training_Epochs']}")
    print(f"Optimal Batch Size: {results['Best_Batch_Size']}")
    print("="*60)
    
    return model, history, results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    final_model, training_history, test_results = train_model(data_path)
