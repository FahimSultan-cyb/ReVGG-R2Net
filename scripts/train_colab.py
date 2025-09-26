import os
import sys
import gc
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers, callbacks


from config import Config
from models.revgg_r2net import create_revgg_r2net
from utils.data_loader import load_and_preprocess_data
from utils.metrics import dice_coef, jaccard_index, precision_metric, recall_metric, f1_metric

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

config = Config()

def clear_memory():
    tf.keras.backend.clear_session()
    gc.collect()

def train_model_colab(data_path, save_path=None):
    clear_memory()
    config.create_directories()
    
    if save_path is None:
        save_path = os.path.join(config.MODELS_DIR, "ReVGG_R2Net_final.keras")
    
    print("Loading and preprocessing data...")
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_preprocess_data(data_path)
    
    print("Dataset Information:")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Input shape: {x_train.shape[1:]}")
    print(f"Output shape: {y_train.shape[1:]}")
    
    print("Creating model...")
    model = create_revgg_r2net()
    
    optimizer = optimizers.Adam(learning_rate=1e-4)
    
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=[dice_coef, jaccard_index, precision_metric, recall_metric, f1_metric],
        run_eagerly=False
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=8, 
        min_lr=1e-7, 
        verbose=1
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_dice_coef', 
        patience=15, 
        restore_best_weights=True, 
        mode='max', 
        verbose=1
    )
    model_checkpoint = callbacks.ModelCheckpoint(
        save_path, 
        monitor='val_dice_coef', 
        save_best_only=True, 
        mode='max', 
        verbose=1,
        save_weights_only=False
    )
    
    callbacks_list = [reduce_lr, early_stopping, model_checkpoint]
    
    print("Starting training with batch size 1...")
    try:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=1,
            epochs=30,
            callbacks=callbacks_list,
            verbose=1,
            shuffle=True
        )
        
        print(f"Training completed successfully!")
        
    except Exception as e:
        print(f"Training error: {e}")
        return None, None
    
    try:
        model.save(save_path)
        print(f"Model saved successfully at: {save_path}")
    except Exception as e:
        print(f"Save error: {e}")
        fallback_path = "model_weights.h5"
        model.save_weights(fallback_path)
        print(f"Model weights saved at: {fallback_path}")
    
    plt.figure(figsize=(15, 10))
    metrics_to_plot = ['loss', 'dice_coef']
    if 'jaccard_index' in history.history:
        metrics_to_plot.extend(['jaccard_index', 'precision_metric', 'recall_metric', 'f1_metric'])
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        if metric in history.history:
            plt.plot(history.history[metric], label=f'Training {metric.replace("_", " ").title()}')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.replace("_", " ").title()}')
        plt.title(f'{metric.replace("_", " ").title()} Convergence')
        plt.xlabel('Epochs')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'training_convergence_colab.png'), dpi=200, bbox_inches='tight')
    plt.show()
    
    print("Evaluating on test set...")
    try:
        test_predictions = model.predict(x_test, batch_size=1, verbose=1)
        test_predictions_binary = (test_predictions > 0.5).astype(np.float32)
        
        y_test_binary = (y_test.flatten() > 0.5).astype(int)
        test_predictions_binary_flat = (test_predictions_binary.flatten() > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score
        test_accuracy = accuracy_score(y_test_binary, test_predictions_binary_flat)
        test_dice = float(dice_coef(y_test, test_predictions).numpy())
        
        results = {
            'Model': 'ReVGG-R2Net-Colab',
            'Accuracy': float(test_accuracy),
            'Dice_Coefficient': test_dice,
            'Training_Epochs': len(history.history['loss']),
            'Batch_Size': 1,
            'Dataset_Size': {
                'Training': len(x_train),
                'Validation': len(x_val),
                'Test': len(x_test)
            }
        }
        
        try:
            test_jaccard = float(jaccard_index(y_test, test_predictions).numpy())
            test_precision = float(precision_metric(y_test, test_predictions).numpy())
            test_recall = float(recall_metric(y_test, test_predictions).numpy())
            test_f1 = float(f1_metric(y_test, test_predictions).numpy())
            
            results.update({
                'Jaccard_Index': test_jaccard,
                'Precision': test_precision,
                'Recall': test_recall,
                'F1_Score': test_f1
            })
        except Exception as metric_error:
            print(f"Extended metrics calculation failed: {metric_error}")
            print("Using basic metrics only")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        results = {
            'Model': 'ReVGG-R2Net-Colab',
            'Training_Epochs': len(history.history['loss']),
            'Batch_Size': 1,
            'Status': 'Training completed, evaluation failed'
        }
    
    with open(os.path.join(config.RESULTS_DIR, 'colab_training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*60)
    print("COLAB TRAINING RESULTS")
    print("="*60)
    print(f"Model: {results['Model']}")
    if 'Accuracy' in results:
        print(f"Accuracy: {results['Accuracy']:.4f}")
        print(f"Dice Coefficient: {results['Dice_Coefficient']:.4f}")
        if 'Jaccard_Index' in results:
            print(f"Jaccard Index: {results['Jaccard_Index']:.4f}")
            print(f"Precision: {results['Precision']:.4f}")
            print(f"Recall: {results['Recall']:.4f}")
            print(f"F1-Score: {results['F1_Score']:.4f}")
    print(f"Training Epochs: {results['Training_Epochs']}")
    print(f"Batch Size: {results['Batch_Size']}")
    print("="*60)
    
    return model, history

if __name__ == "__main__":
    data_path = "/content/drive/MyDrive/test"
    model, history = train_model_colab(data_path)
