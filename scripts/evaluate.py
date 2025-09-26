import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import Config
from utils.data_loader import load_and_preprocess_data
from utils.metrics import dice_coef, jaccard_index, precision_metric, recall_metric, f1_metric, get_custom_objects
from models.revgg_r2net import create_revgg_r2net

config = Config()

def find_model_files(search_path):
    model_files = {}
    search_dirs = []
    
    if os.path.isfile(search_path):
        search_dirs = [os.path.dirname(search_path)]
    elif os.path.isdir(search_path):
        search_dirs = [search_path]
    else:
        search_dirs = ["pretrained", ".", "models"]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.keras'):
                    model_files['keras'] = file_path
                elif file.endswith('.h5') and 'weights' in file.lower():
                    model_files['weights'] = file_path
                elif file.endswith('.h5'):
                    model_files['h5'] = file_path
    
    return model_files

def load_model_robust(model_path):
    model_files = find_model_files(model_path)
    
    if not model_files:
        raise FileNotFoundError(f"No model files found. Searched in: {model_path}")
    
    print("Available model files:")
    for key, path in model_files.items():
        print(f"  {key}: {path}")
    
    custom_objects = get_custom_objects()
    
    if 'keras' in model_files:
        try:
            print(f"Loading .keras model: {model_files['keras']}")
            model = tf.keras.models.load_model(model_files['keras'], custom_objects=custom_objects)
            print("Model loaded successfully from .keras file")
            return model
        except Exception as e:
            print(f"Failed to load .keras file: {e}")
    
    if 'weights' in model_files:
        try:
            print(f"Loading weights: {model_files['weights']}")
            model = create_revgg_r2net()
            model.load_weights(model_files['weights'])
            print("Model loaded successfully from weights file")
            return model
        except Exception as e:
            print(f"Failed to load weights: {e}")
    
    if 'h5' in model_files:
        try:
            print(f"Loading .h5 model: {model_files['h5']}")
            model = tf.keras.models.load_model(model_files['h5'], custom_objects=custom_objects)
            print("Model loaded successfully from .h5 file")
            return model
        except Exception as e:
            print(f"Failed to load .h5 file: {e}")
            try:
                print("Trying to load as weights...")
                model = create_revgg_r2net()
                model.load_weights(model_files['h5'])
                print("Model loaded successfully as weights")
                return model
            except Exception as e2:
                print(f"Also failed as weights: {e2}")
    
    raise ValueError("Could not load model from any available files")

def evaluate_model(model_path, test_data_path, save_results=True):
    config.create_directories()
    
    print("Loading saved model...")
    model = load_model_robust(model_path)
    
    print("Loading test data...")
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_preprocess_data(test_data_path)
    
    print("Making predictions on test set...")
    test_predictions = model.predict(x_test, batch_size=1, verbose=1)
    test_predictions_binary = (test_predictions > 0.5).astype(np.float32)
    
    y_test_flat = np.array(y_test).flatten()
    test_predictions_flat = np.array(test_predictions_binary).flatten()
    test_predictions_prob_flat = np.array(test_predictions).flatten()
    
    y_test_binary = (y_test_flat > 0.5).astype(int)
    test_predictions_binary_int = (test_predictions_flat > 0.5).astype(int)
    
    test_accuracy = accuracy_score(y_test_binary, test_predictions_binary_int)
    test_precision_sklearn = precision_score(y_test_binary, test_predictions_binary_int, average='binary', zero_division=0)
    test_recall_sklearn = recall_score(y_test_binary, test_predictions_binary_int, average='binary', zero_division=0)
    test_f1_sklearn = f1_score(y_test_binary, test_predictions_binary_int, average='binary', zero_division=0)
    
    test_dice = float(dice_coef(y_test, test_predictions).numpy())
    test_jaccard = float(jaccard_index(y_test, test_predictions).numpy())
    test_precision_custom = float(precision_metric(y_test, test_predictions).numpy())
    test_recall_custom = float(recall_metric(y_test, test_predictions).numpy())
    test_f1_custom = float(f1_metric(y_test, test_predictions).numpy())
    
    results = {
        'Model': 'ReVGG-R2Net',
        'Accuracy': float(test_accuracy),
        'Dice_Coefficient': test_dice,
        'Jaccard_Index': test_jaccard,
        'Precision_Custom': test_precision_custom,
        'Recall_Custom': test_recall_custom,
        'F1_Score_Custom': test_f1_custom,
        'Precision_Sklearn': float(test_precision_sklearn),
        'Recall_Sklearn': float(test_recall_sklearn),
        'F1_Score_Sklearn': float(test_f1_sklearn),
        'Dataset_Size': {
            'Test_Samples': len(x_test),
            'Total_Pixels_Tested': len(y_test_binary),
            'Positive_Pixels': int(np.sum(y_test_binary)),
            'Negative_Pixels': int(len(y_test_binary) - np.sum(y_test_binary))
        }
    }
    
    if save_results:
        with open(os.path.join(config.RESULTS_DIR, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS - REVGG-R2NET V5")
    print("="*70)
    print(f"Model: {results['Model']}")
    print(f"Test Accuracy: {results['Accuracy']:.4f}")
    print(f"Dice Coefficient: {results['Dice_Coefficient']:.4f}")
    print(f"Jaccard Index: {results['Jaccard_Index']:.4f}")
    print(f"Precision (Custom): {results['Precision_Custom']:.4f}")
    print(f"Recall (Custom): {results['Recall_Custom']:.4f}")
    print(f"F1-Score (Custom): {results['F1_Score_Custom']:.4f}")
    print(f"Precision (Sklearn): {results['Precision_Sklearn']:.4f}")
    print(f"Recall (Sklearn): {results['Recall_Sklearn']:.4f}")
    print(f"F1-Score (Sklearn): {results['F1_Score_Sklearn']:.4f}")
    print(f"Test Samples: {results['Dataset_Size']['Test_Samples']}")
    print("="*70)
    
    return results, y_test_binary, test_predictions_prob_flat

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <model_path> <test_data_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    
    results, y_true, y_pred_prob = evaluate_model(model_path, test_data_path)
