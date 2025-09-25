import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import Config
from utils.data_loader import load_and_preprocess_data
from utils.metrics import dice_coef, jaccard_index, precision_metric, recall_metric, f1_metric, get_custom_objects

config = Config()

def evaluate_model(model_path, test_data_path, save_results=True):
    config.create_directories()
    
    print("Loading saved model...")
    custom_objects = get_custom_objects()
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    print("Loading test data...")
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_preprocess_data(test_data_path)
    
    print("Making predictions on test set...")
    test_predictions = model.predict(x_test, batch_size=1, verbose=1)
    test_predictions_binary = (test_predictions > 0.5).astype(np.float32)
    
    y_test_flat = np.array(y_test).flatten()
    test_predictions_flat = np.array(test_predictions_binary).flatten()
    test_predictions_prob_flat = np.array(test_predictions).flatten()
    
    y_test_binary = (y_test_flat > 0.5).astype(int)
    test_predictions_binary_int = test_predictions_flat.astype(int)
    
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
    print("EVALUATION RESULTS - REVGG-R2NET")
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
