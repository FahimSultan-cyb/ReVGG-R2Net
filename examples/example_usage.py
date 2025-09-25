#!/usr/bin/env python3

import os
import sys

from scripts.train import train_model
from scripts.inference import load_model_and_predict
from scripts.evaluate import evaluate_model

def example_training():
    print("="*50)
    print("EXAMPLE 1: Training ReVGG-R2Net Model")
    print("="*50)
    
    data_path = "path/to/your/dataset"
    model_save_path = "models/my_revgg_r2net.keras"
    
    print(f"Training with data from: {data_path}")
    print(f"Model will be saved to: {model_save_path}")
    
    try:
        model, history, results = train_model(data_path, model_save_path)
        print("Training completed successfully!")
        print(f"Final Dice Score: {results['Dice_Coefficient']:.4f}")
        return model_save_path
    except Exception as e:
        print(f"Training failed: {e}")
        return None

def example_inference(model_path):
    print("="*50)
    print("EXAMPLE 2: Model Inference")
    print("="*50)
    
    single_image = "path/to/single/image.jpg"
    image_directory = "path/to/test/images"
    output_directory = "results/predictions"
    
    print("Single image prediction:")
    try:
        prediction, binary_pred = load_model_and_predict(
            model_path=model_path,
            image_path=single_image,
            threshold=0.5,
            output_dir=output_directory
        )
        print(f"Prediction shape: {prediction.shape}")
        print(f"Max probability: {prediction.max():.4f}")
        print(f"Results saved to: {output_directory}")
    except Exception as e:
        print(f"Single image prediction failed: {e}")
    
    print("\nBatch prediction on directory:")
    try:
        results = load_model_and_predict(
            model_path=model_path,
            image_path=image_directory,
            threshold=0.5,
            output_dir=output_directory
        )
        print(f"Processed {len(results['filenames'])} images")
        print(f"Results saved to: {output_directory}")
    except Exception as e:
        print(f"Batch prediction failed: {e}")

def example_evaluation(model_path):
    print("="*50)
    print("EXAMPLE 3: Model Evaluation")
    print("="*50)
    
    test_data_path = "path/to/test/dataset"
    
    try:
        results, y_true, y_pred = evaluate_model(model_path, test_data_path)
        print("Evaluation completed successfully!")
        print(f"Test Accuracy: {results['Accuracy']:.4f}")
        print(f"Dice Coefficient: {results['Dice_Coefficient']:.4f}")
        print(f"Jaccard Index: {results['Jaccard_Index']:.4f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")

def example_model_architecture():
    print("="*50)
    print("EXAMPLE 4: Model Architecture Inspection")
    print("="*50)
    
    from models.revgg_r2net import create_revgg_r2net
    
    model = create_revgg_r2net()
    
    print("Model Summary:")
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    print("\nEncoder layers (VGG16 backbone):")
    vgg_layers = [layer for layer in model.layers if 'vgg16' in layer.name]
    for layer in vgg_layers[:5]:
        print(f"  {layer.name}: {layer.output_shape}")

def main():
    print("ReVGG-R2Net Usage Examples")
    print("="*50)
    
    print("\nAvailable examples:")
    print("1. Training a new model")
    print("2. Model inference")
    print("3. Model evaluation")
    print("4. Architecture inspection")
    
    print("\n" + "="*50)
    print("NOTE: Update the file paths in this script to match your data locations")
    print("="*50)
    
    example_model_architecture()
    
    model_path = "pretrained/ReVGG_R2Net_final.keras"
    
    if os.path.exists(model_path):
        print(f"\nUsing existing model: {model_path}")
        example_inference(model_path)
        example_evaluation(model_path)
    else:
        print(f"\nModel not found at {model_path}")
        print("Run training example first or provide correct model path")
        trained_model_path = example_training()
        if trained_model_path:
            example_inference(trained_model_path)
            example_evaluation(trained_model_path)

if __name__ == "__main__":
    main()
