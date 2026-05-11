import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

def evaluate_specific_model(model_name, model_path, preprocess_fn):
    print(f"\nEvaluating {model_name}...")

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_gen = test_datagen.flow_from_directory(
        'dataset/split/test', 
        target_size=(224, 224), 
        batch_size=32, 
        shuffle=False 
    )
    model = load_model(model_path)
    predictions = np.argmax(model.predict(test_gen), axis=-1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    print(f"\nClassification Report for {model_name}:")
    print(classification_report(true_classes, predictions, target_names=class_labels))

    cm = confusion_matrix(true_classes, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = f'results/confusion_matrix_{model_name}.png'
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    evaluate_specific_model('ResNet50', 'saved_models/ResNet50_politicians.h5', resnet_preprocess)
    evaluate_specific_model('EfficientNetB0', 'saved_models/EfficientNetB0_politicians.h5', effnet_preprocess)