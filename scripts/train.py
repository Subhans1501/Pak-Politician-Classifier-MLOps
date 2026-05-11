import os
import mlflow
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.optimizers import Adam
from model_architectures import build_resnet50_finetuned, build_efficientnet_finetuned

def build_compiled_resnet():
    model = build_resnet50_finetuned()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_compiled_effnet():
    model = build_efficientnet_finetuned()

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models():
    print("Setting up Data Generators with Model-Specific Preprocessing...")
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,
        rotation_range=20, 
        width_shift_range=0.2,
        height_shift_range=0.2, 
        brightness_range=[0.8, 1.2],
        zoom_range=0.2, 
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)

    train_gen = train_datagen.flow_from_directory('dataset/split/train', target_size=(224, 224), batch_size=32)
    val_gen = val_datagen.flow_from_directory('dataset/split/val', target_size=(224, 224), batch_size=32)

    print("Calculating strict class weights...")
    weights = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
    class_weight_dict = dict(enumerate(weights))

    models = {
        'ResNet50': build_compiled_resnet(), 
        'EfficientNetB0': build_compiled_effnet()
    }
    
    os.makedirs('saved_models', exist_ok=True)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")

        if name == 'EfficientNetB0':
            train_gen.image_data_generator.preprocessing_function = effnet_preprocess
            val_gen.image_data_generator.preprocessing_function = effnet_preprocess

        with mlflow.start_run(run_name=name):
            mlflow.log_param("architecture", name)
            mlflow.log_param("learning_rate", 0.0001)
            
            model.fit(
                train_gen, 
                epochs=20, 
                validation_data=val_gen, 
                class_weight=class_weight_dict
            )
            
            model.save(f'saved_models/{name}_politicians.h5')
            mlflow.log_artifact(f'saved_models/{name}_politicians.h5')

if __name__ == "__main__":
    train_models()