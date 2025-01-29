import os
from src.preprocessing.audio_processing import preprocess_audio
from src.models.model_training import train_model

def main():
    # Paso 1: Preprocesar datos de audio
    data_path = "data/raw"
    processed_path = "data/processed"
    preprocess_audio(data_path, processed_path)

    # Paso 2: Entrenar un modelo de IA
    train_model(processed_path)

if __name__ == "__main__":
    main()
