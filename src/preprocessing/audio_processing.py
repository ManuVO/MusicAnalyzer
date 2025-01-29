import librosa
import os
import numpy as np

def preprocess_audio(input_folder, output_folder):
    """Procesa archivos de audio y guarda los espectrogramas en un directorio de salida."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            audio_path = os.path.join(input_folder, file)
            y, sr = librosa.load(audio_path, duration=10)
            
            # Extraer espectrograma
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Guardar como archivo .npy
            output_file = os.path.join(output_folder, file.replace(".wav", ".npy"))
            np.save(output_file, mel_spec_db)
            print(f"Procesado: {file}")
