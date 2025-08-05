import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=8000)
    features = {
        'zcr': np.mean(librosa.feature.zero_crossing_rate(y)),
        'energy': np.mean(y ** 2),
        'rmse': np.mean(librosa.feature.rms(y=y)),
        'mfcc1': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[0]),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
    }
    return features
