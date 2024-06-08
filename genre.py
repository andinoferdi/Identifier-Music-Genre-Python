import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import tkinter as tk
from tkinter import filedialog

# Fix the numpy complex issue
if not hasattr(np, 'complex'):
    np.complex = np.complex128

# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    features = np.hstack((
        tempo,
        chroma_stft.mean(),
        spectral_contrast.mean(),
        mfcc.mean(axis=1)
    ))
    return features

# Definisikan scaler dan pca di luar blok if-else
scaler = StandardScaler()
pca = PCA(n_components=0.95)  # Menjaga 95% varians

# Check if the processed data file exists
processed_data_file = 'processed_data.joblib'
if os.path.exists(processed_data_file):
    # Load processed data
    print("Loading processed data...")
    X_train, X_test, y_train, y_test, scaler, pca = joblib.load(processed_data_file)
else:
    # Load dataset and extract features
    def load_dataset(dataset_path):
        genres = os.listdir(dataset_path)
        X, y = [], []
        for genre in genres:
            genre_path = os.path.join(dataset_path, genre)
            for file in glob.glob(os.path.join(genre_path, "*.mp3")):
                try:
                    features = extract_features(file)
                    X.append(features)
                    y.append(genre)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        return np.array(X), np.array(y)

    # Path to the dataset
    dataset_path = 'MusicDataset'  # Path relatif ke dataset Anda

    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset(dataset_path)

    # Check if the dataset is loaded correctly
    if X.size == 0:
        raise ValueError("Dataset is empty. Please check the dataset path and contents.")

    # Normalisasi fitur
    X = scaler.fit_transform(X)

    # Reduksi dimensi dengan PCA
    X = pca.fit_transform(X)

    # Split dataset into training and testing sets
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save processed data with scaler and PCA
    joblib.dump((X_train, X_test, y_train, y_test, scaler, pca), processed_data_file)

# Adjust n_neighbors based on the number of samples
n_neighbors = min(len(X_train), 5)

# Train KNN classifier
print("Training classifier...")
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)

# Evaluate the classifier
print("Evaluating classifier...")
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Simpan model
joblib.dump(knn, 'knn_genre_classifier.joblib')

# Fungsi untuk memilih file audio
def select_audio_file():
    root = tk.Tk()
    root.withdraw()  # Menyembunyikan jendela tkinter
    file_path = filedialog.askopenfilename()
    return file_path

# Predict genre for a new audio file
audio_file_path = select_audio_file()
print(f"Predicting genre for {audio_file_path}...")

try:
    features = extract_features(audio_file_path).reshape(1, -1)
    
    # Pastikan jumlah fitur sesuai dengan yang diharapkan oleh scaler dan pca
    if features.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Jumlah fitur tidak sesuai. Model diharapkan {scaler.n_features_in_} fitur, tetapi mendapatkan {features.shape[1]} fitur.")
    
    features = scaler.transform(features)  # Normalisasi fitur
    features = pca.transform(features)     # Reduksi dimensi
    predicted_genre = knn.predict(features)
    print(f"The predicted genre is: {predicted_genre[0]}")
except Exception as e:
    print(f"Error processing {audio_file_path}: {e}")
