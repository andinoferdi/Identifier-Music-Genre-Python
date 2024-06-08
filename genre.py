import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
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
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features = np.hstack((
        tempo,
        chroma_stft.mean(),
        spectral_contrast.mean(),
        mfcc.mean(axis=1),
        zero_crossing_rate.mean(),
        spectral_rolloff.mean()
    ))
    return features

# Definisikan scaler dan pca di luar blok if-else
scaler = StandardScaler()
pca = PCA(n_components=0.95)  # Menjaga 95% varians

# Path to the dataset and processed data file
dataset_path = 'MusicDataset'  # Path relatif ke dataset Anda
processed_data_file = 'processed_data.joblib'

# Check if the processed data file exists
if os.path.exists(processed_data_file):
    # Load processed data
    print("Loading processed data...")
    X_train, X_test, y_train, y_test, scaler, pca = joblib.load(processed_data_file)
else:
    # Load dataset and extract features
    print("Loading dataset...")
    X, y = load_dataset(dataset_path)

    # Normalisasi fitur dan reduksi dimensi dengan PCA
    X = scaler.fit_transform(X)
    X = pca.fit_transform(X)

    # Split dataset into training and testing sets
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save processed data with scaler and PCA
    print("Saving processed data...")
    joblib.dump((X_train, X_test, y_train, y_test, scaler, pca), processed_data_file)

# Tentukan jumlah lipatan berdasarkan jumlah sampel terkecil di setiap kelas
unique, counts = np.unique(y_train, return_counts=True)
min_samples = min(counts)
n_splits = max(2, min_samples)  # Pastikan n_splits minimal 2

# Gunakan StratifiedKFold untuk mempertahankan distribusi kelas
stratified_kfold = StratifiedKFold(n_splits=n_splits)

# Hyperparameter tuning with GridSearchCV
params = {'n_neighbors': range(1, 10)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, params, cv=stratified_kfold)
grid_search.fit(X_train, y_train)

# Best KNN classifier
best_knn = grid_search.best_estimator_

# Evaluate the classifier with cross-validation
print("Evaluating classifier with cross-validation...")
scores = cross_val_score(best_knn, X_train, y_train, cv=stratified_kfold)
print(f"Cross-validation scores: {scores.mean()}")

# Train the best KNN classifier
print("Training classifier...")
best_knn.fit(X_train, y_train)

# Evaluate the classifier
print("Evaluating classifier...")
y_pred = best_knn.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Simpan model dan komponen preprocessing
joblib.dump((best_knn, scaler, pca), 'knn_genre_classifier.joblib')

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
    features = scaler.transform(features)  # Normalisasi fitur
    features = pca.transform(features)     # Reduksi dimensi
    predicted_genre = best_knn.predict(features)
    print(f"The predicted genre is: {predicted_genre[0]}")
except Exception as e:
    print(f"Error processing {audio_file_path}: {e}")
