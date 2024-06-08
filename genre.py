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

# Perbaiki masalah numpy complex jika diperlukan
if not hasattr(np, 'complex'):
    np.complex = np.complex128

# Fungsi untuk mengekstrak fitur dari file audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features = np.hstack((tempo, chroma_stft.mean(), spectral_contrast.mean(),
                          mfcc.mean(axis=1), zero_crossing_rate.mean(), spectral_rolloff.mean()))
    return features

# Inisialisasi scaler dan PCA
scaler = StandardScaler()
pca = PCA(n_components=0.95)

# Jalur ke dataset dan file data yang diproses
dataset_path = 'MusicDataset'
processed_data_file = 'processed_data.joblib'

# Memuat data yang diproses atau mengekstrak fitur dari dataset
if os.path.exists(processed_data_file):
    print("Memuat data yang diproses...")
    X_train, X_test, y_train, y_test, scaler, pca = joblib.load(processed_data_file)
else:
    print("Memuat dataset...")
    X, y = load_dataset(dataset_path)
    X = scaler.fit_transform(X)
    X = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Menyimpan data yang diproses...")
    joblib.dump((X_train, X_test, y_train, y_test, scaler, pca), processed_data_file)

# Penyetelan hiperparameter dan pelatihan pengklasifikasi
unique, counts = np.unique(y_train, return_counts=True)
min_samples = min(counts)
n_splits = max(2, min_samples)
stratified_kfold = StratifiedKFold(n_splits=n_splits)
params = {'n_neighbors': range(1, 10)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, params, cv=stratified_kfold)
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_

# Validasi silang dan evaluasi pengklasifikasi
print("Mengevaluasi pengklasifikasi dengan validasi silang...")
scores = cross_val_score(best_knn, X_train, y_train, cv=stratified_kfold)
print(f"Cross-validation scores: {scores.mean()}")
best_knn.fit(X_train, y_train)
print("Mengevaluasi pengklasifikasi...")
y_pred = best_knn.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Simpan model dan komponen preprocessing
joblib.dump((best_knn, scaler, pca), 'knn_genre_classifier.joblib')

# Fungsi untuk memilih dan memprediksi genre file audio baru
def select_audio_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()

audio_file_path = select_audio_file()
print(f"Memprediksi genre untuk {audio_file_path}...")
try:
    features = extract_features(audio_file_path).reshape(1, -1)
    features = scaler.transform(features)
    features = pca.transform(features)
    predicted_genre = best_knn.predict(features)
    print(f"Genre yang diprediksi adalah: {predicted_genre[0]}")
except Exception as e:
    print(f"Kesalahan pemrosesan {audio_file_path}: {e}")
