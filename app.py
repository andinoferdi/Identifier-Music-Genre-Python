from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import numpy as np
import librosa
import os

app = Flask(__name__, static_folder='static')

# Load the model and preprocessing components
model_path = 'knn_genre_classifier.joblib'
model, scaler, pca = joblib.load(model_path)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio-file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['audio-file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        features = extract_features(file_path).reshape(1, -1)
        features = scaler.transform(features)
        features = pca.transform(features)
        predicted_genre = model.predict(features)
        return jsonify({'genre': predicted_genre[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(file_path)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
