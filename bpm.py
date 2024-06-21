import librosa
import tkinter as tk
from tkinter.filedialog import askopenfilename

# Fungsi untuk mengidentifikasi BPM
def identify_bpm(file_path):
    # Memuat file audio
    y, sr = librosa.load(file_path, sr=None)
    
    # Menghitung tempo dan beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    return tempo

# Membuat instance Tkinter
root = tk.Tk()
root.withdraw()  # Menyembunyikan jendela utama Tkinter

# Meminta pengguna untuk memilih file lagu melalui dialog file
file_path = askopenfilename()

# Memanggil fungsi dengan file path yang dipilih pengguna
bpm = identify_bpm(file_path)
print(f"BPM lagu adalah: {bpm} BPM")
