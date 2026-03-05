import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import collections

# ================= PARAMETERS =================
SAMPLE_RATE = 16000
DURATION = 1
SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 20
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40

THRESHOLD = 0.7
SMOOTHING_WINDOW = 3

# =============================================

LABELS = [
    'yes','no','on','off',
    'up','down','left','right',
    'unknown','silence'
]

model = tf.keras.models.load_model("kws_mfcc_cnn.h5")
prediction_queue = collections.deque(maxlen=SMOOTHING_WINDOW)

# =============================================

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mfcc = mfcc[..., np.newaxis]
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc

# =============================================

print("🎤 Listening... Speak a keyword")

while True:
    audio = sd.rec(SAMPLES, samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    audio = audio.flatten()
    audio = audio / np.max(np.abs(audio) + 1e-9)

    mfcc = extract_mfcc(audio)
    prediction = model.predict(mfcc, verbose=0)[0]

    confidence = np.max(prediction)
    label_index = np.argmax(prediction)

    prediction_queue.append(label_index)

    if prediction_queue.count(label_index) >= 2 and confidence >= THRESHOLD:
        print(f"Detected: {LABELS[label_index]} ({confidence:.2f})")
