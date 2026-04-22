# 🎤 SIA — Speech and Image Analysis (Keyword Spotting System)

A real-time, offline voice keyword detection system built with a lightweight CNN trained on MFCC audio features. Speak into your microphone — SIA detects it instantly.

> Inspired by "Hey Siri" / "OK Google" — but built from scratch.

---

## 🧠 How It Works

SIA converts 1-second audio clips into **MFCC (Mel-Frequency Cepstral Coefficients)** — a 2D numerical fingerprint of sound — and feeds them into a small **Convolutional Neural Network (CNN)** to classify the spoken word in real time.

```
🎤 Microphone → Normalize → MFCC Features → CNN Model → Detected Keyword
```

A **smoothing window** (majority vote over last 3 predictions) prevents false triggers from background noise.

---

## 🔑 Supported Keywords

| Label | Description |
|-------|-------------|
| `yes` | Confirmation |
| `no` | Rejection |
| `on` | Turn on |
| `off` | Turn off |
| `up` | Move / scroll up |
| `down` | Move / scroll down |
| `left` | Move left |
| `right` | Move right |
| `unknown` | Any other word |
| `silence` | Background noise / no speech |

---

## 📁 Project Structure

```
SIA/
│
├── train_SIA.ipynb        # Model training (Google Colab, GPU recommended)
├── testing_SIA.py         # Real-time inference from microphone
├── kws_mfcc_cnn.h5        # Trained model (generated after training)
└── README.md
```

---

## ⚙️ Model Architecture

A lightweight CNN with only **~289K parameters** (~3.5MB saved model):

```
Input: MFCC (20 × 101 × 1)
  → Conv2D(16, 3×3) + BatchNorm + MaxPool
  → Conv2D(32, 3×3) + BatchNorm + MaxPool
  → Flatten
  → Dense(128) + Dropout(0.5)
  → Dense(10, softmax)
```

---

## 📊 Model Performance

Trained for **40 epochs** on ~24,700 audio samples from [Google Speech Commands v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz).

| Metric | Score |
|--------|-------|
| Test Accuracy | **91.9%** |
| Macro F1-Score | **0.92** |

**Per-class F1 scores:**

| Word | Precision | Recall | F1 |
|------|-----------|--------|----|
| yes | 0.97 | 0.95 | 0.96 |
| right | 0.96 | 0.96 | 0.96 |
| on | 0.90 | 0.94 | 0.92 |
| left | 0.93 | 0.91 | 0.92 |
| down | 0.92 | 0.91 | 0.91 |
| no | 0.88 | 0.92 | 0.90 |
| off | 0.92 | 0.86 | 0.89 |
| up | 0.87 | 0.91 | 0.89 |
| silence | 1.00 | 0.90 | 0.95 |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/SIA.git
cd SIA
```

### 2. Install Dependencies

```bash
pip install tensorflow librosa sounddevice numpy
```

> **Note:** On some systems, `sounddevice` requires PortAudio. Install it with:
> - **Ubuntu/Debian:** `sudo apt install portaudio19-dev`
> - **macOS:** `brew install portaudio`
> - **Windows:** Usually works out of the box.

### 3. Train the Model (Optional — Skip if you have `kws_mfcc_cnn.h5`)

Open `train_SIA.ipynb` in **Google Colab** (free T4 GPU recommended).

The notebook will:
1. Download the Google Speech Commands dataset (~2.3GB)
2. Extract MFCC features from all audio clips
3. Train the CNN for 40 epochs
4. Save and download `kws_mfcc_cnn.h5` to your machine

### 4. Run Real-Time Detection

Make sure `kws_mfcc_cnn.h5` is in the same folder as `testing_SIA.py`, then:

```bash
python testing_SIA.py
```

You should see:
```
🎤 Listening... Speak a keyword
Detected: yes (0.94)
Detected: left (0.88)
```

---

## 🔧 Configuration

You can tune these parameters in `testing_SIA.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `DURATION` | `1` | Recording window (seconds) |
| `THRESHOLD` | `0.7` | Minimum confidence to trigger detection |
| `SMOOTHING_WINDOW` | `3` | Number of recent predictions for majority vote |
| `N_MFCC` | `20` | Number of MFCC coefficients |

---

## 💡 Use Cases

- 🏠 **Smart Home Control** — Trigger lights/devices with "on" / "off"
- 🤖 **Robotics** — Send directional commands: "left", "right", "up", "down"
- ♿ **Accessibility Tools** — Hands-free UI navigation
- 📱 **Edge AI / IoT** — Runs fully offline, no cloud required
- 🎮 **Game Control** — Voice-activated in-game commands

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `TensorFlow / Keras` | Model training and inference |
| `Librosa` | Audio loading and MFCC extraction |
| `SoundDevice` | Live microphone recording |
| `NumPy` | Audio normalization and array ops |
| `scikit-learn` | Train/test split and evaluation metrics |

---

## 📋 Requirements

```
tensorflow>=2.0
librosa>=0.9
sounddevice
numpy
scikit-learn
```

---

## 📌 Notes

- The model listens in **1-second chunks** continuously in a loop
- Audio is **normalized** before feature extraction so volume differences don't affect accuracy
- The **smoothing window** requires the same word to appear in at least 2 of the last 3 predictions before reporting — this reduces false positives significantly
- Training was done on a **T4 GPU** in Google Colab — each epoch takes ~3–4 seconds

---

## 📄 License

MIT License — feel free to use, modify, and build on top of this project.

---

## 🙌 Acknowledgements

- [Google Speech Commands Dataset v0.02](https://arxiv.org/abs/1804.03209) by Pete Warden
- [Librosa](https://librosa.org/) for audio processing
- [TensorFlow](https://www.tensorflow.org/) for model training
