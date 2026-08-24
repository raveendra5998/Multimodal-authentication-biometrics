import time
import joblib
import librosa
import numpy as np
import cv2
import torch
import os
from facenet_pytorch import InceptionResnetV1

def extract_spectral_contrast(signal, sr):
    return librosa.feature.spectral_contrast(y=signal, sr=sr).flatten()

def get_voice_features(path):
    signal, sr = librosa.load(path, sr=16000)
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).flatten()
    sc = extract_spectral_contrast(signal, sr)
    feat = np.hstack((mfcc, sc))
    if len(feat) < 300:
        feat = np.pad(feat, (0, 300 - len(feat)), mode="constant")
    else:
        feat = feat[:300]
    return feat

def get_face_features(path, model, device):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_rgb = cv2.resize(img_rgb, (160, 160))
    tensor = (torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 127.5) - 1.0
    with torch.no_grad():
        emb = model(tensor.unsqueeze(0).to(device)).cpu().numpy()[0]
    return emb

if __name__ == "__main__":
    # Get the first image and audio file we can find for a quick test
    face_dir = "face"
    audio_dir = "Audio"
    
    # Just grabbing dummy files for user '01' to test speed
    try:
        user = input("Enter User ID: ").strip()

        face_user_dir = os.path.join(face_dir, user)
        audio_user_dir = os.path.join(audio_dir, user)

        face_file = os.path.join(face_user_dir, os.listdir(face_user_dir)[0])
        audio_file = os.path.join(audio_user_dir, os.listdir(audio_user_dir)[0])

    except Exception as e:
        print("Invalid User ID or files not found!")
        print(e)
        exit(1)

    print("Loading saved model and FaceNet (VGGFace2)...")
    try:
        clf = joblib.load("final_multimodal_model.pkl")
    except FileNotFoundError: 
        print("Error: 'final_multimodal_model.pkl' not found! Please let Biometrics.py finish running first so it can save the model.")
        exit(1)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    print(f"\nFiles selected for inference test:")
    print(f" - Face: {face_file}")
    print(f" - Voice: {audio_file}")
    
    print("\n--- Starting inference ---")
    t0 = time.time()
    
    # 1. Extract Face Features
    face_feat = get_face_features(face_file, face_model, device)
    
    # 2. Extract Voice Features
    voice_feat = get_voice_features(audio_file)
    
    # 3. Combine and Predict
    combined = np.hstack((face_feat, voice_feat)).reshape(1, -1)
    prediction = clf.predict(combined)
    
    total_time = time.time() - t0
    print(f"\nActual User: {user}")

    print(f"\n=> PREDICTION RESULT: User {prediction[0]}")
    print(f"=> INFERENCE TIME: {total_time:.4f} seconds!")
