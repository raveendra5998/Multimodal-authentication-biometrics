from pathlib import Path

import numpy as np
from PIL import Image
import soundfile as sf

from Biometrics import collect_subjects, extract_audio_features, extract_face_features


def test_collect_subjects_detects_matching_folders(tmp_path: Path) -> None:
    (tmp_path / "face" / "01").mkdir(parents=True)
    (tmp_path / "face" / "02").mkdir(parents=True)
    (tmp_path / "Audio" / "01").mkdir(parents=True)
    (tmp_path / "Audio" / "02").mkdir(parents=True)

    subjects = collect_subjects(tmp_path / "face", tmp_path / "Audio")

    assert subjects == ["01", "02"]


def test_feature_extractors_return_non_empty_vectors(tmp_path: Path) -> None:
    face_path = tmp_path / "face_sample.png"
    img = np.zeros((128, 128), dtype=np.uint8)
    Image.fromarray(img).save(face_path)

    face_features = extract_face_features(face_path)
    assert face_features.shape[0] > 0

    audio_path = tmp_path / "audio_sample.wav"
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(audio_path, audio, sr)

    audio_features = extract_audio_features(audio_path)
    assert audio_features.shape[0] > 0
