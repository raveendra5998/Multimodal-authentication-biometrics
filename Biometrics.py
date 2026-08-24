#!/usr/bin/env python
# coding: utf-8
########################################################################
# Research Questions
'''RQ1: How does feature-level fusion of facial and voice
biometric data impact the overall system performance
compared to unimodal systems?

Feature-level fusion integrates complementary informa-
tion from face and voice modalities, leading to improved
authentication accuracy. Experiments show that multi-
modal systems outperform unimodal systems in Equal
Error Rate (EER) and ROC AUC metrics, demonstrating
the robustness of the fused approach.

RQ2: How does the use of SMOTE for balancing class
distributions affect the performance of classification
models in multimodal biometric authentication?

SMOTE effectively mitigates class imbalance, resulting
in improved precision, recall, and F1-scores across all
classes. Models trained on balanced datasets exhibited
consistent performance enhancements, particularly in
reducing false negatives for underrepresented classes.

RQ3: Which classifier (Random Forest, SVM, or kNN)
performs best in a multimodal biometric authentication
system based on facial and voice features?

SVM demonstrated the highest performance in the
multimodal biometric authentication system, achieving
the best accuracy and separability due to its efficiency
with high-dimensional data. Random Forest ranked
second, leveraging complex feature interactions, while
kNN showed limitations in scalability and parameter
sensitivity.'''

# ─── Libraries ────────────────────────────────────────────────────────────────
import os
import time
import numpy as np
import pandas as pd
import cv2
import librosa
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, roc_auc_score, auc)
import torch
from facenet_pytorch import InceptionResnetV1
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from collections import Counter
import matplotlib
matplotlib.use('Agg')          # non-interactive backend: saves files, never blocks
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings

warnings.filterwarnings("ignore")

# ─── Tunable Performance Parameters ──────────────────────────────────────────
#  Adjust these to trade speed vs. dataset coverage.

FACE_DATA_DIR    = "face"
VOICE_DATA_DIR   = "Audio"
RESULTS_DIR      = "Results02"

FACE_NUM_SPEAKERS      = 27      # synthetic speaker IDs "01"–"27" for flat face dir
FACE_IMG_SIZE          = 160     # px – InceptionResnetV1 expects 160×160
FACE_EMBED_DIM         = 512     # embedding dimension from VGGFace2 model
FACE_EMBED_PCA         = True   # set True to apply PCA to 512-dim face embeddings
AUGMENT_COPIES         = 3       # augmented copies per image – restored from 2;
                                  # applied INSIDE CV loop to train fold only
MAX_WAV_PER_SPEAKER    = 200     # raised from 50 – 4× more audio coverage
N_SPLITS               = 5       # CV folds
PCA_COMPONENTS         = 100     # reduce to 100 dims after scaling
RF_ESTIMATORS          = 200     # trees – restored from 100
N_JOBS                 = -1      # use all CPU cores
MIN_CLASS_SAMPLES_FOR_CV = 5     # warn when identity classes are too small for stable CV
MIN_SAMPLES_PER_CLASS    = 5     # [Fix 4] exclude identity classes below this count

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Startup Runtime-Impact Banner ────────────────────────────────────────────
print("=" * 70)
print("Biometric Authentication System  — Started Working...")
print("=" * 70)

# ─── Plot Saving (non-blocking) ───────────────────────────────────────────────
_plot_counter = [0]

def _save(title: str):
    """Save current figure to RESULTS_DIR and close it (never blocks)."""
    _plot_counter[0] += 1
    safe = title.replace(" ", "_").replace("/", "-").replace("\\", "-")
    path = os.path.join(RESULTS_DIR, f"{_plot_counter[0]:03d}_{safe}.png")
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close('all')
    print(f"  [plot saved] {path}")


# ─── Helper / Plotting Functions ──────────────────────────────────────────────

def plot_confusion_matrix(cm, classes, tag="", fold_count=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    title = f"Confusion Matrix {tag}"
    if fold_count is not None:
        title = f"{title} ({fold_count}-fold CV)"
    plt.title(title)
    plt.tight_layout()
    _save(title)


def plot_multiclass_roc_curve(classifier, X_test, y_test, classes, tag="", fold_count=None):
    y_test_binarized = label_binarize(y_test, classes=classes)
    y_score = classifier.predict_proba(X_test)

    fpr, tpr, roc_auc_vals = {}, {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc_vals[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i, cl in enumerate(classes):
        plt.plot(fpr[i], tpr[i],
                 label=f"Class {cl} (AUC={roc_auc_vals[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = f"ROC Curve {tag}"
    if fold_count is not None:
        title = f"{title} ({fold_count}-fold CV)"
    plt.title(title)
    plt.legend(loc="lower right", fontsize=6)
    plt.tight_layout()
    _save(title)

    macro_auc = roc_auc_score(y_test_binarized, y_score, average="macro")
    print(f"  Macro-average ROC AUC: {macro_auc:.4f}")
    return macro_auc


def calculate_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return fpr[idx], thresholds[idx]


def calculate_d_prime(genuine_scores, impostor_scores):
    mg, mi = np.mean(genuine_scores), np.mean(impostor_scores)
    sg, si = np.std(genuine_scores),  np.std(impostor_scores)
    denom = np.sqrt(0.5 * (sg**2 + si**2))
    return (mg - mi) / denom if denom > 0 else 0.0


def plot_score_distribution(genuine_scores, impostor_scores, tag="", fold_count=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(genuine_scores,  color='green', label='Genuine',
                 kde=True, stat='density')
    sns.histplot(impostor_scores, color='red',   label='Impostor',
                 kde=True, stat='density')
    plt.xlabel("Score")
    plt.ylabel("Density")
    title = f"Score Distribution {tag}"
    if fold_count is not None:
        title = f"{title} ({fold_count}-fold CV)"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _save(title)


# ─── Face Embedding Model ─────────────────────────────────────────────────────

def load_face_model():
    """Load InceptionResnetV1 (VGGFace2) once; move to GPU if available."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print(f"  Face embedding model: InceptionResnetV1 (VGGFace2) on {device}")
    return model, device


def embed_faces(pixel_arrays, model, device, batch_size=32):
    """
    Convert flat normalised pixel arrays → 512-dim L2-normalised face embeddings.
    Each pixel array is reshaped to (H, W), converted grayscale→RGB, resized to
    160×160, and passed through InceptionResnetV1.

    Returns an (N, 512) float32 numpy array.
    Called ONCE on the full dataset; result is cached and indexed by row.
    """
    embeddings = []
    size = FACE_IMG_SIZE
    for start in range(0, len(pixel_arrays), batch_size):
        batch_px  = pixel_arrays[start:start + batch_size]
        tensors   = []
        for px in batch_px:
            img_gray = (px.reshape(size, size) * 255).astype(np.uint8)
            img_rgb  = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)  # (H,W,3) uint8
            # Resize to 160×160 (InceptionResnetV1 expected input)
            img_rgb  = cv2.resize(img_rgb, (160, 160))
            # Normalize to [-1, 1] as expected by facenet-pytorch
            tensor   = (torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 127.5) - 1.0
            tensors.append(tensor)
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            emb = model(batch_tensor).cpu().numpy()  # (B, 512), L2-normalised by model
        embeddings.append(emb)
    return np.vstack(embeddings).astype(np.float32)


# ─── Image Feature Helpers ────────────────────────────────────────────────────

def extract_face_features(pixel_array):
    """
    TEST-TIME feature extraction: flat pixel array → 512-dim embedding.
    No augmentation is applied — used exclusively on test-fold images.
    """
    return embed_faces(pixel_array[np.newaxis], _face_model, _face_device)[0]


def augment_image(image):
    """Random rotation + flip + brightness shift on a (H, W) float [0,1] image."""
    rows, cols = image.shape
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated  = cv2.warpAffine(image, M, (cols, rows))
    flipped  = cv2.flip(rotated, random.choice([-1, 0, 1]))
    adjusted = np.clip(flipped * random.uniform(0.8, 1.2), 0, 1)
    return adjusted


def augment_and_extract_face_features(pixel_array):
    """
    TRAIN-TIME only: produces AUGMENT_COPIES augmented feature vectors from one
    base image.

    Returns a list of AUGMENT_COPIES 512-dim embedding arrays.
    NEVER call this on test-fold data.
    """
    size = FACE_IMG_SIZE
    image = pixel_array.reshape(size, size)
    aug_px = [augment_image(image).flatten() for _ in range(AUGMENT_COPIES)]
    return list(embed_faces(np.array(aug_px), _face_model, _face_device))


# ─── Voice Feature Helper ─────────────────────────────────────────────────────

def extract_spectral_contrast(signal, sr):
    return librosa.feature.spectral_contrast(y=signal, sr=sr).flatten()


# ─── Data Loading ─────────────────────────────────────────────────────────────

def create_face_labels(directory, num_speakers=FACE_NUM_SPEAKERS):
    """
    Loads face images from a flat directory (Caltech frontal-face dataset).
    Assigns synthetic speaker labels "01"–"{num_speakers:02d}" round-robin so
    they overlap with AudioMNIST speaker IDs.
    Falls back to subdirectory-based loading if subdirs are present.

    Fix 2 — Preprocessing: applies CLAHE (Contrast Limited Adaptive Histogram
    Equalisation) to each grayscale image before resize/normalise, correcting
    for lighting variation.

    Returns raw normalised pixel arrays (FACE_IMG_SIZE²).  Edge features are
    NOT concatenated here — that happens inside the CV loop so that augmented
    training copies and unaugmented test copies are handled separately.
    """
    images, labels = [], []
    size  = FACE_IMG_SIZE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    all_files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not all_files:
        # Subdirectory layout
        for label in os.listdir(directory):
            label_path = os.path.join(directory, label)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img = cv2.imread(os.path.join(label_path, file),
                                         cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = clahe.apply(img)          # Fix 2: CLAHE
                            images.append(
                                cv2.resize(img, (size, size)).flatten() / 255.0)
                            labels.append(label)
        print(f"Loaded {len(images)} face images with "
              f"{len(set(labels))} labels (subdirectory mode).")
        return np.array(images), np.array(labels)

    # Flat directory: contiguous block label assignment to reconstruct true subjects
    images_per_speaker = len(all_files) // num_speakers
    for idx, file in enumerate(all_files):
        img = cv2.imread(os.path.join(directory, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = clahe.apply(img)                          # Fix 2: CLAHE
            images.append(cv2.resize(img, (size, size)).flatten() / 255.0)
            speaker_idx = min((idx // images_per_speaker) + 1, num_speakers)
            labels.append(f"{speaker_idx:02d}")

    print(f"Loaded {len(images)} face images with "
          f"{len(set(labels))} labels (contiguous-block mode).")
    return np.array(images), np.array(labels)


def process_voice(directory, max_per_speaker=MAX_WAV_PER_SPEAKER):
    voice_features, voice_labels = [], []
    max_length = 300
    for label in sorted(os.listdir(directory)):
        label_path = os.path.join(directory, label)
        if not os.path.isdir(label_path):
            continue
        wav_files = [f for f in os.listdir(label_path)
                     if f.lower().endswith('.wav')]
        # Deterministic subsample
        wav_files = sorted(wav_files)[:max_per_speaker]

        for file in wav_files:
            path = os.path.join(label_path, file)
            try:
                signal, sr = librosa.load(path, sr=16000)

                # Fix 2: amplitude normalisation
                signal = signal / (np.max(np.abs(signal)) + 1e-9)
                # Fix 2: pre-emphasis filter (high-pass, reduces low-freq noise)
                signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

                mfcc = librosa.feature.mfcc(
                    y=signal, sr=sr, n_mfcc=13).flatten()
                sc   = extract_spectral_contrast(signal, sr)
                feat = np.hstack((mfcc, sc))
                if len(feat) < max_length:
                    feat = np.pad(feat, (0, max_length - len(feat)),
                                  mode="constant")
                else:
                    feat = feat[:max_length]
                voice_features.append(feat)
                voice_labels.append(label)
            except Exception as e:
                print(f"  Error processing {path}: {e}")

    print(f"Voice features extracted: {len(voice_features)}")
    return np.array(voice_features), np.array(voice_labels)


# ─── Alignment ────────────────────────────────────────────────────────────────

def align_face_voice_data(X_faces, y_faces, X_voices, y_voices, X_faces_raw=None):
    """
    Keeps only speakers common to both modalities.
    Pairs every face of Subject X with every voice of Subject X
    (1-to-1, min pairing) to ensure all valid non-random pairs are used.

    Fix 1: now operates on BASE (non-augmented) face pixel arrays.
    Augmentation happens inside the CV loop after the fold split.
    """
    common_labels = set(y_faces) & set(y_voices)
    print(f"Common users: {len(common_labels)}")

    X_fa, y_fa, X_va, X_fa_raw = [], [], [], []
    for label in sorted(common_labels):
        fi = [i for i, y in enumerate(y_faces)  if y == label]
        vi = [i for i, y in enumerate(y_voices) if y == label]

        # 1-to-1 pairing within the same subject
        n = min(len(fi), len(vi))
        for idx in range(n):
            X_fa.append(X_faces[fi[idx]])
            X_va.append(X_voices[vi[idx]])
            y_fa.append(label)
            if X_faces_raw is not None:
                X_fa_raw.append(X_faces_raw[fi[idx]])

    print(f"  Total Valid Pairs Generated: {len(y_fa)}")
    print(f"Aligned Face Data Shape:  {np.array(X_fa).shape}")
    print(f"Aligned Voice Data Shape: {np.array(X_va).shape}")
    return (np.array(X_fa, dtype=np.float32), np.array(y_fa),
            np.array(X_va, dtype=np.float32), np.array(y_fa),
            np.array(X_fa_raw, dtype=np.float32) if X_faces_raw is not None else None)


# ─── Thin Identity Class Filter ───────────────────────────────────────────────

def filter_thin_classes(X, y, label="dataset"):
    """
    Fix 4: Removes identity classes with fewer than MIN_SAMPLES_PER_CLASS
    samples.  Prints the full identity list with counts, marks excluded classes,
    and prints an exclusion summary.

    Returns (X_filtered, y_filtered, bool_mask) so callers can apply the same
    mask to any parallel array (e.g. X_voices_aligned).
    """
    counts   = Counter(y)
    excluded = {cls: c for cls, c in counts.items() if c < MIN_SAMPLES_PER_CLASS}
    kept     = {cls: c for cls, c in counts.items() if c >= MIN_SAMPLES_PER_CLASS}

    print(f"\n[Class Filter — {label}] All identity sample counts:")
    for cls, c in sorted(counts.items()):
        marker = "  ← EXCLUDED (< {})".format(MIN_SAMPLES_PER_CLASS) \
                 if cls in excluded else ""
        print(f"  Identity '{cls}': {c} sample(s){marker}")

    if excluded:
        print(f"\n  Excluded {len(excluded)} identit(ies) "
              f"with < {MIN_SAMPLES_PER_CLASS} samples:")
        for cls, c in sorted(excluded.items()):
            print(f"    Identity '{cls}': {c} sample(s)")
        print(f"  ▸ Kept    : {len(kept)} identities / "
              f"{sum(kept.values())} samples")
        print(f"  ▸ Removed : {len(excluded)} identities / "
              f"{sum(excluded.values())} samples")
    else:
        print(f"\n  All {len(kept)} identities meet the minimum "
              f"threshold of {MIN_SAMPLES_PER_CLASS} samples — none excluded.")

    mask = np.array([yi in kept for yi in y])
    return X[mask], y[mask], mask


# ─── Visualisation ────────────────────────────────────────────────────────────

def visualize_results(fold_metrics, classifier_name, n_splits):
    metrics  = ['accuracy', 'macro_roc_auc', 'average_eer', 'd_prime']
    titles   = ['Accuracy', 'Macro-average ROC AUC', 'Average EER', 'D-prime']
    y_labels = ['Accuracy', 'ROC AUC', 'EER', 'D-prime']

    for metric, title, y_label in zip(metrics, titles, y_labels):
        if not fold_metrics[metric]:
            continue
        plt.figure(figsize=(8, 6))
        plt.plot(fold_metrics[metric], marker='o', linestyle='-',
                 label=f'{metric} per fold')
        plt.axhline(y=np.mean(fold_metrics[metric]), color='r',
                    linestyle='--', label=f'Mean {title}')
        full_title = f"{title} – {classifier_name} ({n_splits}-fold CV)"
        plt.title(full_title)
        plt.xlabel('Fold')
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        _save(full_title)


def resolve_cv_splits(y, requested_n_splits):
    class_counts = Counter(y)
    min_samples  = min(class_counts.values())
    actual_n_splits = min(requested_n_splits, min_samples)
    insufficient_classes = [
        f"{cls} ({count} samples)"
        for cls, count in sorted(class_counts.items())
        if count < MIN_CLASS_SAMPLES_FOR_CV
    ]
    reduced_classes = [
        f"{cls} ({count} samples)"
        for cls, count in sorted(class_counts.items())
        if count < requested_n_splits
    ]
    return actual_n_splits, class_counts, insufficient_classes, reduced_classes


def compare_and_visualize_results(system_results, metric_names=None,
                                   selection_metric="accuracy",
                                   fold_count=None):
    if metric_names is None:
        metric_names = ["accuracy", "macro_roc_auc", "average_eer", "d_prime"]

    classifier_rows = []
    best_system_rows = []
    for system_name, classifier_results in system_results.items():
        for classifier_name, metrics in classifier_results.items():
            classifier_rows.append({
                "System":     system_name,
                "Classifier": classifier_name,
                "Accuracy":   metrics.get("accuracy", 0),
                "ROC AUC":    metrics.get("macro_roc_auc", 0),
                "EER":        metrics.get("average_eer", 0),
                "D-prime":    metrics.get("d_prime", 0),
            })

        best_name = max(
            classifier_results.items(),
            key=lambda item: item[1].get(selection_metric, 0),
            default=(None, None),
        )[0]
        best_metrics = classifier_results.get(best_name, {})
        best_system_rows.append({
            "System":          system_name,
            "Best Classifier": best_name,
            "Accuracy":        best_metrics.get("accuracy", 0),
            "ROC AUC":         best_metrics.get("macro_roc_auc", 0),
            "EER":             best_metrics.get("average_eer", 0),
            "D-prime":         best_metrics.get("d_prime", 0),
        })

    print("\nPer-classifier results by system:")
    print(pd.DataFrame(classifier_rows).to_string(index=False))

    print(f"\nBest classifier per system (selected by {selection_metric}):")
    print(pd.DataFrame(best_system_rows).to_string(index=False))

    summary_df = pd.DataFrame(best_system_rows)
    plot_df = summary_df.drop(columns=["System", "Best Classifier"])
    plot_df.index = summary_df["System"]

    plot_df.plot(kind="bar", figsize=(12, 6),
                 title="Best Classifier per System")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(title="Metric")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    title = "Best_Classifier_per_System"
    if fold_count is not None:
        title = f"{title}_{fold_count}_folds"
    _save(title)


# ─── Training & Evaluation ────────────────────────────────────────────────────

def train_and_evaluate_with_cv(X, y, n_splits=N_SPLITS,
                                system_tag="", raw_face_dim=None, X_faces_raw=None):
    """
    Trains RF, SVM (GridSearchCV-tuned), kNN via stratified K-Fold CV.

    raw_face_dim : int or None
        If not None, the first `raw_face_dim` columns of X are face embeddings.
        If `X_faces_raw` is provided, those base pixel arrays are used to generate
        augmented embeddings for the TRAINING FOLD ONLY inside this loop. Test-fold
        images use the pre-computed embeddings directly.

        If X has additional columns beyond raw_face_dim, they are treated as
        pre-computed voice features (multimodal mode).

        If raw_face_dim is None, X is a plain pre-computed feature matrix
        (voice-only mode) and no augmentation is performed.

    SMOTE is strictly applied inside the CV loop on the (post-augmentation)
    training fold only.

    For multimodal data, scaling and PCA (if enabled) are applied independently to
    face and voice sub-vectors.

    Fix 5: SVM is tuned with GridSearchCV(kernel, C, gamma) inside each
    training fold — the grid search never sees test-fold data.
    """
    if len(X) == 0:
        print("  No data - skipping.")
        return {"accuracy": 0, "macro_roc_auc": 0,
                "average_eer": 0, "d_prime": 0}

    actual_n_splits, class_counts, insufficient_classes, reduced_classes = \
        resolve_cv_splits(y, n_splits)

    if len(class_counts) < 2:
        print("  Need >= 2 classes - skipping.")
        return {"accuracy": 0, "macro_roc_auc": 0,
                "average_eer": 0, "d_prime": 0}

    if insufficient_classes:
        print(
            f"  WARNING: Identity class(es) with fewer than "
            f"{MIN_CLASS_SAMPLES_FOR_CV} samples detected: "
            f"{', '.join(insufficient_classes)}. These may make per-class "
            f"precision/recall estimates unstable in CV."
        )

    if actual_n_splits < n_splits:
        print(
            f"  WARNING: Requested n_splits={n_splits} but using "
            f"{actual_n_splits} folds because class(es) "
            f"{', '.join(reduced_classes)} have too few samples to support "
            f"{n_splits} folds."
        )

    if actual_n_splits < 2:
        print(f"  Too few samples/class "
              f"({min(class_counts.values())}) - skipping.")
        return {"accuracy": 0, "macro_roc_auc": 0,
                "average_eer": 0, "d_prime": 0}

    # Determine pipeline mode
    has_face  = (raw_face_dim is not None)
    has_voice = has_face and (X.shape[1] > raw_face_dim)
    # Embeddings don't double in size after augmentation
    aug_face_dim = raw_face_dim if has_face else 0

    # Fix 5: SVM hyperparameter grid — searched INSIDE each training fold
    svm_param_grid = {
        "kernel": ["linear", "rbf"],
        "C":      [0.1, 1, 10],
        "gamma":  ["scale", "auto"],
    }

    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, random_state=42, n_jobs=N_JOBS),
        "SVM": GridSearchCV(
            SVC(probability=True, random_state=42, max_iter=2000),
            param_grid=svm_param_grid,
            cv=3, n_jobs=N_JOBS, refit=True, verbose=0),
        "kNN": KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS),
    }

    skf     = StratifiedKFold(n_splits=actual_n_splits, shuffle=True, random_state=42)
    classes = np.unique(y)
    classifier_results = {}

    for clf_name, clf in classifiers.items():
        print(f"\n  [{system_tag}] Evaluating {clf_name} "
              f"({actual_n_splits}-fold CV)...")
        fold_metrics = {"accuracy": [], "macro_roc_auc": [],
                        "average_eer": [], "d_prime": []}

        for fold_num, (tr_idx, te_idx) in enumerate(
                skf.split(X, y), start=1):

            # ── Build training & test feature matrices ─────────────────────────
            if has_face:
                # Fix 1: augment TRAINING fold only — test fold untouched ──────
                aug_face_list, aug_label_list = [], []
                aug_voice_list = [] if has_voice else None

                for i in tr_idx:
                    if X_faces_raw is not None:
                        px_arr = X_faces_raw[i]
                    else:
                        px_arr = X[i, :raw_face_dim]
                    aug_feats = augment_and_extract_face_features(px_arr)
                    for feat in aug_feats:
                        aug_face_list.append(feat)
                        aug_label_list.append(y[i])
                        if has_voice:
                            aug_voice_list.append(X[i, raw_face_dim:])

                X_tr_faces = np.array(aug_face_list, dtype=np.float32)
                y_tr       = np.array(aug_label_list)

                X_tr = (np.hstack((X_tr_faces,
                                   np.array(aug_voice_list, dtype=np.float32)))
                        if has_voice else X_tr_faces)

                # Fix 1: test fold — pre-computed embeddings, zero augmentation ──
                if X_faces_raw is not None:
                    # X already contains embeddings
                    X_te_faces = X[te_idx, :raw_face_dim].astype(np.float32)
                else:
                    X_te_faces = np.array(
                        [extract_face_features(X[i, :raw_face_dim]) for i in te_idx],
                        dtype=np.float32)

                X_te = (np.hstack((X_te_faces,
                                   X[te_idx, raw_face_dim:].astype(np.float32)))
                        if has_voice else X_te_faces)

                y_te = y[te_idx]

                if fold_num == 1:
                    print(f"    [Fold 1] Training: {X_tr.shape[0]} augmented "
                          f"samples (from {len(tr_idx)} base), "
                          f"{X_tr.shape[1]} features.")
                    print(f"    [Fold 1] Test:     {X_te.shape[0]} original "
                          f"(non-augmented) samples.")

            else:
                # Voice-only: no augmentation
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

            # ── SMOTE (strictly on training data, after augmentation) ──────────
            train_class_counts = Counter(y_tr)
            smote_k = (min(5, min(train_class_counts.values()) - 1)
                       if min(train_class_counts.values()) > 1 else 1)
            smote = SMOTE(random_state=42, k_neighbors=smote_k)
            try:
                before = len(y_tr)
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
                if fold_num == 1:
                    print(f"    [Fold 1] SMOTE generated "
                          f"{len(y_tr) - before} synthetic training samples.")
            except ValueError:
                pass

            # ── Scale & PCA ────────────────────────────────────────────────────
            if has_face and has_voice:
                # Multimodal: scale face and voice sub-vectors independently
                X_tr_fp = X_tr[:, :aug_face_dim]
                X_tr_vp = X_tr[:, aug_face_dim:]
                X_te_fp = X_te[:, :aug_face_dim]
                X_te_vp = X_te[:, aug_face_dim:]

                scaler_f = StandardScaler()
                X_tr_fp  = scaler_f.fit_transform(X_tr_fp)
                X_te_fp  = scaler_f.transform(X_te_fp)

                scaler_v = StandardScaler()
                X_tr_vp  = scaler_v.fit_transform(X_tr_vp)
                X_te_vp  = scaler_v.transform(X_te_vp)

                # PCA on face features only
                if FACE_EMBED_PCA:
                    n_comp = min(PCA_COMPONENTS,
                                 X_tr_fp.shape[0] - 1, X_tr_fp.shape[1])
                    pca    = PCA(n_components=n_comp, random_state=42)
                    X_tr_fp = pca.fit_transform(X_tr_fp)
                    X_te_fp = pca.transform(X_te_fp)

                X_tr = np.hstack((X_tr_fp, X_tr_vp))
                X_te = np.hstack((X_te_fp, X_te_vp))

                if fold_num == 1:
                    print(f"    [Fold 1] Multimodal features after fusion: "
                          f"{X_tr.shape[1]} dims")

            elif has_face:
                # Face-Only: scale + PCA
                scaler = StandardScaler()
                X_tr   = scaler.fit_transform(X_tr)
                X_te   = scaler.transform(X_te)

                if FACE_EMBED_PCA:
                    n_comp = min(PCA_COMPONENTS,
                                 X_tr.shape[0] - 1, X_tr.shape[1])
                    pca    = PCA(n_components=n_comp, random_state=42)
                    X_tr   = pca.fit_transform(X_tr)
                    X_te   = pca.transform(X_te)

            else:
                # Voice-Only: scale only
                scaler = StandardScaler()
                X_tr   = scaler.fit_transform(X_tr)
                X_te   = scaler.transform(X_te)

            # ── Train & predict ────────────────────────────────────────────────
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            fit_time = time.time() - t0

            # Fix 5: print best SVM params found by GridSearchCV this fold
            if clf_name == "SVM" and hasattr(clf, 'best_params_'):
                print(f"    [Fold {fold_num}] SVM best params: "
                      f"{clf.best_params_}  (grid-search fit: {fit_time:.1f}s)")
            elif fold_num == 1:
                print(f"    [Fold 1] {clf_name} fit time: {fit_time:.1f}s")

            y_pred = clf.predict(X_te)

            print(f"\n  Classification Report – {clf_name} Fold {fold_num}:")
            print(classification_report(y_te, y_pred, zero_division=0))

            tag = f"{system_tag}_{clf_name}_fold{fold_num}"

            # Confusion matrix
            cm = confusion_matrix(y_te, y_pred, labels=classes)
            plot_confusion_matrix(cm, classes=classes, tag=tag,
                                  fold_count=actual_n_splits)

            # ROC / EER / d-prime
            if hasattr(clf, "predict_proba"):
                if len(np.unique(y_te)) < 2:
                    fold_metrics["accuracy"].append(np.mean(y_te == y_pred))
                    continue

                y_score   = clf.predict_proba(X_te)
                macro_auc = plot_multiclass_roc_curve(
                    clf, X_te, y_te, classes=classes, tag=tag,
                    fold_count=actual_n_splits)

                y_te_bin  = label_binarize(y_te, classes=classes)
                n_cls     = y_te_bin.shape[1]
                eer_vals, genuine_sc, impostor_sc = [], [], []

                for i in range(n_cls):
                    fp, tp, thr = roc_curve(y_te_bin[:, i], y_score[:, i])
                    eer, _      = calculate_eer(fp, tp, thr)
                    eer_vals.append(eer)
                    genuine_sc.extend(
                        y_score[y_te_bin[:, i] == 1, i].tolist())
                    impostor_sc.extend(
                        y_score[y_te_bin[:, i] == 0, i].tolist())

                if genuine_sc and impostor_sc:
                    plot_score_distribution(genuine_sc, impostor_sc, tag=tag,
                                            fold_count=actual_n_splits)

                fold_metrics["accuracy"].append(np.mean(y_te == y_pred))
                fold_metrics["macro_roc_auc"].append(macro_auc)
                fold_metrics["average_eer"].append(np.mean(eer_vals))
                fold_metrics["d_prime"].append(
                    calculate_d_prime(np.array(genuine_sc),
                                      np.array(impostor_sc)))

        # ── Per-classifier report ──────────────────────────────────────────────
        if fold_metrics["accuracy"]:
            print(f"\n  {clf_name} [{system_tag}] CV Results "
                  f"({actual_n_splits}-fold):")
            print(f"    Mean Accuracy:      "
                  f"{np.mean(fold_metrics['accuracy']):.4f}")
            print(f"    Mean Macro ROC AUC: "
                  f"{np.mean(fold_metrics['macro_roc_auc']):.4f}")
            print(f"    Mean Average EER:   "
                  f"{np.mean(fold_metrics['average_eer']):.4f}")
            print(f"    Mean D-prime:       "
                  f"{np.mean(fold_metrics['d_prime']):.4f}")
            classifier_results[clf_name] = {k: np.mean(v)
                                            for k, v in fold_metrics.items()}
            visualize_results(fold_metrics, f"{clf_name} [{system_tag}]",
                              actual_n_splits)
        else:
            classifier_results[clf_name] = {"accuracy": 0, "macro_roc_auc": 0,
                                            "average_eer": 0, "d_prime": 0}

    return classifier_results


# ─── Main Execution ───────────────────────────────────────────────────────────
_pipeline_start = time.time()

# Step 1 – Load raw data
print("\n[1/4] Loading face images + extracting embeddings (VGGFace2)...")
t0 = time.time()
X_faces, y_faces = create_face_labels(FACE_DATA_DIR)

# ── Embed all base images once (cached in-memory; no fold leakage) ──────────
_face_model, _face_device = load_face_model()
print(f"  Extracting embeddings for {len(X_faces)} face images...")
X_faces_emb = embed_faces(X_faces, _face_model, _face_device)
print(f"  Embedding shape: {X_faces_emb.shape}  "
      f"(embedding time: {time.time() - t0:.1f}s)")

print("\n[1/4] Loading voice features (amplitude norm + pre-emphasis applied)...")
t0 = time.time()
X_voices, y_voices = process_voice(VOICE_DATA_DIR)
print(f"  Voice loading time: {time.time() - t0:.1f}s")

# Step 2 – Align BASE (non-augmented) face images with voice data
#           Fix 1: augmentation happens inside the CV loop (train fold only)
print("\n[2/4] Aligning modalities (base images — no augmentation before split)...")
(X_faces_emb_aligned, y_aligned,
 X_voices_aligned, y_voices_aligned,
 X_faces_raw_aligned) = align_face_voice_data(
    X_faces_emb, y_faces, X_voices, y_voices, X_faces_raw=X_faces)

# Step 3 – Filter thin identity classes
#           Fix 4: exclude identities with < MIN_SAMPLES_PER_CLASS samples
print("\n[3/4] Filtering thin identity classes...")
X_faces_emb_aligned, y_aligned, thin_mask = filter_thin_classes(
    X_faces_emb_aligned, y_aligned, label="aligned multimodal dataset")
X_voices_aligned = X_voices_aligned[thin_mask]
X_faces_raw_aligned = X_faces_raw_aligned[thin_mask]
print(f"  Remaining aligned pairs after filter: {len(y_aligned)}")

# Step 4 – Evaluate all three systems
print("\n[4/4] Evaluating systems...\n")

# Raw face feature dimension (now 512 for embeddings)
raw_face_dim = FACE_EMBED_DIM

face_fold_count,       _, _, _ = resolve_cv_splits(y_aligned, N_SPLITS)
voice_fold_count,      _, _, _ = resolve_cv_splits(y_aligned, N_SPLITS)
multimodal_fold_count, _, _, _ = resolve_cv_splits(y_aligned, N_SPLITS)

print(f"\nActual CV folds: face={face_fold_count}, "
      f"voice={voice_fold_count}, multimodal={multimodal_fold_count}")

# ── Face-Only ──────────────────────────────────────────────────────────────────
print("\nEvaluating Face-Only System...")
t0 = time.time()
face_metrics = train_and_evaluate_with_cv(
    X_faces_emb_aligned, y_aligned,
    system_tag="Face-Only", raw_face_dim=raw_face_dim, X_faces_raw=X_faces_raw_aligned)
print(f"  Face-Only total evaluation time: {time.time() - t0:.1f}s")

# ── Voice-Only ─────────────────────────────────────────────────────────────────
print("\nEvaluating Voice-Only System...")
t0 = time.time()
voice_metrics = train_and_evaluate_with_cv(
    X_voices_aligned, y_aligned,
    system_tag="Voice-Only")
print(f"  Voice-Only total evaluation time: {time.time() - t0:.1f}s")

# ── Multimodal (feature-level fusion) ──────────────────────────────────────────
print("\nEvaluating Multimodal System (feature-level fusion)...")
# Concatenate face embeddings + voice features; raw_face_dim tells the CV
# function where the face block ends so it can use X_faces_raw to augment face train-fold images
# and pair each augmented copy with its identity's voice vector.
X_combined = np.hstack((X_faces_emb_aligned, X_voices_aligned))
t0 = time.time()
multimodal_metrics = train_and_evaluate_with_cv(
    X_combined, y_aligned,
    system_tag="Multimodal", raw_face_dim=raw_face_dim, X_faces_raw=X_faces_raw_aligned)
print(f"  Multimodal total evaluation time: {time.time() - t0:.1f}s")

# ── Final Comparison ──────────────────────────────────────────────────────────
system_results = {
    "Face-Only": face_metrics,
    "Voice-Only": voice_metrics,
    "Multimodal": multimodal_metrics,
}

fold_count_final = max(face_fold_count, voice_fold_count, multimodal_fold_count)
compare_and_visualize_results(system_results, fold_count=fold_count_final)

print(f"\nResults Summary ({fold_count_final}-fold CV):")
summary_rows = []
for system_name, classifier_results in system_results.items():
    best_name = max(
        classifier_results.items(),
        key=lambda item: item[1].get("accuracy", 0),
        default=(None, None),
    )[0]
    best_metrics = classifier_results.get(best_name, {})
    summary_rows.append({
        "System":          system_name,
        "Accuracy":        best_metrics.get("accuracy", 0),
        "ROC AUC":         best_metrics.get("macro_roc_auc", 0),
        "EER":             best_metrics.get("average_eer", 0),
        "D-prime":         best_metrics.get("d_prime", 0),
        "Best Classifier": best_name,
    })
print(pd.DataFrame(summary_rows).to_string(index=False))

print(f"\nTotal pipeline time: {time.time() - _pipeline_start:.1f}s")
print(f"\nDone! All plots saved to: {RESULTS_DIR}")

import joblib
print("\n[5/5] Training final Multimodal SVM model for inference...")
final_svm = GridSearchCV(
    SVC(probability=True, random_state=42, max_iter=2000),
    param_grid={"kernel": ["linear", "rbf"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    cv=3, n_jobs=N_JOBS, refit=True, verbose=0)
final_svm.fit(X_combined, y_aligned)
joblib.dump(final_svm, 'final_multimodal_model.pkl')
print(f"  Best SVM params: {final_svm.best_params_}")
print("  Saved to 'final_multimodal_model.pkl'")