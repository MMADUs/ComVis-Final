import pickle
import io

import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops

IMAGE_WIDTH = 260
IMAGE_HEIGHT = 260

CLASS_NAMES = ["normal", "glaucoma", "cataract", "diabetic_retinopathy"]


with open("best_logreg.pkl", "rb") as f:
    logreg = pickle.load(f)

with open("best_rf.pkl", "rb") as f:
    random_forest = pickle.load(f)

with open("best_svm.pkl", "rb") as f:
    svm = pickle.load(f)

_MODELS = {
    "lr": logreg,
    "rf": random_forest,
    "svm": svm,
}


def _load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes → RGB ndarray (260×260)."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        dsize=(IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )
    return image


def apply_clahe(
    image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """Apply CLAHE per channel on an RGB image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = cv2.split(image_rgb)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)


def extract_intensity_stats(image_rgb: np.ndarray) -> np.ndarray:
    """27 features: 9 per-channel stats × 3 channels (R, G, B)."""
    features = []
    for ch in range(3):
        pixels = image_rgb[:, :, ch].astype(np.float64).ravel()
        features.extend(
            [
                np.mean(pixels),
                np.std(pixels),
                np.var(pixels),
                skew(pixels),
                kurtosis(pixels),
                np.percentile(pixels, 25),
                np.percentile(pixels, 50),
                np.percentile(pixels, 75),
                np.percentile(pixels, 90),
            ]
        )
    return np.array(features, dtype=np.float32)


def extract_glcm_features(
    image_rgb: np.ndarray, distances: tuple = (1, 3), angles: tuple = (0, 45, 90, 135)
) -> np.ndarray:
    """144 features: 6 props × 2 distances × 4 angles × 3 channels."""
    angles_rad = [np.deg2rad(a) for a in angles]
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    features = []
    for ch in range(3):
        glcm = graycomatrix(
            image_rgb[:, :, ch],
            distances=list(distances),
            angles=angles_rad,
            levels=256,
            symmetric=True,
            normed=True,
        )
        for prop in props:
            features.extend(graycoprops(glcm, prop).flatten())

    return np.array(features, dtype=np.float32)


def _extract_features(image_bytes: bytes) -> np.ndarray:
    """Full pipeline: decode → resize → CLAHE → intensity + GLCM → concat."""
    image = _load_image_from_bytes(image_bytes)
    image = apply_clahe(image)
    intensity = extract_intensity_stats(image)  # 27-dim
    glcm = extract_glcm_features(image)  # 144-dim
    features = np.concatenate([intensity, glcm])  # 171-dim
    return features.reshape(1, -1)


def predict_retinal(image_bytes: bytes, model: str = "lr") -> str:
    """
    Return the predicted class label (str) for the given image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image (JPG / PNG).
    model : str
        One of 'lr', 'svm', 'rf'.

    Returns
    -------
    str  e.g. 'glaucoma'
    """
    clf = _MODELS[model]
    features = _extract_features(image_bytes)
    pred = clf.predict(features)[0]

    # Handle numeric labels (0-3) or string labels gracefully
    try:
        return CLASS_NAMES[int(pred)]
    except (ValueError, IndexError):
        return str(pred).lower().replace(" ", "_")


def predict_retinal_proba(image_bytes: bytes, model: str = "lr") -> dict:
    """
    Return a dict mapping class name → probability.

    For SVM without probability calibration, falls back to a one-hot dict
    based on the hard prediction.

    Parameters
    ----------
    image_bytes : bytes
    model : str  — 'lr', 'svm', or 'rf'

    Returns
    -------
    dict  e.g. {'normal': 0.05, 'glaucoma': 0.80, 'cataract': 0.10,
                'diabetic_retinopathy': 0.05}
    """
    clf = _MODELS[model]
    features = _extract_features(image_bytes)

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(features)[0]
        # Map model classes to canonical names
        classes = []
        for c in clf.classes_:
            try:
                classes.append(CLASS_NAMES[int(c)])
            except (ValueError, IndexError):
                classes.append(str(c).lower().replace(" ", "_"))
        return dict(zip(classes, proba.tolist()))

    # Fallback for SVM without probability estimation
    pred = predict_retinal(image_bytes, model=model)
    return {cls: (1.0 if cls == pred else 0.0) for cls in CLASS_NAMES}
