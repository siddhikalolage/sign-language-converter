import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

from gesture_pipeline import EXPECTED_FEATURE_COUNT

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
HAND_ANGLE_TRIPLETS = [
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
]


def _safe_norm(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.where(norms < 1e-6, 1.0, norms)


class GestureFeatureExtractor(BaseEstimator, TransformerMixin):
    """Mix raw landmarks with simple hand-shape features for better generalization."""

    def __init__(self, include_raw: bool = True):
        self.include_raw = include_raw

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected input shape (*, {EXPECTED_FEATURE_COUNT}), got {array.shape}"
            )
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        array = np.asarray(X, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected input shape (*, {EXPECTED_FEATURE_COUNT}), got {array.shape}"
            )

        face = array[:, :18].reshape(-1, 6, 3)
        left_hand = array[:, 18:81].reshape(-1, 21, 3)
        right_hand = array[:, 81:].reshape(-1, 21, 3)

        face_center = face.mean(axis=1)
        face_width = np.linalg.norm(face[:, 1, :2] - face[:, 4, :2], axis=1, keepdims=True)
        face_width = np.where(face_width < 1e-6, 1.0, face_width)

        feature_blocks = [array] if self.include_raw else []
        feature_blocks.append(
            np.concatenate(
                [
                    self._hand_shape_features(left_hand, face_center, face_width),
                    self._hand_shape_features(right_hand, face_center, face_width),
                ],
                axis=1,
            )
        )
        return np.concatenate(feature_blocks, axis=1)

    def _hand_shape_features(
        self,
        hand: np.ndarray,
        face_center: np.ndarray,
        face_width: np.ndarray,
    ) -> np.ndarray:
        wrist = hand[:, 0:1, :]
        centroid = hand.mean(axis=1)
        tips = hand[:, FINGERTIP_INDICES, :]

        centroid_relative = (centroid - face_center) / face_width
        tip_wrist_distances = np.linalg.norm(tips - wrist, axis=2) / face_width
        joint_angles = self._joint_angle_features(hand)
        return np.concatenate([centroid_relative, tip_wrist_distances, joint_angles], axis=1)

    def _joint_angle_features(self, hand: np.ndarray) -> np.ndarray:
        features: list[np.ndarray] = []
        for start_index, pivot_index, end_index in HAND_ANGLE_TRIPLETS:
            incoming = hand[:, start_index] - hand[:, pivot_index]
            outgoing = hand[:, end_index] - hand[:, pivot_index]
            cosine = (incoming * outgoing).sum(axis=1, keepdims=True) / (
                _safe_norm(incoming) * _safe_norm(outgoing)
            )
            features.append(np.clip(cosine, -1.0, 1.0))
        return np.concatenate(features, axis=1)


def make_gesture_model_pipeline(
    *,
    random_state: int = 42,
    n_estimators: int = 350,
    n_jobs: int = 1,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", GestureFeatureExtractor(include_raw=True)),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
