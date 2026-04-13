"""Player identity database — face embeddings and DB operations.

Face recognition pipeline (no insightface Python package required):
  - Face detection: uses existing YOLO pose model keypoints (nose, left_eye, right_eye)
    to locate and align the face region within each detected body.
  - Face embedding: uses InsightFace ArcFace ResNet50 ONNX model (w600k_r50.onnx)
    via onnxruntime. Produces 512-dim L2-normalized embeddings.

Model setup (one-time):
    python scripts/download_face_models.py

Face DB:
    data/players/faces.json  — list of PlayerRecord dicts

Usage:
    from backend.player_identity import FaceDB, FaceEmbedder, align_face_from_keypoints

    db = FaceDB(Path("data/players/faces.json"))
    embedder = FaceEmbedder(Path("data/models/face/w600k_r50.onnx"))

    emb = embedder.embed(aligned_face_bgr)          # np.ndarray shape [512]
    record = db.match(emb, threshold=0.4)            # PlayerRecord | None
    if record is None:
        db.enroll("Anh Thao", emb)
    db.save()
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ArcFace standard 3-landmark template (left_eye, right_eye, nose_tip) at 112×112
_ARCFACE_TEMPLATE_3PT = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366]],
    dtype=np.float32,
)

# Minimum keypoint confidence to use a YOLO pose keypoint for alignment
_MIN_KPT_CONF = 0.4

# Default cosine-similarity threshold for face matching.
# ArcFace embeddings are L2-normalised, so cosine similarity = dot product.
# > 0.35 is commonly used for "same person".
DEFAULT_MATCH_THRESHOLD: float = 0.35


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PlayerRecord:
    player_id: str
    name: str
    embedding: list[float]          # 512 floats, L2-normalised
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def embedding_array(self) -> np.ndarray:
        return np.array(self.embedding, dtype=np.float32)


# ---------------------------------------------------------------------------
# Face DB
# ---------------------------------------------------------------------------

class FaceDB:
    """Persistent local player face database (JSON-backed)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.records: list[PlayerRecord] = []
        self._dirty = False
        if db_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.db_path.read_text(encoding="utf-8"))
            self.records = [PlayerRecord(**r) for r in data.get("players", [])]
        except Exception as exc:
            raise RuntimeError(f"Failed to load face DB from {self.db_path}: {exc}") from exc

    def save(self) -> None:
        """Persist DB to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "players": [asdict(r) for r in self.records]}
        self.db_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._dirty = False

    def match(
        self,
        embedding: np.ndarray,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> Optional[PlayerRecord]:
        """Return the best-matching PlayerRecord, or None if below threshold."""
        if not self.records:
            return None
        emb = _l2norm(embedding)
        best_record: Optional[PlayerRecord] = None
        best_sim = -1.0
        for record in self.records:
            sim = float(np.dot(emb, _l2norm(record.embedding_array())))
            if sim > best_sim:
                best_sim = sim
                best_record = record
        if best_sim >= threshold:
            return best_record
        return None

    def enroll(self, name: str, embedding: np.ndarray, notes: str = "") -> PlayerRecord:
        """Add a new player to the DB and mark dirty."""
        record = PlayerRecord(
            player_id=str(uuid.uuid4()),
            name=name,
            embedding=_l2norm(embedding).tolist(),
            notes=notes,
        )
        self.records.append(record)
        self._dirty = True
        return record

    def update_embedding(self, player_id: str, new_embedding: np.ndarray) -> None:
        """Average a new embedding into an existing record (online update)."""
        for record in self.records:
            if record.player_id == player_id:
                old = record.embedding_array()
                averaged = _l2norm((old + _l2norm(new_embedding)) / 2.0)
                record.embedding = averaged.tolist()
                self._dirty = True
                return

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return f"FaceDB({self.db_path}, {len(self.records)} players)"


# ---------------------------------------------------------------------------
# Face alignment
# ---------------------------------------------------------------------------

def align_face_from_keypoints(
    frame: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    out_size: int = 112,
) -> Optional[np.ndarray]:
    """Produce an aligned face crop from YOLO pose keypoints.

    Args:
        frame: BGR image.
        kpts_xy: shape [17, 2] pixel coords (YOLO pose order).
        kpts_conf: shape [17] confidence scores.
        out_size: output face size in pixels (default 112 for ArcFace).

    YOLO pose keypoint indices used:
        0 = nose, 1 = left_eye, 2 = right_eye

    Returns:
        BGR image of shape [out_size, out_size, 3], or None if keypoints
        are not reliable enough.
    """
    # Indices: nose=0, left_eye=1, right_eye=2
    idxs = [1, 2, 0]  # left_eye, right_eye, nose (matches template order)
    confs = kpts_conf[idxs]
    if np.any(confs < _MIN_KPT_CONF):
        return None

    src = kpts_xy[idxs].astype(np.float32)  # shape [3, 2]
    dst = _ARCFACE_TEMPLATE_3PT * (out_size / 112.0)

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None

    aligned = cv2.warpAffine(frame, M, (out_size, out_size), flags=cv2.INTER_LINEAR)
    return aligned


def crop_face_from_keypoints(
    frame: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    padding: float = 0.35,
) -> Optional[np.ndarray]:
    """Fallback: simple bbox crop around face keypoints (no alignment).

    Used when estimateAffinePartial2D fails. Less accurate than aligned crop.
    Returns a 112×112 BGR crop or None.
    """
    idxs = [0, 1, 2, 3, 4]  # nose + eyes + ears
    valid = [i for i in idxs if kpts_conf[i] >= _MIN_KPT_CONF]
    if len(valid) < 2:
        return None

    pts = kpts_xy[valid]
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    half = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min()) * (0.5 + padding)
    h, w = frame.shape[:2]
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(w, int(cx + half))
    y2 = min(h, int(cy + half))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# ArcFace embedding model (onnxruntime)
# ---------------------------------------------------------------------------

class FaceEmbedder:
    """Wraps the InsightFace w600k_r50.onnx ArcFace model via onnxruntime.

    Input:  [1, 3, 112, 112] float32, pixel range [-1, 1]
            (i.e. (pixel - 127.5) / 128.0)
    Output: [1, 512] float32 embedding (L2-normalised by the model)
    """

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found: {model_path}\n"
                f"Run: python scripts/download_face_models.py"
            )
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime-gpu is required: pip install onnxruntime-gpu")

        providers = ort.get_available_providers()
        # Prefer GPU
        ep = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
        self._session = ort.InferenceSession(str(model_path), providers=[ep])
        self._input_name: str = self._session.get_inputs()[0].name

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return 512-dim L2-normalised embedding for a 112×112 BGR face crop."""
        assert face_bgr.shape == (112, 112, 3), f"Expected (112,112,3), got {face_bgr.shape}"
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        blob = rgb.transpose(2, 0, 1)[np.newaxis]          # [1, 3, 112, 112]
        out = self._session.run(None, {self._input_name: blob})[0]  # [1, 512]
        return _l2norm(out[0])

    def embed_batch(self, faces_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Embed multiple faces; returns list of 512-dim arrays."""
        return [self.embed(f) for f in faces_bgr]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return v
    return v / norm


def face_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised embeddings (range [-1, 1])."""
    return float(np.dot(_l2norm(a), _l2norm(b)))
