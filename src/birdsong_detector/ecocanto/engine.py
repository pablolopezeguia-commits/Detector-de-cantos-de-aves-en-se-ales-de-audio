"""YAMNet + MLP inference engine for EcoCanto."""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from .audio_preprocessing import (
    count_raw_windows,
    iter_raw_windows,
    prepare_window_for_yamnet,
    read_mono_audio,
)
from .config import MODEL_DIR, SILENCE_RMS_THRESHOLD, SUPPORTED_EXTENSIONS


WINDOW_COLUMNS = [
    "archivo",
    "t_inicio",
    "t_fin",
    "p_canto",
    "prediccion",
    "source_sample_rate",
    "preprocess_gain",
    "preprocess_snr_estimated",
    "preprocess_input_rms",
    "preprocess_input_peak_abs",
    "preprocess_output_peak_abs",
    "preprocess_noise_floor_rms",
]


def _resolve_threshold(threshold: float | None, default_threshold: float) -> float:
    thr = default_threshold if threshold is None else float(threshold)
    if not np.isfinite(thr) or not 0.0 <= thr <= 1.0:
        raise ValueError(f"Threshold outside [0, 1]: {thr}")
    return thr


def verify_model(model_dir: pathlib.Path = MODEL_DIR) -> None:
    """Validate the local inference assets before the app starts."""
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json was not found in {model_dir}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    required = {"model_id", "umbral_operativo", "ventana_s", "solape", "sample_rate", "embedding_dim"}
    missing = sorted(required - set(metadata))
    if missing:
        raise RuntimeError(f"metadata.json is incomplete. Missing keys: {missing}")

    threshold = float(metadata["umbral_operativo"])
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError(f"Operating threshold outside [0, 1]: {threshold}")
    if float(metadata["ventana_s"]) != 3.0:
        raise RuntimeError(f"Unexpected window length in metadata: {metadata['ventana_s']}")
    if float(metadata["solape"]) != 0.5:
        raise RuntimeError(f"Unexpected overlap in metadata: {metadata['solape']}")
    if int(metadata["sample_rate"]) != 16000:
        raise RuntimeError(f"Unexpected model sample rate: {metadata['sample_rate']}")
    if int(metadata["embedding_dim"]) != 1024:
        raise RuntimeError(f"Unexpected embedding dimension: {metadata['embedding_dim']}")

    yamnet_path = model_dir / "yamnet_base"
    if not (yamnet_path / "saved_model.pb").exists():
        raise FileNotFoundError(
            f"Local YAMNet was not found in {yamnet_path}. "
            "The application must be distributed with the complete model/yamnet_base folder."
        )

    mlp_path = model_dir / "yamnet_final_mlp.keras"
    if not mlp_path.exists():
        raise FileNotFoundError(f"MLP model was not found: {mlp_path}")

    expected_sha = str(metadata.get("sha256_mlp", "")).strip().lower()
    if expected_sha and expected_sha != "rellenar_tras_copiar":
        actual_sha = hashlib.sha256(mlp_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(
                "The MLP model hash does not match.\n"
                f"Expected: {expected_sha}\n"
                f"Actual: {actual_sha}\n"
                "The file may be corrupt or may not be the final MLP model."
            )


class BirdSongEngine:
    """Inference engine: YAMNet extracts embeddings and the MLP classifies them."""

    def __init__(self, model_dir: pathlib.Path | None = None):
        self._model_dir = model_dir or MODEL_DIR
        verify_model(self._model_dir)
        self._load_metadata()
        self._load_yamnet()
        self._load_mlp()

    def _load_metadata(self) -> None:
        meta_path = self._model_dir / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        self.threshold = float(metadata["umbral_operativo"])
        self.window_s = float(metadata["ventana_s"])
        self.overlap = float(metadata["solape"])
        self.sample_rate = int(metadata["sample_rate"])
        self.model_id = str(metadata["model_id"])
        self.metadata = metadata

    def _load_yamnet(self) -> None:
        yamnet_path = self._model_dir / "yamnet_base"
        if not (yamnet_path / "saved_model.pb").exists():
            raise FileNotFoundError(f"Local YAMNet folder is incomplete: {yamnet_path}")
        self._yamnet = tf.saved_model.load(str(yamnet_path))

    def _load_mlp(self) -> None:
        mlp_path = self._model_dir / "yamnet_final_mlp.keras"
        self._mlp = tf.keras.models.load_model(str(mlp_path), compile=False)

    def _load_audio(self, path: str | pathlib.Path) -> tuple[np.ndarray, int]:
        try:
            audio, sr, _channels = read_mono_audio(path)
        except Exception as exc:
            raise OSError(f"Audio file cannot be read: {path}\n{exc}") from exc
        if len(audio) == 0:
            raise ValueError(f"Empty audio file: {path}")
        return audio.astype(np.float32, copy=False), int(sr)

    def preprocess_window_for_model(
        self,
        audio_chunk: np.ndarray,
        source_sr: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Apply the final preprocessing contract and adapt the window to YAMNet."""
        return prepare_window_for_yamnet(
            audio_chunk,
            source_sr=source_sr,
            target_sr=self.sample_rate,
            window_s=self.window_s,
        )

    def _iter_windows(self, audio: np.ndarray, source_sr: int):
        """Yield raw windows and preprocess them exactly as in final inference."""
        for t_ini, t_fin, raw_chunk in iter_raw_windows(audio, source_sr, self.window_s, self.overlap):
            processed, metrics = self.preprocess_window_for_model(raw_chunk, source_sr)
            yield t_ini, t_fin, processed, metrics

    def _call_yamnet(self, waveform: tf.Tensor):
        try:
            return self._yamnet(waveform)
        except TypeError:
            serving = self._yamnet.signatures.get("serving_default")
            if serving is None:
                raise RuntimeError("YAMNet is not callable and does not expose serving_default.")
            input_name = next(iter(serving.structured_input_signature[1]))
            return serving(**{input_name: waveform})

    @staticmethod
    def _extract_embedding_output(outputs) -> np.ndarray:
        if isinstance(outputs, dict):
            for key in ("output_1", "embeddings", "embedding"):
                if key in outputs:
                    emb = outputs[key]
                    break
            else:
                raise RuntimeError(f"Embeddings were not found: {list(outputs.keys())}")
        elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            emb = outputs[1]
        else:
            raise RuntimeError(f"Unexpected YAMNet output: {type(outputs)}")

        emb_np = np.asarray(emb.numpy() if hasattr(emb, "numpy") else emb, dtype=np.float32)
        if emb_np.ndim != 2 or emb_np.shape[1] != 1024:
            raise RuntimeError(f"Unexpected YAMNet embedding shape: {emb_np.shape}")
        return emb_np

    def _extract_embedding(self, audio_chunk: np.ndarray) -> np.ndarray:
        waveform = tf.convert_to_tensor(audio_chunk, dtype=tf.float32)
        outputs = self._call_yamnet(waveform)
        embeddings = self._extract_embedding_output(outputs)
        return embeddings.mean(axis=0)

    def _predict_window(self, audio_chunk: np.ndarray, metrics: dict[str, float]) -> float:
        input_rms = float(metrics.get("input_rms", 0.0))
        input_peak = float(metrics.get("input_peak_abs", 0.0))
        output_rms = float(np.sqrt(np.mean(np.square(audio_chunk, dtype=np.float32))))
        if input_rms <= SILENCE_RMS_THRESHOLD or input_peak <= SILENCE_RMS_THRESHOLD:
            return 0.0
        if output_rms <= SILENCE_RMS_THRESHOLD:
            return 0.0
        embedding = self._extract_embedding(audio_chunk)
        prob = self._mlp.predict(embedding.reshape(1, -1), verbose=0).reshape(-1)[0]
        return float(np.clip(prob, 0.0, 1.0))

    def analyze_file(
        self,
        path: str | pathlib.Path,
        threshold: float | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> pd.DataFrame:
        thr = _resolve_threshold(threshold, self.threshold)
        audio, source_sr = self._load_audio(path)
        total_windows = count_raw_windows(len(audio), source_sr, self.window_s, self.overlap)
        if total_windows == 0:
            return pd.DataFrame(columns=WINDOW_COLUMNS)
        rows = []
        for index, (t_ini, t_fin, chunk, metrics) in enumerate(self._iter_windows(audio, source_sr), start=1):
            if cancel_callback and cancel_callback():
                break
            p_canto = self._predict_window(chunk, metrics)
            rows.append(
                {
                    "archivo": str(path),
                    "t_inicio": round(t_ini, 3),
                    "t_fin": round(t_fin, 3),
                    "p_canto": round(p_canto, 4),
                    "prediccion": int(p_canto >= thr),
                    "source_sample_rate": int(source_sr),
                    "preprocess_gain": round(float(metrics["gain"]), 6),
                    "preprocess_snr_estimated": round(float(metrics["snr_estimated"]), 6),
                    "preprocess_input_rms": round(float(metrics["input_rms"]), 8),
                    "preprocess_input_peak_abs": round(float(metrics["input_peak_abs"]), 6),
                    "preprocess_output_peak_abs": round(float(metrics["output_peak_abs"]), 6),
                    "preprocess_noise_floor_rms": round(float(metrics["noise_floor_rms"]), 8),
                }
            )
            if progress_callback:
                progress_callback(index, total_windows)
        return pd.DataFrame(rows, columns=WINDOW_COLUMNS)

    def analyze_folder(
        self,
        folder: str | pathlib.Path,
        recursive: bool = True,
        threshold: float | None = None,
        file_callback: Callable[[str, int, int], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        folder_path = pathlib.Path(folder)
        pattern = "**/*" if recursive else "*"
        audio_files = sorted(
            p for p in folder_path.glob(pattern) if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not audio_files:
            raise FileNotFoundError(f"No supported audio files were found in: {folder}")

        all_dfs: list[pd.DataFrame] = []
        errors: list[dict[str, str]] = []
        for idx, audio_path in enumerate(audio_files):
            if cancel_callback and cancel_callback():
                break
            if file_callback:
                file_callback(str(audio_path), idx + 1, len(audio_files))
            try:
                df = self.analyze_file(
                    audio_path,
                    threshold=threshold,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                )
                if not df.empty:
                    all_dfs.append(df)
            except Exception as exc:
                errors.append({"archivo": str(audio_path), "error": str(exc)})

        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(columns=WINDOW_COLUMNS)
        return combined, errors
