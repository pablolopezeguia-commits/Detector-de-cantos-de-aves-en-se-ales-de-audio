import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "birdsong_detector" / "ecocanto"


def test_model_assets_exist_and_hash_matches():
    metadata = json.loads((ROOT / "model" / "metadata.json").read_text(encoding="utf-8"))
    mlp = ROOT / "model" / "yamnet_final_mlp.keras"
    yamnet = ROOT / "model" / "yamnet_base" / "saved_model.pb"

    assert mlp.exists()
    assert yamnet.exists()
    assert metadata["model_id"] == "yamnet_mlp_final_augmented"
    assert metadata["umbral_operativo"] == 0.689144
    assert metadata["ventana_s"] == 3.0
    assert metadata["solape"] == 0.5
    assert metadata["sample_rate"] == 16000
    assert metadata["embedding_dim"] == 1024
    assert hashlib.sha256(mlp.read_bytes()).hexdigest() == metadata["sha256_mlp"]
