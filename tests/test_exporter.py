import pandas as pd

from birdsong_detector.ecocanto.exporter import export_csv


def test_export_empty_writes_header(tmp_path):
    output = tmp_path / "empty.csv"
    export_csv(pd.DataFrame(), output)
    text = output.read_text(encoding="utf-8-sig")
    assert text.startswith("archivo;t_inicio;t_fin")


def test_export_uses_file_name(tmp_path):
    output = tmp_path / "segments.csv"
    df = pd.DataFrame(
        [
            {
                "archivo": r"C:\campo\audio.wav",
                "t_inicio": 1.0,
                "t_fin": 4.0,
                "duracion_s": 3.0,
                "n_ventanas_canto": 2,
                "n_ventanas_gap": 0,
                "score_medio": 0.91,
            }
        ]
    )
    export_csv(df, output)
    text = output.read_text(encoding="utf-8-sig")
    assert "audio.wav" in text
