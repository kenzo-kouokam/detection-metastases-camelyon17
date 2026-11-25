import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "aggregation"))
import cartes  # noqa: E402

DONNEES = ROOT / "donnees"


def test_grille_lame_valeurs_et_forme():
    g = pd.DataFrame({"x_coord": [64, 96, 64], "y_coord": [32, 32, 64], "label": [0, 1, 1]})
    grille = cartes.grille_lame(g)
    assert grille.shape == (2, 2)                    # y : 32..64 -> 2 lignes ; x : 64..96 -> 2 colonnes
    assert grille.tolist() == [[1, 2], [2, 0]]       # normal, tumoral / tumoral, vide


@pytest.mark.skipif(not (DONNEES / "metadata_complete.zip").exists(), reason="données absentes")
def test_cartes_ecrites(tmp_path):
    cartes.main(DONNEES, tmp_path)
    assert len(list((tmp_path / "cartes_verite").glob("lame_*.png"))) == 50
    assert (tmp_path / "cartes_verite_montage.png").stat().st_size > 10_000
