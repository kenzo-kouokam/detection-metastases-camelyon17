import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "aggregation"))
from slide_aggregator import charger  # noqa: E402
from verifier_predictions import verifier  # noqa: E402

meta, _ = charger(ROOT / "donnees")


def faux_csv(tmp_path, ids, proba=None):
    p = tmp_path / "p.csv"
    pd.DataFrame({"image_id": ids, "proba": proba if proba is not None else np.full(len(ids), 0.5)}).to_csv(p, index=False)
    return p


def test_fichier_complet_h1_valide(tmp_path):
    ids = meta[meta.center == 1].image_id.values
    erreurs, r = verifier(faux_csv(tmp_path, ids), meta)
    assert erreurs == [] and r["lignes"] == r["attendu"]


def test_incomplet_doublon_et_h2_refuses(tmp_path):
    ids = meta[meta.center == 1].image_id.values
    assert any("sans prédiction" in e for e in verifier(faux_csv(tmp_path, ids[:100]), meta)[0])
    assert any("doublons" in e for e in verifier(faux_csv(tmp_path, np.r_[ids, ids[:3]]), meta)[0])
    h2 = meta[meta.center == 2].image_id.values[:5]
    assert any("hôpital 2" in e for e in verifier(faux_csv(tmp_path, np.r_[ids, h2]), meta)[0])


def test_proba_hors_bornes(tmp_path):
    ids = meta[meta.center == 1].image_id.values
    assert any("hors de" in e for e in verifier(faux_csv(tmp_path, ids, np.full(len(ids), 1.5)), meta)[0])
