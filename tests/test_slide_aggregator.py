
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import pytest
 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "aggregation"))
import slide_aggregator as sa  # noqa: E402
 
DONNEES = ROOT / "donnees"
P1_ORACLE = ROOT / "p1_data" / "sorties" / "audit_oracle.csv"
 
 
def test_composantes_8_voisins():
    # (0,0)-(1,1) sont voisines en diagonale ; (5,5) est isolée
    comps = sa.composantes([(0, 0), (1, 1), (5, 5)])
    assert sorted(len(c) for c in comps) == [1, 2]
    assert sa.composantes([]) == []
 
 
def test_seuils_classes():
    assert sa.classe_depuis_mm(0.2) == "itc"
    assert sa.classe_depuis_mm(0.21) == "micro"
    assert sa.classe_depuis_mm(2.0) == "micro"
    assert sa.classe_depuis_mm(2.01) == "macro"
 
 
def test_lame_sans_positif_est_negative():
    assert sa.classe_lame([])["classe_pred"] == "negative"
 
 
def test_taille_connue():
    # une ligne de 10 cellules -> grand axe = (9 + 1) cellules
    cel = [(i, 0) for i in range(10)]
    r = sa.classe_lame(cel)
    assert r["cellules_max"] == 10
    assert r["axe_max_mm"] == pytest.approx(10 * sa.UM_PAR_CELLULE / 1000)
 
 
@pytest.mark.skipif(not (DONNEES / "metadata_complete.zip").exists(), reason="données absentes")
def test_oracle_vraies_donnees():
    meta, stages = sa.charger(DONNEES)
    lames = sa.table_lames(meta, stages)
    cl = sa.classes_depuis_patchs(meta.assign(positif=meta.label == 1))
    o = lames.merge(cl, on="slide")
    assert len(o) == 50
    assert (o.classe == o.classe_pred).sum() == 44      # plafond de la méthode, avec l'échelle supposée
    # les classes officielles redonnent exactement le stade officiel : aucun changement
    eff = sa.effet_sur_pn(dict(zip(o.slide, o.classe)), lames, stages)
    assert (eff.ecart == 0).all() and len(eff) == 43
    if P1_ORACLE.exists():                              # même méthode que l'oracle de P1
        p1 = pd.read_csv(P1_ORACLE).set_index("slide")
        assert (o.set_index("slide").cellules_max == p1.cellules_max).all()
 
 
@pytest.mark.skipif(not (DONNEES / "metadata_complete.zip").exists(), reason="données absentes")
def test_simulation_sans_erreur_egale_oracle():
    meta, stages = sa.charger(DONNEES)
    pos = sa.simuler_positifs(meta, 0.0, 0.0, 0)
    assert (pos == (meta.label.values == 1)).all()
 
 
def test_appliquer_seuil():
    d = pd.DataFrame({"proba": [0.2, 0.5, 0.8]})
    assert sa.appliquer_seuil(d, 0.5).positif.tolist() == [False, True, True]   # >= seuil
 
 
def test_appliquer_seuil_refuse_les_entrees_invalides():
    d = pd.DataFrame({"proba": [0.2, 0.8]})
    with pytest.raises(ValueError):
        sa.appliquer_seuil(d, 1.5)
    with pytest.raises(ValueError):
        sa.appliquer_seuil(pd.DataFrame({"proba": [0.2, float("nan")]}), 0.5)
    with pytest.raises(ValueError):
        sa.appliquer_seuil(pd.DataFrame({"proba": [-2.0, 3.0]}), 0.5)           # logits, pas probabilités
    with pytest.raises(ValueError):
        sa.appliquer_seuil(pd.DataFrame({"autre": [0.1]}), 0.5)
 
 
def test_charger_predictions_refuse_doublons_et_ids_inconnus(tmp_path):
    meta = pd.DataFrame({"image_id": [1, 2, 3], "slide": [0, 0, 0], "x_coord": [0, 32, 64], "y_coord": [0, 0, 0]})
    (tmp_path / "dup.csv").write_text("image_id,proba\n1,0.5\n1,0.6\n")
    (tmp_path / "inconnu.csv").write_text("image_id,proba\n1,0.5\n99,0.6\n")
    (tmp_path / "ok.csv").write_text("image_id,proba\n1,0.9\n2,0.8\n")
    with pytest.raises(ValueError):
        sa.charger_predictions(tmp_path / "dup.csv", meta)
    with pytest.raises(ValueError):
        sa.charger_predictions(tmp_path / "inconnu.csv", meta)
    d = sa.charger_predictions(tmp_path / "ok.csv", meta)
    r = sa.classes_depuis_probas(d, 0.5, meta=meta)
    assert r.n_predits.iloc[0] == 2 and r.part_predite.iloc[0] == pytest.approx(0.667, abs=1e-3)
    assert r.cellules_max.iloc[0] == 2
 
 
@pytest.mark.skipif(not (DONNEES / "metadata_complete.zip").exists(), reason="données absentes")
def test_probas_parfaites_redonnent_l_oracle():
    meta, stages = sa.charger(DONNEES)
    lames = sa.table_lames(meta, stages)
    d = meta.assign(proba=meta.label.astype(float))
    cl = sa.classes_depuis_probas(d, 0.5)
    o = lames.merge(cl, on="slide")
    assert (o.classe == o.classe_pred).sum() == 44