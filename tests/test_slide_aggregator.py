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
