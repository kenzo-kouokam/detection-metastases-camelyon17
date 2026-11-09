"""Tests sur des patchs synthétiques de type H&E.

Ils vérifient la mécanique du module. Ils ne prouvent rien sur les vraies
images : l'effet réel se mesure ensuite par hôpital.
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import preprocessing as pp

H = np.array([0.65, 0.70, 0.29])
E = np.array([0.07, 0.99, 0.11])


def patch(seed, h=H, e=E, gain=1.0, taille=96):
    """Patch où hématoxyline et éosine occupent des zones aléatoires."""
    rng = np.random.default_rng(seed)
    ch = rng.gamma(2.0, 0.25, (taille, taille)) * gain
    ce = rng.gamma(2.0, 0.20, (taille, taille)) * gain
    od = ch[..., None] * h / np.linalg.norm(h) + ce[..., None] * e / np.linalg.norm(e)
    return np.clip(256 * np.exp(-od) - 1, 0, 255).astype(np.uint8)


def domaines():
    ref = [patch(i) for i in range(6)]
    decale = [patch(i + 50, h=H + [0.15, -0.1, 0.1], e=E + [0.2, -0.1, 0.05], gain=1.4) for i in range(6)]
    return ref, decale


def lames_exemple():
    lignes = []
    slide = 0
    for centre in (0, 1, 2, 3, 4):
        for k, classe in enumerate(["itc", "itc", "micro", "micro", "macro", "macro"]):
            lignes.append(dict(slide=slide, patient=centre * 100 + k, center=centre, classe=classe,
                               tum_sub=1000 if k == 5 else 10))
            slide += 1
    return pd.DataFrame(lignes)


def test_lab_aller_retour():
    img = patch(1)
    assert np.abs(pp.lab_vers_rgb(pp.rgb_vers_lab(img)).astype(int) - img.astype(int)).max() <= 1


def test_lab_blanc():
    lab = pp.rgb_vers_lab(np.full((1, 1, 3), 255, dtype=np.uint8))[0, 0]
    assert lab[0] == pytest.approx(100, abs=0.01)
    assert abs(lab[1]) < 0.01 and abs(lab[2]) < 0.01


@pytest.mark.parametrize("classe", [pp.Reinhard, pp.Macenko])
def test_normalisation_rapproche_les_domaines(classe):
    ref, decale = domaines()
    norm = classe().fit(ref, [0] * len(ref))
    cible = pp.rgb_vers_lab(np.stack(ref)).reshape(-1, 3).mean(0)
    avant = pp.rgb_vers_lab(np.stack(decale)).reshape(-1, 3).mean(0)
    apres = pp.rgb_vers_lab(np.stack([norm(i) for i in decale])).reshape(-1, 3).mean(0)
    assert np.linalg.norm(apres - cible) < 0.5 * np.linalg.norm(avant - cible)


@pytest.mark.parametrize("classe", [pp.Reinhard, pp.Macenko])
def test_format_conserve_pil_et_numpy(classe):
    ref, _ = domaines()
    norm = classe().fit(ref, [3] * len(ref))
    sortie_pil = norm(Image.fromarray(ref[0]))
    assert isinstance(sortie_pil, Image.Image) and sortie_pil.size == (96, 96)
    sortie_np = norm(ref[0])
    assert sortie_np.shape == (96, 96, 3) and sortie_np.dtype == np.uint8


@pytest.mark.parametrize("centre", [1, 2])
@pytest.mark.parametrize("classe", [pp.Reinhard, pp.Macenko])
def test_reference_interdite_hors_entrainement(classe, centre):
    ref, _ = domaines()
    with pytest.raises(ValueError, match="interdits"):
        classe().fit(ref, [centre] * len(ref))


def test_macenko_patch_blanc_inchange_et_compte():
    ref, _ = domaines()
    norm = pp.Macenko().fit(ref, [0] * len(ref))
    blanc = np.full((96, 96, 3), 255, dtype=np.uint8)
    assert np.array_equal(norm(blanc), blanc)
    assert norm.n_echecs == 1


def test_reinhard_patch_uniforme_ne_plante_pas():
    ref, _ = domaines()
    norm = pp.Reinhard().fit(ref, [0] * len(ref))
    sortie = norm(np.full((96, 96, 3), 200, dtype=np.uint8))
    assert sortie.shape == (96, 96, 3) and np.isfinite(sortie).all()


def test_jitter_hed_change_l_image_sans_la_casser():
    img = patch(2)
    sortie = pp.JitterHED(sigma=0.1, biais=0.1, seed=0)(img)
    assert sortie.shape == img.shape and sortie.dtype == np.uint8
    assert np.abs(sortie.astype(int) - img.astype(int)).mean() > 0.5


def test_jitter_hed_nul_est_quasi_identique():
    img = patch(3)
    sortie = pp.JitterHED(sigma=0.0, biais=0.0, seed=0)(img)
    assert np.abs(sortie.astype(int) - img.astype(int)).max() <= 2


def test_split_interne_garde_patientes_entieres_et_evite_la_dominante():
    lames = lames_exemple()
    split = pp.faire_split_interne(lames)
    assert {lames.set_index("slide").center[s] for s in split} == {0, 3, 4}
    val = lames[lames.slide.isin([s for s, v in split.items() if v == "val_interne"])]
    assert len(val) == 9
    assert (val.tum_sub == 10).all()


def test_split_interne_patiente_a_plusieurs_lames():
    lames = lames_exemple()
    lames.loc[lames.slide == 1, "patient"] = lames.loc[lames.slide == 2, "patient"].iloc[0]
    split = pp.faire_split_interne(lames)
    p = lames.loc[lames.slide == 1, "patient"].iloc[0]
    assert len({split[int(s)] for s in lames[lames.patient == p].slide}) == 1


def test_split_interne_reproductible():
    lames = lames_exemple()
    assert pp.faire_split_interne(lames, seed=7) == pp.faire_split_interne(lames, seed=7)


def test_controle_detecte_lame_partagee():
    df = pd.DataFrame(dict(slide=[1, 1], patient=[1, 1], center=[0, 0], split=["train", "id_val"]))
    with pytest.raises(ValueError, match="Fuite"):
        pp.controler_fuite(df, "split")


def test_controle_detecte_patiente_partagee():
    df = pd.DataFrame(dict(slide=[1, 2], patient=[1, 1], center=[0, 0], split=["train_interne", "val_interne"]))
    with pytest.raises(ValueError, match="patient"):
        pp.controler_fuite(df, "split")


def test_controle_detecte_hopital_test_a_l_entrainement():
    df = pd.DataFrame(dict(slide=[1], patient=[1], center=[2], split=["train"]))
    with pytest.raises(ValueError, match="interdits"):
        pp.controler_fuite(df, "split")


def test_controle_accepte_un_split_propre():
    df = pd.DataFrame(dict(slide=[1, 2], patient=[1, 2], center=[0, 3], split=["train_interne", "val_interne"]))
    assert pp.controler_fuite(df, "split")


def test_distances_lab_symetriques_et_nulles_en_diagonale():
    ref, decale = domaines()
    stats = pp.stats_lab(ref + decale, [0] * len(ref) + [3] * len(decale))
    d = pp.distances_lab(stats)
    assert d.loc[0, 0] == 0 and d.loc[0, 3] == d.loc[3, 0] > 1
