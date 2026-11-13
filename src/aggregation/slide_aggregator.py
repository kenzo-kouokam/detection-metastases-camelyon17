import sys
import zipfile
from pathlib import Path
 
import numpy as np
import pandas as pd
from scipy import ndimage
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pn_rules import stade_pn  # noqa: E402
 
PAS_GRILLE_PX = 32
UM_PAR_PX_NIVEAU0 = 0.24
UM_PAR_CELLULE = PAS_GRILLE_PX * 4 * UM_PAR_PX_NIVEAU0
SEUIL_ITC_MM = 0.2
SEUIL_MICRO_MM = 2.0
ORDRE_PN = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]
 
 
def charger(dossier):
    dossier = Path(dossier)
    with zipfile.ZipFile(dossier / "metadata_complete.zip") as z:
        meta = pd.read_csv(z.open("metadata_complete.csv"))
    stages = pd.read_csv(dossier / "stages.csv")
    meta["nom"] = [f"patient_{p:03d}_node_{n}.tif" for p, n in zip(meta.patient, meta.node)]
    return meta, stages
 
 
def table_lames(meta, stages):
    lames = meta.groupby("slide").agg(patient=("patient", "first"), node=("node", "first"),
                                      center=("center", "first"), nom=("nom", "first")).reset_index()
    classes = stages.rename(columns={"patient": "nom", "stage": "classe"})[["nom", "classe"]]
    return lames.merge(classes, on="nom")
 
 
def composantes(cellules):
    """Composantes connexes (8 voisins). cellules : liste de (i, j). Retourne une liste de tableaux (n, 2)."""
    if len(cellules) == 0:
        return []
    pts = np.asarray(cellules, dtype=np.int64)
    mini = pts.min(axis=0)
    p = pts - mini
    grille = np.zeros(p.max(axis=0) + 1, dtype=bool)
    grille[p[:, 0], p[:, 1]] = True
    etiq, n = ndimage.label(grille, structure=np.ones((3, 3)))
    lab = etiq[p[:, 0], p[:, 1]]
    ordre = np.argsort(lab, kind="stable")
    coupures = np.flatnonzero(np.diff(lab[ordre])) + 1
    return [pts[idx] for idx in np.split(ordre, coupures)]
 
 
def grand_axe_mm(comp, um_par_cellule=UM_PAR_CELLULE):
    """Plus grande distance entre deux cellules (calculée sur les extrêmes selon 8 directions)."""
    pts = np.asarray(comp, dtype=float)
    directions = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [2, 1], [1, 2], [2, -1], [1, -2]])
    extremes = {tuple(pts[(pts @ d).argmin()]) for d in directions} | {tuple(pts[(pts @ d).argmax()]) for d in directions}
    e = np.array(list(extremes))
    dist = np.sqrt(((e[:, None] - e[None, :]) ** 2).sum(-1)).max()
    return (dist + 1) * um_par_cellule / 1000
 
 
def classe_depuis_mm(mm):
    if mm <= SEUIL_ITC_MM:
        return "itc"
    return "micro" if mm <= SEUIL_MICRO_MM else "macro"
 
 
def classe_lame(cellules, um_par_cellule=UM_PAR_CELLULE):
    """Classe d'une lame à partir de ses cellules positives. Aucune cellule -> 'negative'."""
    comps = composantes(cellules)
    if not comps:
        return {"classe_pred": "negative", "n_amas": 0, "cellules_max": 0, "axe_max_mm": 0.0}
    axe = max(grand_axe_mm(c, um_par_cellule) for c in comps)
    return {"classe_pred": classe_depuis_mm(axe), "n_amas": len(comps),
            "cellules_max": max(len(c) for c in comps), "axe_max_mm": axe}
 
 
def classes_depuis_patchs(df, col_positif="positif", um_par_cellule=UM_PAR_CELLULE):
    """df : colonnes slide, x_coord, y_coord, col_positif (bool). Une ligne par lame."""
    lignes = []
    for slide, g in df.groupby("slide"):
        pos = g[g[col_positif]]
        cel = np.column_stack([pos.x_coord // PAS_GRILLE_PX, pos.y_coord // PAS_GRILLE_PX]) if len(pos) else []
        lignes.append({"slide": slide, **classe_lame(cel, um_par_cellule)})
    return pd.DataFrame(lignes)
 
 
def appliquer_seuil(df, seuil, col_proba="proba", col_positif="positif"):
    """Ajoute la colonne bool `col_positif` = (proba >= seuil). Le seuil est OBLIGATOIRE : il se choisit sur
    la validation (val_ood ou validation interne selon un objectif explicite), jamais sur H2 (test final)."""
    if col_proba not in df.columns:
        raise ValueError(f"Colonne '{col_proba}' absente.")
    if not 0.0 <= seuil <= 1.0:
        raise ValueError(f"Le seuil doit être entre 0 et 1 (reçu {seuil}).")
    p = df[col_proba]
    if p.isna().any():
        raise ValueError("Des probabilités sont manquantes (NaN).")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Des probabilités sont hors de [0, 1] : fournir des probabilités, pas des logits.")
    return df.assign(**{col_positif: p >= seuil})
 
 
def charger_predictions(chemin_csv, meta):
    """Lit un CSV de prédictions (colonnes image_id, proba) et le rattache aux métadonnées (slide, x_coord, y_coord).
    Refuse les image_id inconnus et les doublons. Seuls les patchs prédits sont conservés : un patch SANS
    prédiction n'est pas un patch normal. La carte reste donc creuse si P3 ne prédit que le sous-échantillon."""
    pred = pd.read_csv(chemin_csv)
    for c in ("image_id", "proba"):
        if c not in pred.columns:
            raise ValueError(f"Colonne '{c}' absente de {chemin_csv}.")
    if pred.image_id.duplicated().any():
        raise ValueError("Des image_id sont en double dans les prédictions.")
    inconnus = ~pred.image_id.isin(meta.image_id)
    if inconnus.any():
        raise ValueError(f"{int(inconnus.sum())} image_id absents des métadonnées.")
    return pred[["image_id", "proba"]].merge(meta, on="image_id", how="left")
 
 
def classes_depuis_probas(df, seuil, um_par_cellule=UM_PAR_CELLULE, meta=None):
    """df : colonnes slide, x_coord, y_coord, proba. Applique le seuil puis classe chaque lame.
    Ajoute n_predits (patchs prédits dans la lame) et, si meta est fourni, part_predite (couverture)."""
    d = appliquer_seuil(df, seuil)
    out = classes_depuis_patchs(d, "positif", um_par_cellule)
    n = d.groupby("slide").size().rename("n_predits").reset_index()
    out = out.merge(n, on="slide")
    if meta is not None:
        tot = meta.groupby("slide").size().rename("n_total").reset_index()
        out = out.merge(tot, on="slide")
        out["part_predite"] = (out.n_predits / out.n_total).round(3)
    return out
 
 
def effet_sur_pn(classes_pred, lames, stages):
    """Remplace la classe officielle par la classe prédite pour les lames annotées et recalcule le stade pN.
    classes_pred : dict slide -> classe. Retourne une ligne par patiente ayant au moins une lame annotée.
    ecart = rang(stade substitué) - rang(stade officiel) ; négatif = stade sous-estimé (le plus grave)."""
    gl = stages[stages.patient.str.endswith(".tif")].copy()
    gl["pid"] = gl.patient.str[:11]
    officiel = dict(zip(gl.patient, gl.stage))
    subst = {nom: classes_pred[s] for s, nom in zip(lames.slide, lames.nom) if s in classes_pred}
    lignes = []
    for pid in sorted(lames.patient.unique()):
        noms = gl[gl.pid == f"patient_{pid:03d}"].patient
        avant = stade_pn([officiel[n] for n in noms])
        apres = stade_pn([subst.get(n, officiel[n]) for n in noms])
        lignes.append({"patient": pid, "stade_officiel": avant, "stade_substitue": apres,
                       "ecart": ORDRE_PN.index(apres) - ORDRE_PN.index(avant)})
    return pd.DataFrame(lignes)
 
 
def simuler_positifs(meta, p_fn, p_fp, graine):
    """Simulation d'un modèle imparfait : chaque patch tumoral est manqué avec proba p_fn,
    chaque patch normal est signalé à tort avec proba p_fp. Erreurs INDÉPENDANTES : en vrai elles sont
    corrélées spatialement. Ce n'est PAS un résultat de modèle."""
    rng = np.random.default_rng(graine)
    u = rng.random(len(meta))
    tum = meta.label.values == 1
    return np.where(tum, u >= p_fn, u < p_fp)
 
 
def main(dossier, sortie):
    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    meta, stages = charger(dossier)
    lames = table_lames(meta, stages)
 
    # Mode 1 : oracle (vrais labels)
    oracle = classes_depuis_patchs(meta.assign(positif=meta.label == 1))
    oracle = lames.merge(oracle, on="slide")
    oracle.to_csv(sortie / "oracle_p4.csv", index=False)
    ok = (oracle.classe == oracle.classe_pred).sum()
    print(f"Oracle : {ok}/{len(oracle)} lames dans la bonne classe (1 cellule = {UM_PAR_CELLULE:.1f} µm, hypothèse)")
    print(pd.crosstab(oracle.classe, oracle.classe_pred).to_string())
    eff = effet_sur_pn(dict(zip(oracle.slide, oracle.classe_pred)), lames, stages)
    print(f"Oracle -> stades pN modifiés : {(eff.ecart != 0).sum()}/{len(eff)} patientes concernées")
 
    # Mode 2 : simulation d'erreurs
    lignes = []
    for p_fn in (0.0, 0.1, 0.3, 0.5):
        for p_fp in (0.0, 0.02):
            for graine in range(3):
                pos = simuler_positifs(meta, p_fn, p_fp, graine)
                cl = classes_depuis_patchs(meta.assign(positif=pos)).merge(lames[["slide", "classe"]], on="slide")
                e = effet_sur_pn(dict(zip(cl.slide, cl.classe_pred)), lames, stages)
                lignes.append({"p_fn": p_fn, "p_fp": p_fp, "graine": graine,
                               "lames_correctes": int((cl.classe == cl.classe_pred).sum()),
                               "lames_total": len(cl),
                               "pn_modifies": int((e.ecart != 0).sum()),
                               "pn_sous_estimes": int((e.ecart < 0).sum()),
                               "pn_sur_estimes": int((e.ecart > 0).sum()),
                               "patientes": len(e)})
    sim = pd.DataFrame(lignes)
    sim.to_csv(sortie / "simulation_erreurs.csv", index=False)
    resume = sim.groupby(["p_fn", "p_fp"]).mean(numeric_only=True).drop(columns="graine").round(1)
    print("\nSimulation (moyenne de 3 graines ; erreurs indépendantes, pas un résultat de modèle)")
    print(resume.to_string())
 
 
def main_predictions(dossier, sortie, chemin_csv, seuil):
    """Mode prédictions : CSV (image_id, proba) + seuil -> classes de lames, matrice de confusion, effet sur le pN."""
    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    meta, stages = charger(dossier)
    lames = table_lames(meta, stages)
    df = charger_predictions(chemin_csv, meta)
    res = lames.merge(classes_depuis_probas(df, seuil, meta=meta), on="slide")
    res.to_csv(sortie / "classes_predites_p4.csv", index=False)
    print(f"Seuil = {seuil} | patchs prédits : {len(df)} sur {len(meta)}")
    print(f"Lames dans la bonne classe : {(res.classe == res.classe_pred).sum()}/{len(res)}")
    print(pd.crosstab(res.classe, res.classe_pred).to_string())
    print(f"Couverture médiane des lames : {res.part_predite.median():.1%} des patchs (un patch non prédit n'est pas un patch normal)")
    eff = effet_sur_pn(dict(zip(res.slide, res.classe_pred)), lames, stages)
    eff.to_csv(sortie / "effet_pn_predit.csv", index=False)
    print(f"Stades pN modifiés : {(eff.ecart != 0).sum()}/{len(eff)} patientes (sous-estimés : {(eff.ecart < 0).sum()})")
 
 
if __name__ == "__main__":
    if len(sys.argv) == 5:
        main_predictions(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
    else:
        main(sys.argv[1], sys.argv[2])