import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from slide_aggregator import charger  # noqa: E402


def verifier(chemin_csv, meta, centre=1):
    pred = pd.read_csv(chemin_csv)
    erreurs = []
    for c in ("image_id", "proba"):
        if c not in pred.columns:
            return [f"Colonne '{c}' absente."], {}
    attendu = meta[meta.center == centre]
    if pred.image_id.duplicated().any():
        erreurs.append(f"{int(pred.image_id.duplicated().sum())} doublons d'image_id.")
    if pred.proba.isna().any():
        erreurs.append(f"{int(pred.proba.isna().sum())} proba manquantes.")
    if ((pred.proba < 0) | (pred.proba > 1)).any():
        erreurs.append("Des proba sont hors de [0, 1].")
    inconnus = int((~pred.image_id.isin(meta.image_id)).sum())
    if inconnus:
        erreurs.append(f"{inconnus} image_id absents des métadonnées.")
    hors_centre = int((pred.image_id.isin(meta.image_id) & ~pred.image_id.isin(attendu.image_id)).sum())
    if hors_centre:
        erreurs.append(f"{hors_centre} patchs n'appartiennent pas à l'hôpital {centre} (H2 interdit avant l'évaluation finale).")
    if (meta[meta.image_id.isin(pred.image_id)].center == 2).any():
        erreurs.append("ALERTE : des patchs de l'hôpital 2 (test) sont présents.")
    manquants = int((~attendu.image_id.isin(pred.image_id)).sum())
    if manquants:
        erreurs.append(f"{manquants} patchs de l'hôpital {centre} sans prédiction (attendu {len(attendu)}).")
    couv = (attendu.assign(ok=attendu.image_id.isin(pred.image_id)).groupby("slide").ok.mean().round(3))
    resume = {"lignes": len(pred), "attendu": len(attendu), "proba_min": float(pred.proba.min()),
              "proba_max": float(pred.proba.max()), "couverture_min_par_lame": float(couv.min())}
    return erreurs, resume


def main(dossier, chemin_csv, centre=1):
    meta, _ = charger(dossier)
    erreurs, resume = verifier(chemin_csv, meta, centre)
    print(resume)
    for e in erreurs:
        print("ERREUR :", e)
    if erreurs:
        sys.exit(1)
    print("OK : fichier utilisable par slide_aggregator.py")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
