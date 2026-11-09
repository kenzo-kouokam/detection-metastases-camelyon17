"""Produit le découpage officiel de l'équipe : splits_par_patch.csv.

Usage : python creer_split.py <dossier_donnees> <dossier_sortie>

Colonnes : image_id, slide, patient, center, label, split_wilds, split_equipe.
split_equipe vaut train_interne, val_interne, val_ood ou test. Les patchs
`id_val` de WILDS sont reversés dans le jeu de leur lame, ce qui supprime la
fuite tout en gardant leurs images.
"""
import sys
from pathlib import Path

import pandas as pd

from audit_metadonnees import charger
from preprocessing import controler_fuite, faire_split_interne


def main(dossier, sortie):
    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    meta, ids, stages = charger(dossier)
    sub = meta.drop(columns="split").merge(ids.rename(columns={"split": "split_wilds"}), on="image_id")

    lames = meta.groupby("slide").agg(patient=("patient", "first"), node=("node", "first"),
                                      center=("center", "first")).reset_index()
    lames["nom"] = [f"patient_{p:03d}_node_{n}.tif" for p, n in zip(lames.patient, lames.node)]
    classes = stages.rename(columns={"patient": "nom", "stage": "classe"})[["nom", "classe"]]
    lames = lames.merge(classes, on="nom")
    tum_sub = sub.groupby("slide").label.sum().rename("tum_sub").reset_index()
    lames = lames.merge(tum_sub, on="slide", how="left").fillna({"tum_sub": 0})

    affectation = faire_split_interne(lames)
    sub["split_equipe"] = sub.slide.map(affectation)
    sub.loc[sub.center == 1, "split_equipe"] = "val_ood"
    sub.loc[sub.center == 2, "split_equipe"] = "test"
    if sub.split_equipe.isna().any():
        raise ValueError("Des patchs n'ont pas de jeu attribué.")

    controler_fuite(sub, "split_equipe")

    colonnes = ["image_id", "slide", "patient", "center", "label", "split_wilds", "split_equipe"]
    sub[colonnes].sort_values("image_id").to_csv(sortie / "splits_par_patch.csv", index=False)

    pd.set_option("display.width", 160)
    print(pd.crosstab(sub.split_equipe, [sub.center, sub.label]).to_string())
    val = lames[lames.slide.map(affectation) == "val_interne"]
    print("\nLames réservées :")
    print(val[["slide", "patient", "center", "classe", "tum_sub"]].astype({"tum_sub": int}).to_string(index=False))
    print("\nLe contrôle signale-t-il bien le split WILDS d'origine ?")
    try:
        controler_fuite(sub[sub.split_wilds.isin(["train", "id_val"])], "split_wilds")
        print("Non : anomalie à examiner.")
    except ValueError as e:
        print("Oui :", str(e)[:110])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
