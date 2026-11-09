"""Audit P1 des métadonnées (aucune image requise).

Pourquoi : vérifier ce que le découpage et le sous-échantillon permettent
réellement avant que P2, P3 et P4 ne construisent dessus.

Usage : python audit_metadonnees.py <dossier_donnees> <dossier_sortie>
Le dossier de données contient metadata_complete.zip, ids_sous_echantillon.csv
et stages.csv. Sorties : audit_lames.csv, audit_oracle.csv.
"""
import sys
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

PAS_GRILLE_PX = 32          # CENTER_SIZE de WILDS, en pixels du niveau 2
UM_PAR_PX_NIVEAU0 = 0.24    # ordre de grandeur, à confirmer par scanner
UM_PAR_CELLULE = PAS_GRILLE_PX * 4 * UM_PAR_PX_NIVEAU0  # niveau 2 = facteur 4


def charger(dossier):
    dossier = Path(dossier)
    with zipfile.ZipFile(dossier / "metadata_complete.zip") as z:
        meta = pd.read_csv(z.open("metadata_complete.csv"))
    ids = pd.read_csv(dossier / "ids_sous_echantillon.csv")
    stages = pd.read_csv(dossier / "stages.csv")
    return meta, ids, stages


def composantes(cellules):
    """Composantes connexes (8 voisins) d'un ensemble de cellules (i, j)."""
    restantes, sorties = set(cellules), []
    while restantes:
        depart = restantes.pop()
        file, comp = deque([depart]), [depart]
        while file:
            i, j = file.popleft()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    v = (i + di, j + dj)
                    if v in restantes:
                        restantes.remove(v)
                        file.append(v)
                        comp.append(v)
        sorties.append(comp)
    return sorties


def grand_axe_mm(comp, um_par_cellule):
    """Plus grande distance entre deux cellules, sur l'enveloppe des extrêmes."""
    pts = np.array(comp, dtype=float)
    directions = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [2, 1], [1, 2], [2, -1], [1, -2]])
    extremes = set()
    for d in directions:
        proj = pts @ d
        extremes.add(tuple(pts[proj.argmin()]))
        extremes.add(tuple(pts[proj.argmax()]))
    e = np.array(list(extremes))
    dist = np.sqrt(((e[:, None] - e[None, :]) ** 2).sum(-1)).max()
    return (dist + 1) * um_par_cellule / 1000  # +1 : une cellule a une largeur


def classe_depuis_mm(mm):
    if mm <= 0.2:
        return "itc"
    return "micro" if mm <= 2.0 else "macro"


def main(dossier, sortie):
    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    meta, ids, stages = charger(dossier)
    pd.set_option("display.width", 180, "display.max_rows", 100)

    # 1. Table des lames et couverture du sous-échantillon
    sub = meta.merge(ids[["image_id"]], on="image_id")
    lames = meta.groupby("slide").agg(
        center=("center", "first"), patient=("patient", "first"), node=("node", "first"),
        split=("split", "first"), n=("label", "size"), tum=("label", "sum")).reset_index()
    lames["nom"] = [f"patient_{p:03d}_node_{n}.tif" for p, n in zip(lames.patient, lames.node)]
    classes = stages.rename(columns={"patient": "nom", "stage": "classe"})[["nom", "classe"]]
    lames = lames.merge(classes, on="nom", how="left")
    cover = sub.groupby("slide").agg(n_sub=("label", "size"), tum_sub=("label", "sum")).reset_index()
    lames = lames.merge(cover, on="slide", how="left").fillna({"n_sub": 0, "tum_sub": 0})
    lames.to_csv(sortie / "audit_lames.csv", index=False)

    print("== Concentration : part des patchs tumoraux du sous-échantillon venant de la 1re lame")
    for h, g in lames.groupby("center"):
        top = g.sort_values("tum_sub").iloc[-1]
        print(f"H{h}: lame {int(top.slide)} ({top.classe}) = {top.tum_sub / g.tum_sub.sum():.0%}"
              f" des {int(g.tum_sub.sum())} patchs tumoraux")

    print("\n== Part de chaque classe de lame dans les patchs tumoraux")
    print((lames.groupby("classe").tum.sum() / lames.tum.sum()).round(4).to_string())
    print("\n== Même calcul dans le sous-échantillon")
    print((lames.groupby("classe").tum_sub.sum() / lames.tum_sub.sum()).round(4).to_string())

    print("\n== Patchs tumoraux disponibles par lame ITC dans le sous-échantillon")
    print(lames[lames.classe == "itc"].groupby("center").tum_sub.apply(lambda s: s.astype(int).tolist()).to_string())

    # 2. Oracle : remonter à la classe avec les VRAIS labels, sans modèle.
    # Donne le plafond atteignable par toute méthode fondée sur cette grille.
    tum = meta[meta.label == 1]
    lignes = []
    for slide, g in tum.groupby("slide"):
        cellules = list(zip(g.x_coord // PAS_GRILLE_PX, g.y_coord // PAS_GRILLE_PX))
        comps = composantes(cellules)
        axes = [grand_axe_mm(c, UM_PAR_CELLULE) for c in comps]
        lignes.append({"slide": slide, "n_amas": len(comps), "cellules_max": max(map(len, comps)),
                       "axe_max_mm": max(axes), "oracle": classe_depuis_mm(max(axes))})
    oracle = lames[["slide", "center", "classe"]].merge(pd.DataFrame(lignes), on="slide")
    oracle.to_csv(sortie / "audit_oracle.csv", index=False)

    print(f"\n== Oracle (1 cellule ≈ {UM_PAR_CELLULE:.1f} µm)")
    print(pd.crosstab(oracle.classe, oracle.oracle))
    print("Concordance :", f"{(oracle.classe == oracle.oracle).mean():.0%}")
    print(oracle.groupby("classe").axe_max_mm.describe().round(2).to_string())

    # Effet du piège : compter 96 px au lieu de 32 px par cellule
    faux = [classe_depuis_mm(a * 3) for a in oracle.axe_max_mm]
    print("Concordance si on compte 96 px par cellule :", f"{(oracle.classe == faux).mean():.0%}")

    print("\n== Erreurs de l'oracle")
    print(oracle[oracle.classe != oracle.oracle].round(2).to_string(index=False))

    # 3. Couverture : le maillage sain a-t-il été éclairci ?
    print("\n== Part tumorale par hôpital (métadonnées complètes)")
    print(meta.groupby("center").label.mean().round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
