"""Charge le parquet partagé et y ajoute le découpage de l'équipe.

Tout le monde passe par cette fonction : mêmes patchs, mêmes jeux, et le
contrôle de fuite s'exécute à chaque chargement.
"""
import io
from pathlib import Path

import pandas as pd
from PIL import Image

from preprocessing import controler_fuite

CHEMIN_DRIVE = "/content/drive/MyDrive/SPE1/camelyon17_sous_echantillon.parquet"
SPLITS_CSV = Path(__file__).parent / "sorties" / "splits_par_patch.csv"
N_PATCHS_ATTENDU = 39000


def verifier_parquet(df, splits):
    """Lève une erreur si le parquet ne correspond pas à la liste de référence."""
    if len(df) != N_PATCHS_ATTENDU:
        raise ValueError(f"{len(df)} patchs au lieu de {N_PATCHS_ATTENDU}.")
    if df.image_id.duplicated().any():
        raise ValueError("Des patchs apparaissent plusieurs fois.")
    if set(df.image_id) != set(splits.image_id):
        raise ValueError("Les patchs ne correspondent pas à la liste de référence.")
    apercu = Image.open(io.BytesIO(df.image.iloc[0]))
    if apercu.size != (96, 96):
        raise ValueError(f"Taille d'image inattendue : {apercu.size}.")
    return True


def charger_donnees(chemin=CHEMIN_DRIVE, splits_csv=SPLITS_CSV):
    """Renvoie les patchs avec les colonnes split_equipe et patient."""
    df = pd.read_parquet(chemin)
    splits = pd.read_csv(splits_csv)
    verifier_parquet(df, splits)
    a_ajouter = ["image_id", "split_equipe"] + (["patient"] if "patient" not in df.columns else [])
    df = df.merge(splits[a_ajouter], on="image_id", validate="one_to_one")
    controler_fuite(df, "split_equipe")
    return df
