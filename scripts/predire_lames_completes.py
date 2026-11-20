"""Script d'inférence complète pour P4 (slide_aggregator).

Ce script résout la limitation du sous-échantillon (qui ne contient que 4 000 patchs de H1 sur 34 904,
faisant chuter la classification de lames à 13/50 même avec l'oracle à cause des trous spatiaux).

Il prédit l'INTEGRALITE des 34 904 patchs des 10 lames de val_ood (Hôpital 1) depuis les shards
Hugging Face originaux, puis les patchs de val_interne.

Sortie : un CSV propre avec 2 colonnes sans doublons :
image_id,proba
"""
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms as T
    import pyarrow.parquet as pq
    import pyarrow as pa
    A_DEPENDANCES = True
except ImportError:
    A_DEPENDANCES = False


HF_API_URL = "https://huggingface.co/api/datasets/wltjr1007/Camelyon17-WILDS/parquet/default"
MOY_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


def charger_modele_resnet(chemin_poids, device):
    """Reconstruit l'architecture ResNet-18 de P3 et charge best_model.pth."""
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    )
    if not os.path.exists(chemin_poids):
        raise FileNotFoundError(f"Fichier de poids introuvable : {chemin_poids}")
    
    state = torch.load(chemin_poids, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def predire_batch_images(model, liste_bytes, transform, device, batch_size=128):
    """Calcule les probabilités tumorales par mini-batchs."""
    probas = []
    n = len(liste_bytes)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_bytes = liste_bytes[i:i+batch_size]
            tensors = []
            for b in batch_bytes:
                img = Image.open(io.BytesIO(b)).convert('RGB')
                tensors.append(transform(img))
            batch_tensors = torch.stack(tensors).to(device)
            outputs = model(batch_tensors)
            p = torch.softmax(outputs, dim=1)[:, 1]
            probas.extend(p.cpu().numpy().tolist())
    return probas


def predire_val_ood_complet(chemin_modele, dossier_sortie="sorties", device=None):
    """Télécharge les 3 shards de val_ood (H1) depuis Hugging Face et génère predictions_val_ood.csv."""
    if not A_DEPENDANCES:
        raise RuntimeError("torch, torchvision ou pyarrow n'est pas installé dans cet environnement.")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilisé : {device}")
    
    dossier_sortie = Path(dossier_sortie)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    fichier_csv = dossier_sortie / "predictions_val_ood.csv"
    
    print("\n1. Chargement du modèle...")
    model = charger_modele_resnet(chemin_modele, device)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(MOY_IMAGENET, STD_IMAGENET)
    ])
    
    print("\n2. Récupération des URLs des shards de validation (H1) depuis Hugging Face...")
    shards_info = json.load(urllib.request.urlopen(HF_API_URL))
    urls_val = shards_info.get("validation", [])
    print(f"Trouvé {len(urls_val)} shard(s) pour validation (34 904 patchs attendus).")
    
    lignes_pred = []
    total_vus = 0
    start_t = time.time()
    
    for idx, url in enumerate(urls_val):
        print(f"\n--- Traitement Shard {idx+1}/{len(urls_val)} ---")
        f_tmp = f"tmp_shard_val_{idx}.parquet"
        print(f"Téléchargement : {url} ...")
        urllib.request.urlretrieve(url, f_tmp)
        
        table = pq.read_table(f_tmp)
        df_shard = table.to_pandas()
        os.remove(f_tmp)
        
        n_shard = len(df_shard)
        total_vus += n_shard
        print(f"Shard {idx+1} chargé : {n_shard} patchs. Inférence en cours...")
        
        # Extraction des octets d'images
        bytes_list = [d["bytes"] if isinstance(d, dict) else d for d in df_shard["image"]]
        image_ids = df_shard["image_id"].values
        
        # Inférence par batchs
        p = predire_batch_images(model, bytes_list, transform, device)
        
        for i_id, prob in zip(image_ids, p):
            lignes_pred.append({"image_id": int(i_id), "proba": float(prob)})
            
        print(f"Shard {idx+1} terminé ({len(lignes_pred)} patchs cumulés).")
    
    df_pred = pd.DataFrame(lignes_pred)
    
    # Contrôles stricts
    print("\n3. Contrôles qualité :")
    print(f"Total patchs prédits : {len(df_pred)} (attendu : 34 904)")
    if df_pred["image_id"].duplicated().any():
        print("ATTENTION : Détection de doublons ! Nettoyage...")
        df_pred = df_pred.drop_duplicates(subset=["image_id"])
    else:
        print(" Aucun doublon détecté.")
    
    # Sauvegarde
    df_pred[["image_id", "proba"]].to_csv(fichier_csv, index=False)
    duree = time.time() - start_t
    print(f"\n Fichier généré avec succès en {duree:.1f}s : {fichier_csv}")
    print(f"Aperçu :\n{df_pred.head()}")
    return df_pred


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python scripts/predire_lames_completes.py <chemin_vers_best_model.pth> [dossier_sortie]")
        sys.exit(1)
    poids = sys.argv[1]
    sortie = sys.argv[2] if len(sys.argv) > 2 else "sorties"
    predire_val_ood_complet(poids, sortie)
