import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HF_API = "https://huggingface.co/api/datasets/wltjr1007/Camelyon17-WILDS/parquet/default"
MOY, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def charger_modele(poids, device):
    import torch
    import torch.nn as nn
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2))
    m.load_state_dict(torch.load(poids, map_location=device))
    return m.to(device).eval()


def predire(model, liste_bytes, device, batch=128):
    import torch
    from PIL import Image
    from torchvision import transforms as T
    tr = T.Compose([T.ToTensor(), T.Normalize(MOY, STD)])
    out = []
    with torch.no_grad():
        for i in range(0, len(liste_bytes), batch):
            x = torch.stack([tr(Image.open(io.BytesIO(b)).convert("RGB")) for b in liste_bytes[i:i + batch]])
            out.extend(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy().tolist())
    return out


def main(poids, dossier_donnees, sortie):
    import pyarrow.parquet as pq
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sortie = Path(sortie); sortie.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(Path(dossier_donnees) / "metadata_complete.zip") as z:
        meta = pd.read_csv(z.open("metadata_complete.csv"))
    h1 = set(meta.loc[meta.center == 1, "image_id"])
    print(f"Appareil : {device} | patchs H1 attendus : {len(h1)}")
    model = charger_modele(poids, device)
    urls = json.load(urllib.request.urlopen(HF_API))["validation"]
    print(f"{len(urls)} fichier(s) 'validation'")
    morceaux = []
    for i, url in enumerate(urls):
        part = sortie / f"_val_shard_{i}.csv"
        if part.exists():
            morceaux.append(pd.read_csv(part)); print(f"shard {i + 1}: repris"); continue
        tmp = sortie / f"_tmp_{i}.parquet"
        urllib.request.urlretrieve(url, tmp)
        df = pq.read_table(tmp).to_pandas(); tmp.unlink()
        df = df[df["image_id"].isin(h1)]
        p = predire(model, [d["bytes"] if isinstance(d, dict) else d for d in df["image"]], device)
        res = pd.DataFrame({"image_id": df["image_id"].astype(int).values, "proba": p})
        res.to_csv(part, index=False); morceaux.append(res)
        print(f"shard {i + 1}/{len(urls)} : {len(res)} patchs")
    pred = pd.concat(morceaux, ignore_index=True)
    assert not pred.image_id.duplicated().any(), "doublons"
    assert set(pred.image_id) == h1, f"{len(h1 - set(pred.image_id))} patchs H1 manquants : le split 'validation' n'est pas H1 ?"
    pred.to_csv(sortie / "predictions_val_ood.csv", index=False)
    print("OK :", sortie / "predictions_val_ood.csv", len(pred), "lignes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "donnees", sys.argv[3] if len(sys.argv) > 3 else "sorties")
