import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CLASSES = ["itc", "micro", "macro"]


def matrice(classes_predites):
    m = pd.crosstab(classes_predites["classe"], classes_predites["classe_pred"])
    return m.reindex(index=CLASSES, columns=CLASSES, fill_value=0)


def rappel_precision(m):
    diag = np.diag(m.values)
    return pd.DataFrame({"classe": CLASSES, "effectif_reel": m.sum(axis=1).values, "effectif_predit": m.sum(axis=0).values,
                         "rappel": np.where(m.sum(axis=1) > 0, diag / m.sum(axis=1).clip(lower=1), np.nan).round(3),
                         "precision": np.where(m.sum(axis=0) > 0, diag / m.sum(axis=0).clip(lower=1), np.nan).round(3)})


def tracer(m, chemin, titre):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(m.values, cmap="Blues", vmin=0, vmax=max(m.values.max(), 1))
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("classe prédite"); ax.set_ylabel("classe officielle")
    for i in range(3):
        for j in range(3):
            v = int(m.values[i, j])
            ax.text(j, i, v, ha="center", va="center", fontsize=16, color="white" if v > m.values.max() / 2 else "black")
    ax.set_title(titre, fontsize=10)
    fig.tight_layout()
    fig.savefig(chemin, dpi=130)
    plt.close(fig)


def main(chemin_classes, sortie):
    sortie = Path(sortie)
    cp = pd.read_csv(chemin_classes)
    m = matrice(cp)
    m.to_csv(sortie / "matrice_confusion_lames.csv")
    rp = rappel_precision(m)
    rp.to_csv(sortie / "rappel_precision_lames.csv", index=False)
    n, ok = int(m.values.sum()), int(np.trace(m.values))
    tracer(m, sortie.parent / "figures" / "matrice_confusion_lames.png", f"Lames de H1 : {ok}/{n} bien classées")
    print(m.to_string()); print(rp.to_string(index=False)); print(f"{ok}/{n} lames bien classées")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
