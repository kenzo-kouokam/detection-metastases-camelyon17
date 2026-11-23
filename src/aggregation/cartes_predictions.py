import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slide_aggregator as sa  # noqa: E402

LEGENDE = [Patch(color="#d9d9d9", label="vrai négatif"), Patch(color="#c0392b", label="vrai positif"),
           Patch(color="#f39c12", label="faux positif"), Patch(color="#2e6fba", label="faux négatif")]


def positions(g):
    i = (g.x_coord // sa.PAS_GRILLE_PX).to_numpy()
    j = (g.y_coord // sa.PAS_GRILLE_PX).to_numpy()
    return i - i.min(), j - j.min()


def grille(g, codes):
    i, j = positions(g)
    t = np.zeros((j.max() + 1, i.max() + 1), dtype=np.uint8)
    t[j, i] = codes
    return t


def codes_erreurs(vrai, pred):
    return np.select([(vrai == 1) & (pred == 1), (vrai == 0) & (pred == 1), (vrai == 1) & (pred == 0)], [2, 3, 4], default=1)


COULEURS = {1: "#d9d9d9", 2: "#c0392b", 3: "#f39c12", 4: "#2e6fba"}


def tracer(ax, t, titre, taille=9, pt=14):
    for code in (1, 2, 3, 4):
        j, i = np.nonzero(t == code)
        ax.scatter(i, j, s=pt if code == 1 else pt * 2.5, marker="s", linewidths=0, color=COULEURS[code], zorder=code)
    ax.set_xlim(-1, t.shape[1]); ax.set_ylim(t.shape[0], -1)
    ax.set_aspect("equal")
    ax.set_title(titre, fontsize=taille)
    ax.set_xticks([]); ax.set_yticks([])


def main(dossier, sortie, chemin_csv, seuil):
    sortie = Path(sortie) / "cartes_predictions"
    sortie.mkdir(parents=True, exist_ok=True)
    meta, stages = sa.charger(dossier)
    lames = sa.table_lames(meta, stages)
    pred = sa.charger_predictions(chemin_csv, meta)
    pred = sa.appliquer_seuil(pred, seuil)
    classes = sa.classes_depuis_probas(pred, seuil, meta=meta)
    lames = lames[lames.slide.isin(pred.slide.unique())].merge(classes[["slide", "classe_pred", "axe_max_mm"]], on="slide").sort_values("slide")

    colonnes = 6
    lignes = int(np.ceil(len(lames) / 2))
    fig_m, axes_m = plt.subplots(lignes, colonnes, figsize=(5 * colonnes, 8 * lignes))
    axes_m = np.atleast_2d(axes_m)
    for k, (_, l) in enumerate(lames.iterrows()):
        g = pred[pred.slide == l.slide]
        vrai = (g.label == 1).astype(int).to_numpy()
        p = g.positif.astype(int).to_numpy()
        fn, fp = int(((vrai == 1) & (p == 0)).sum()), int(((vrai == 0) & (p == 1)).sum())
        panneaux = [(grille(g, 1 + vrai), f"lame {l.slide} · vérité\n{l.classe}"),
                    (grille(g, 1 + p), f"prédiction (seuil {seuil:.3f})\n{l.classe_pred} · axe {l.axe_max_mm:.2f} mm"),
                    (grille(g, codes_erreurs(vrai, p)), f"erreurs : {fn} FN · {fp} FP")]
        fig, axes = plt.subplots(1, 3, figsize=(18, 9))
        for ax, (t, titre) in zip(axes, panneaux):
            tracer(ax, t, titre, 11, 30)
        fig.legend(handles=LEGENDE, loc="lower center", ncol=4, fontsize=9)
        fig.savefig(sortie / f"lame_{int(l.slide):02d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        r, c = divmod(k, 2)
        for q, (t, titre) in enumerate(panneaux):
            tracer(axes_m[r, c * 3 + q], t, titre, 12, 30)
    for ax in axes_m.ravel():
        if not ax.collections:
            ax.axis("off")
    fig_m.legend(handles=LEGENDE, loc="lower center", ncol=4, fontsize=11)
    fig_m.suptitle(f"Hôpital 1 : vérité, prédiction ResNet-18 et erreurs par patch (seuil {seuil:.3f}). Seuil choisi sur cet hôpital.", fontsize=13)
    fig_m.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig_m.savefig(Path(sortie).parent / "cartes_predictions_montage.png", dpi=90)
    plt.close(fig_m)
    print(f"{len(lames)} lames : cartes dans {sortie} ; montage : {Path(sortie).parent / 'cartes_predictions_montage.png'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
