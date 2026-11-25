import sys
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slide_aggregator as sa  # noqa: E402
 
CMAP = ListedColormap(["white", "#9e9e9e", "#c0392b"])   # 0 absent, 1 normal, 2 tumoral
 
 
def grille_lame(g, col_tumoral="label"):
    """g : patchs d'une lame (x_coord, y_coord, col_tumoral). Retourne un tableau (lignes = y, colonnes = x)."""
    i = (g.x_coord // sa.PAS_GRILLE_PX).to_numpy()
    j = (g.y_coord // sa.PAS_GRILLE_PX).to_numpy()
    i, j = i - i.min(), j - j.min()
    grille = np.zeros((j.max() + 1, i.max() + 1), dtype=np.uint8)
    grille[j, i] = 1 + (g[col_tumoral].to_numpy() == 1)
    return grille
 
 
def tracer(ax, grille, titre, taille_titre=9):
    ax.imshow(grille, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(titre, fontsize=taille_titre)
    ax.set_xticks([]); ax.set_yticks([])
 
 
def main(dossier, sortie):
    sortie = Path(sortie) / "cartes_verite"
    sortie.mkdir(parents=True, exist_ok=True)
    meta, stages = sa.charger(dossier)
    lames = sa.table_lames(meta, stages)
    oracle = sa.classes_depuis_patchs(meta.assign(positif=meta.label == 1))
    lames = lames.merge(oracle[["slide", "classe_pred"]], on="slide").sort_values(["center", "slide"])
 
    fig_m, axes_m = plt.subplots(5, 10, figsize=(24, 13))
    compteur = {h: 0 for h in range(5)}
    for _, l in lames.iterrows():
        g = meta[meta.slide == l.slide]
        grille = grille_lame(g)
        n_tum = int((g.label == 1).sum())
        titre = f"lame {l.slide} · H{l.center}\n{l.classe} (oracle : {l.classe_pred}) · {n_tum} patchs tum."
        fig, ax = plt.subplots(figsize=(6, 6))
        tracer(ax, grille, titre)
        fig.savefig(sortie / f"lame_{int(l.slide):02d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        tracer(axes_m[l.center, compteur[l.center]], grille, f"{l.slide} · {l.classe}", 8)
        compteur[l.center] += 1
    for h in range(5):                       # cases vides éventuelles
        for k in range(compteur[h], 10):
            axes_m[h, k].axis("off")
        axes_m[h, 0].set_ylabel(f"H{h}", fontsize=12, rotation=0, labelpad=18)
    fig_m.suptitle("Vérité terrain : patchs tumoraux (rouge) par lame, une ligne par hôpital. "
                   "Titre = numéro de lame · classe officielle", fontsize=13)
    fig_m.tight_layout()
    fig_m.savefig(Path(sortie).parent / "cartes_verite_montage.png", dpi=90)
    plt.close(fig_m)
    print(f"{len(lames)} cartes écrites dans {sortie} ; montage : {Path(sortie).parent / 'cartes_verite_montage.png'}")
 
 
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])