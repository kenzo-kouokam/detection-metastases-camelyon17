"""Module P1 : découpage sans fuite et gestion de la couleur.

Deux responsabilités :
1. Découpage : le split `id_val` de WILDS partage ses lames avec `train`. On
   construit donc une validation interne par patiente, et on contrôle la fuite.
2. Couleur : normalisations (Reinhard, Macenko) et augmentations, toutes
   exposées derrière `get_transforms(mode)` pour comparer les variantes par hôpital.

Toute référence de couleur s'apprend sur H0, H3 et H4 uniquement : utiliser
H1 ou H2 pour l'apprendre ferait entrer la cible dans le prétraitement.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

HOPITAUX_ENTRAINEMENT = {0, 3, 4}
HOPITAL_VALIDATION = 1
HOPITAL_TEST = 2

# ---------------------------------------------------------------------------
# 1. Découpage et contrôle de fuite
# ---------------------------------------------------------------------------


def faire_split_interne(lames, n_par_classe=1, seed=42):
    """Réserve des patientes de H0/H3/H4 pour une validation interne propre.

    `lames` a une ligne par lame : slide, patient, center, classe, tum_sub.
    Pour chaque hôpital et chaque classe, on tire `n_par_classe` patientes.
    On écarte la lame dominante de l'hôpital (sinon l'entraînement perd la
    majorité de ses patchs tumoraux) et les lames sans patch tumoral dans le
    sous-échantillon (elles ne mesureraient aucun rappel).

    Renvoie {slide: "train_interne" | "val_interne"} pour les lames de
    H0/H3/H4. Une patiente reste entière d'un seul côté.
    """
    rng = np.random.default_rng(seed)
    pool = lames[lames.center.isin(HOPITAUX_ENTRAINEMENT)]
    reservees = set()
    for _, g in pool.groupby("center"):
        dominante = g.loc[g.tum_sub.idxmax(), "patient"]
        for classe in sorted(g.classe.unique()):
            deja = set(g[g.classe == classe].patient) & reservees
            manque = n_par_classe - len(deja)
            if manque <= 0:
                continue
            candidates = g[(g.classe == classe) & (g.tum_sub > 0) & (g.patient != dominante)]
            candidates = sorted(set(candidates.patient) - reservees)
            if candidates:
                tirees = rng.choice(candidates, size=min(manque, len(candidates)), replace=False)
                reservees.update(int(p) for p in tirees)
    return {int(row.slide): ("val_interne" if row.patient in reservees else "train_interne")
            for row in pool.itertuples()}


def controler_fuite(df, colonne_split, entrainement=("train", "train_interne")):
    """Lève une erreur si une lame ou une patiente traverse les jeux.

    `df` contient une ligne par patch avec slide, patient, center et la colonne
    de split. Vérifie aussi qu'aucun hôpital hors entraînement n'y figure.
    """
    erreurs = []
    for groupe in ("slide", "patient"):
        n_splits = df.groupby(groupe)[colonne_split].nunique()
        partages = n_splits[n_splits > 1]
        if len(partages):
            erreurs.append(f"{len(partages)} {groupe}(s) dans plusieurs jeux : {partages.index.tolist()[:10]}")
    centres = set(df[df[colonne_split].isin(entrainement)].center.unique())
    interdits = centres - HOPITAUX_ENTRAINEMENT
    if interdits:
        erreurs.append(f"hôpitaux interdits à l'entraînement : {sorted(interdits)}")
    if erreurs:
        raise ValueError("Fuite détectée : " + " ; ".join(erreurs))
    return True


# ---------------------------------------------------------------------------
# 2. Conversions de couleur (numpy seul : aucune dépendance en plus sur Colab)
# ---------------------------------------------------------------------------
_RGB_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])
_XYZ_RGB = np.linalg.inv(_RGB_XYZ)
_BLANC = np.array([0.95047, 1.0, 1.08883])


def rgb_vers_lab(rgb):
    """RGB uint8 (H, W, 3) vers CIE Lab (illuminant D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    t = c @ _RGB_XYZ.T / _BLANC
    f = np.where(t > 216 / 24389, np.cbrt(t), (24389 / 27 * t + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_vers_rgb(lab):
    """CIE Lab vers RGB uint8, valeurs ramenées dans [0, 255]."""
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    t = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _BLANC
    c = np.clip(t @ _XYZ_RGB.T, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def _densite_optique(rgb):
    return -np.log((np.asarray(rgb, dtype=np.float64) + 1.0) / 256.0)


# ---------------------------------------------------------------------------
# 3. Normalisations de coloration
# ---------------------------------------------------------------------------


def _verifier_hopitaux(centres):
    if centres is None:
        raise ValueError("Indiquez les hôpitaux d'origine des images de référence.")
    interdits = set(int(c) for c in centres) - HOPITAUX_ENTRAINEMENT
    if interdits:
        raise ValueError(f"Référence apprise sur des hôpitaux interdits : {sorted(interdits)}")


@dataclass
class Reinhard:
    """Aligne moyenne et écart-type de chaque canal Lab sur une référence.

    Simple et rapide. Limite : agit sur toute l'image, donc la part de tissu ou
    de fond d'un patch déplace ses statistiques.
    """
    moyenne: np.ndarray = None
    ecart: np.ndarray = None

    def fit(self, images, centres):
        _verifier_hopitaux(centres)
        labs = np.stack([rgb_vers_lab(i).reshape(-1, 3) for i in images])
        self.moyenne = labs.mean(axis=1).mean(axis=0)
        self.ecart = labs.std(axis=1).mean(axis=0)
        return self

    def __call__(self, image):
        rgb = np.asarray(image.convert("RGB")) if isinstance(image, Image.Image) else np.asarray(image)
        lab = rgb_vers_lab(rgb).reshape(-1, 3)
        m, s = lab.mean(0), lab.std(0) + 1e-6
        sortie = lab_vers_rgb(((lab - m) / s * self.ecart + self.moyenne).reshape(rgb.shape))
        return Image.fromarray(sortie) if isinstance(image, Image.Image) else sortie


def _vecteurs_macenko(od, seuil_od=0.15, alpha=1.0):
    """Estime les vecteurs de coloration H et E, ou None si trop peu de tissu."""
    tissu = od[(od > seuil_od).any(axis=1)]
    if len(tissu) < 50:
        return None
    _, vecteurs = np.linalg.eigh(np.cov(tissu.T))
    plan = vecteurs[:, 1:3]
    plan = plan * np.where(plan[0] < 0, -1, 1)
    proj = tissu @ plan
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    bas, haut = np.percentile(angles, [alpha, 100 - alpha])
    v1 = plan @ np.array([np.cos(bas), np.sin(bas)])
    v2 = plan @ np.array([np.cos(haut), np.sin(haut)])
    # Convention : l'hématoxyline absorbe davantage le rouge que l'éosine
    h, e = (v1, v2) if v1[0] > v2[0] else (v2, v1)
    return np.stack([h / np.linalg.norm(h), e / np.linalg.norm(e)], axis=1)


@dataclass
class Macenko:
    """Sépare hématoxyline et éosine par patch, puis recolore vers une référence.

    Respecte la physique de la coloration. Limite : sur 96 × 96 pixels peu
    colorés, l'estimation peut être instable. Le patch est alors rendu inchangé
    et compté dans `n_echecs`, à reporter dans l'analyse.
    """
    stains: np.ndarray = None
    conc_max: np.ndarray = None
    n_echecs: int = 0

    def fit(self, images, centres):
        _verifier_hopitaux(centres)
        od = np.concatenate([_densite_optique(np.asarray(i)).reshape(-1, 3) for i in images])
        self.stains = _vecteurs_macenko(od)
        if self.stains is None:
            raise ValueError("Pas assez de tissu dans les images de référence.")
        conc = np.linalg.lstsq(self.stains, od.T, rcond=None)[0]
        self.conc_max = np.percentile(conc, 99, axis=1)
        return self

    def __call__(self, image):
        est_pil = isinstance(image, Image.Image)
        rgb = np.asarray(image.convert("RGB")) if est_pil else np.asarray(image)
        od = _densite_optique(rgb).reshape(-1, 3)
        stains = _vecteurs_macenko(od)
        if stains is None:
            self.n_echecs += 1
            return image
        conc = np.linalg.lstsq(stains, od.T, rcond=None)[0]
        conc_max = np.percentile(conc, 99, axis=1)
        if not np.all(conc_max > 1e-6):
            self.n_echecs += 1
            return image
        conc = np.clip(conc * (self.conc_max / conc_max)[:, None], 0, None)
        sortie = 256.0 * np.exp(-(self.stains @ conc)).T - 1.0
        sortie = np.clip(np.round(sortie), 0, 255).astype(np.uint8).reshape(rgb.shape)
        return Image.fromarray(sortie) if est_pil else sortie


# ---------------------------------------------------------------------------
# 4. Augmentation dans l'espace des colorants
# ---------------------------------------------------------------------------
# Matrice de Ruifrok & Johnston (2001) : hématoxyline, éosine, résiduel.
_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
_HED = _HED / np.linalg.norm(_HED, axis=1, keepdims=True)
_HED_INV = np.linalg.inv(_HED)


@dataclass
class JitterHED:
    """Perturbe la quantité de chaque colorant (Tellez et al., 2018).

    Simule une autre coloration, ce que ColorJitter ne fait qu'indirectement.
    """
    sigma: float = 0.05
    biais: float = 0.05
    seed: int = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, image):
        est_pil = isinstance(image, Image.Image)
        rgb = np.asarray(image.convert("RGB")) if est_pil else np.asarray(image)
        conc = _densite_optique(rgb).reshape(-1, 3) @ _HED_INV
        a = self.rng.uniform(1 - self.sigma, 1 + self.sigma, 3)
        b = self.rng.uniform(-self.biais, self.biais, 3)
        sortie = 256.0 * np.exp(-((conc * a + b) @ _HED)) - 1.0
        sortie = np.clip(np.round(sortie), 0, 255).astype(np.uint8).reshape(rgb.shape)
        return Image.fromarray(sortie) if est_pil else sortie


# ---------------------------------------------------------------------------
# 5. Interface unique pour P3 : une variante, un mot
# ---------------------------------------------------------------------------
MODES = ("aucun", "colorjitter", "hed", "reinhard", "macenko", "gris")
MOY_IMAGENET, STD_IMAGENET = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_transforms(mode, reference=None):
    """Renvoie (transform_entrainement, transform_evaluation) pour une variante.

    `reference` : objet Reinhard ou Macenko déjà ajusté, requis pour ces modes.
    """
    from torchvision import transforms as T

    if mode not in MODES:
        raise ValueError(f"Mode inconnu : {mode}. Modes possibles : {MODES}")
    if mode in ("reinhard", "macenko") and reference is None:
        raise ValueError(f"Le mode {mode} demande une référence ajustée sur H0, H3 et H4.")

    avant, augmentation = [], []
    if mode in ("reinhard", "macenko"):
        avant = [reference]
    elif mode == "gris":
        avant = [T.Grayscale(num_output_channels=3)]
    elif mode == "colorjitter":
        augmentation = [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)]
    elif mode == "hed":
        augmentation = [JitterHED()]

    fin = [T.ToTensor(), T.Normalize(MOY_IMAGENET, STD_IMAGENET)]
    entrainement = T.Compose(avant + [T.RandomHorizontalFlip(), T.RandomVerticalFlip()] + augmentation + fin)
    evaluation = T.Compose(avant + fin)
    return entrainement, evaluation


# ---------------------------------------------------------------------------
# 6. Mesure du domain shift
# ---------------------------------------------------------------------------


def stats_lab(images, centres):
    """Moyenne Lab par patch, en DataFrame avec l'hôpital d'origine."""
    moyennes = np.stack([rgb_vers_lab(np.asarray(i)).reshape(-1, 3).mean(0) for i in images])
    return pd.DataFrame(moyennes, columns=["L", "a", "b"]).assign(center=list(centres))


def distances_lab(stats):
    """Distance Delta-E entre les couleurs moyennes de chaque paire d'hôpitaux."""
    m = stats.groupby("center")[["L", "a", "b"]].mean()
    d = np.sqrt(((m.values[:, None] - m.values[None, :]) ** 2).sum(-1))
    return pd.DataFrame(d, index=m.index, columns=m.index)
