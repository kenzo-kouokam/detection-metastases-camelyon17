# Guide de l'équipe

## 1. Le projet en clair

Quand une patiente a un cancer du sein, on prélève des ganglions sous le bras. Un pathologiste les observe au microscope pour y chercher des métastases, c'est-à-dire des cellules cancéreuses qui ont migré. Ce résultat fixe le stade de la maladie, appelé stade pN, et donc le traitement. L'examen est long, et une petite métastase peut échapper à l'œil.

On nous demande de construire un outil qui aide le pathologiste. Il ne le remplace pas.

### Les quatre niveaux à retenir

| Niveau | Ce que c'est | Ce qu'on en fait |
|---|---|---|
| Patch | Petit carré d'image de 96 x 96 pixels | Le modèle dit : tumoral ou non |
| Lame | Image géante d'un ganglion, plusieurs milliards de pixels | On assemble les patchs en carte et on mesure la plus grande zone tumorale |
| Ganglion | Une lame correspond à un ganglion | Classe selon la taille : négatif, ITC (0,2 mm ou moins), micro (jusqu'à 2 mm), macro (plus de 2 mm) |
| Patiente | 5 ganglions par patiente | Le stade pN se déduit de règles fixes, le modèle ne le prédit pas |

### Pourquoi c'est difficile

Les hôpitaux n'utilisent ni les mêmes scanners ni les mêmes colorations, donc leurs images n'ont pas les mêmes couleurs. Un modèle peut retenir ces couleurs au lieu de la forme des cellules. Il brille alors sur les hôpitaux connus et chute ailleurs. C'est le domain shift.

D'où le découpage imposé :

| Hôpitaux | Rôle |
|---|---|
| H0, H3, H4 | Entraînement |
| H1 | Réglages et choix du modèle |
| H2 | Test final, utilisé une seule fois, à la toute fin |

### Une limite à connaître

Les images annotées viennent de 50 lames, toutes positives, et d'au plus 3 ganglions par patiente. On ne peut donc pas prédire le stade d'une patiente de bout en bout. On prédit la classe des lames annotées, on la remplace dans le fichier officiel, et on compte les stades qui changent.

## 2. Les documents reçus

| Document | Ce qu'il contient | À quoi il sert |
|---|---|---|
| `SPE1_kickoff_slides.pdf` | Le résumé du projet présenté le 30 octobre | Avoir la vue d'ensemble |
| `01_Sujet_SPE1_histopathologie.pdf` | Le sujet complet, les 5 étapes, les livrables, la grille de notation | La référence : en cas de doute, c'est lui qui tranche |
| `02_Regles_communes_SPE.pdf` | Les règles de rendu, d'usage de l'IA, de plagiat | Éviter les notes de 0 |
| `LISEZMOI.txt` | La marche à suivre pour démarrer sur Colab | Lancer le notebook |
| `SPE1_notebook_demarrage.ipynb` | Un code qui fonctionne de bout en bout | Point de départ à reprendre et à améliorer, ce n'est pas la solution |
| `donnees/ids_sous_echantillon.csv` | Les numéros des 39 000 patchs à utiliser et leur jeu | Dire quelles images garder. Il ne contient aucune image |
| `donnees/metadata_complete.zip` | Pour les 455 954 patchs : hôpital, lame, patiente, ganglion, position, label | Dessiner les cartes de lames. Il ne contient aucune image |
| `donnees/stages.csv` | La classe officielle des 500 ganglions et le stade des 100 patientes | La vérité à laquelle on compare nos résultats |
| `requirements.txt` | La liste des bibliothèques Python | Installer l'environnement hors de Colab |
| `FEUILLE_DE_ROUTE_PROJET_CAMELYON17.pdf` | Notre répartition des rôles et nos 8 phases | Notre organisation interne, à ajuster |

Les PDF et le notebook du cours restent en dehors de ce dépôt public.

## 3. Les rôles

| Rôle | Mission | Livrable principal |
|---|---|---|
| P1 Données et couleur | Exploration, couleur, normalisation, découpage sans fuite | `preprocessing.py`, partie données du rapport |
| P2 Référence simple | Modèle de base sur couleur et texture | `01_baseline_ml.ipynb` |
| P3 Deep learning | ResNet par transfert, entraînement | `02_resnet.ipynb`, poids du modèle |
| P4 Agrégation | Cartes de lames, classes de ganglions, stade pN | `slide_aggregator.py` |
| P5 Évaluation et éthique | Test final, Grad-CAM, faux négatifs, AI Act, coordination | Planches Grad-CAM, partie éthique |

L'enchaînement est le suivant. P1 fournit le découpage à P2 et à P3. P3 fournit les prédictions à P4 et à P5. Sans ces deux maillons, les autres sont bloqués.

## 4. Les images : le fichier parquet

Un fichier parquet est un format de tableau compressé, comparable à un gros CSV mais bien plus efficace, qui peut stocker des images dans une colonne.

`camelyon17_sous_echantillon.parquet` n'est dans aucun dossier reçu : il faut le fabriquer une fois. Le notebook de démarrage télécharge environ 10 Go depuis Hugging Face, ne garde que les 39 000 patchs listés dans `ids_sous_echantillon.csv`, puis écrit ce fichier. Sur Colab, il disparaît à la fin de la session : on le copie donc sur Google Drive.

### Le fabriquer, une seule personne suffit

1. Ouvrir https://colab.research.google.com, puis Fichier, Importer un notebook, et choisir `SPE1_notebook_demarrage.ipynb`.
2. Dans le panneau Fichiers, à gauche, importer les trois fichiers du dossier `donnees/`.
3. Exécuter les cellules des sections 0, 1 et 2 seulement. Compter 10 à 20 minutes.
4. La sortie doit afficher `39000 patchs`.
5. Dans une nouvelle cellule, copier le fichier sur Drive :

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/SPE1
!cp camelyon17_sous_echantillon.parquet /content/drive/MyDrive/SPE1/
```

6. Partager le dossier `SPE1` avec l'équipe. Chacun ajoute un raccourci vers ce dossier dans son Drive et met `SOURCE = "drive"` dans le notebook.

## 5. Utiliser le découpage de l'équipe

Le jeu `id_val` de WILDS partage ses 30 lames avec `train`, ce que le sujet interdit pour nos validations internes. Nous utilisons la colonne `split_equipe` de `p1_data/sorties/splits_par_patch.csv` :

```python
import pandas as pd

splits = pd.read_csv("p1_data/sorties/splits_par_patch.csv")
df = df.merge(splits[["image_id", "split_equipe"]], on="image_id")
```

| `split_equipe` | Rôle |
|---|---|
| `train_interne` | Entraînement |
| `val_interne` | Validation sur des patientes jamais vues, mêmes hôpitaux |
| `val_ood` | Réglages sur H1, un hôpital jamais vu |
| `test` | H2, une seule fois, à la fin |

## 6. Travailler sur le dépôt

```bash
git pull
```

```bash
git checkout -b p2-baseline
```

Chacun travaille sur sa branche et ouvre une pull request vers `main`.

- Aucune image, aucun parquet, aucun poids de modèle dans le dépôt : ils restent sur le Drive.
- Sur un notebook, vider les sorties avant de valider, sauf les figures utiles au rapport.
- Chaque usage d'un outil d'IA se note dans `docs/usage_ia.md`. Un usage non déclaré vaut 0.
