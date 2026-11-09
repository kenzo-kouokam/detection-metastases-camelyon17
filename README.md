# Détection de métastases dans des images de ganglions

Projet de spécialisation 1, MBA 2 Data & BI, Epitech, 2026-2027.

À partir d'images de ganglions lymphatiques issues de cinq hôpitaux, nous entraînons un modèle qui repère les zones tumorales, puis nous remontons à une classe par ganglion et mesurons l'effet des erreurs sur le stade pN des patientes. La question centrale est la tenue du modèle dans un hôpital qu'il n'a jamais vu.

## Dates

| Étape | Date |
|---|---|
| Follow-up (20 % de la note) | vendredi 13 novembre 2026 |
| Dépôt des livrables | jeudi 26 novembre 2026, 18h00, aucun retard accepté |
| Soutenance | vendredi 27 novembre 2026 |

## Organisation du dépôt

| Dossier ou fichier | Contenu |
|---|---|
| `GUIDE_EQUIPE.md` | Le projet expliqué simplement, les documents, les rôles, la façon de travailler |
| `docs/P1_cahier_de_taches.md` | Les tâches de P1, dans l'ordre, avec le pourquoi et le comment |
| `docs/P1_explication_fichiers.md` | Le rôle de chaque fichier de `p1_data/` et la justification des choix |
| `docs/usage_ia.md` | Brouillon de l'annexe obligatoire sur l'usage de l'IA |
| `donnees/` | Les trois petits fichiers du pack : liste des patchs, métadonnées, stades |
| `notebooks/` | Préparation du fichier d'images et début type d'un notebook de travail |
| `p1_data/` | Audit des données, découpage sans fuite, normalisations de couleur, chargement |

Les images ne sont pas dans le dépôt : elles se trouvent dans un fichier parquet sur Google Drive (voir `GUIDE_EQUIPE.md`, section 4).

## Relancer le travail déjà fait

Depuis la racine du dépôt :

```bash
pip install -r requirements.txt
```

```bash
python p1_data/audit_metadonnees.py donnees p1_data/sorties
```

```bash
python p1_data/creer_split.py donnees p1_data/sorties
```

```bash
python -m pytest p1_data -q
```

## Sources

- CAMELYON17, licence CC0 : https://camelyon17.grand-challenge.org
- Bándi P. et al., From detection of individual metastases to classification of lymph node status at the patient level, IEEE TMI, 2019
- Koh P. W. et al., WILDS: A Benchmark of in-the-Wild Distribution Shifts, ICML 2021
- Code de préparation de WILDS : https://github.com/p-lambda/wilds
- Macenko M. et al., A method for normalizing histology slides for quantitative analysis, ISBI 2009
- Reinhard E. et al., Color transfer between images, IEEE CG&A, 2001
- Tellez D. et al., Whole-slide mitosis detection in H&E breast histology, IEEE TMI, 2018
