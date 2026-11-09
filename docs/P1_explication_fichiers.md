# Les fichiers de `p1_data/` expliqués

## En résumé

| Fichier | Rôle | Question à laquelle il répond |
|---|---|---|
| `audit_metadonnees.py` | Analyse les métadonnées, sans image | Que valent vraiment nos données ? |
| `sorties/audit_lames.csv` | Une ligne par lame, résultat de l'audit | Combien de patchs, quelle classe, pour chaque lame ? |
| `sorties/audit_oracle.csv` | Résultat du test avec les vrais labels | Notre méthode de mesure en millimètres tient-elle ? |
| `preprocessing.py` | Le module que P1 livre à l'équipe | Comment découper sans fuite et traiter la couleur ? |
| `creer_split.py` | Fabrique le découpage de l'équipe | Quel patch va dans quel jeu ? |
| `sorties/splits_par_patch.csv` | Le découpage, une ligne par patch | Le fichier à utiliser par P2 et P3 |
| `test_preprocessing.py` | Vérifie que le module fait ce qu'il annonce | Le code est-il fiable ? |

## 1. `audit_metadonnees.py`

**Pourquoi il existe.** Avant de construire sur des données, il faut vérifier ce qu'elles contiennent. Ce script n'a besoin d'aucune image, il tourne sur un ordinateur portable en quelques secondes.

**Ce qu'il a montré.**

| Constat | Chiffre | Ce que cela change |
|---|---|---|
| Une seule lame fournit l'essentiel des patchs tumoraux d'un hôpital | 78 % pour H4, 75 % pour H3, 64 % pour H2 | Le 50/50 cache une faible diversité |
| Les lames macro dominent | 97 % des patchs tumoraux, 0,2 % pour les ITC | Le rappel par patch dit peu sur les petites lésions |
| Les lames ITC ont très peu de patchs tumoraux dans le sous-échantillon | 0 à 9, et 3 lames de H3 n'en ont aucun | Le sous-échantillon ne suffit pas pour classer les lames |

**Le test « oracle ».** On remplace le modèle par les vrais labels et on déroule le reste : regrouper les cellules tumorales voisines, mesurer la plus grande zone, en déduire une classe, comparer à `stages.csv`. C'est le meilleur résultat possible avec cette méthode, puisqu'il n'y a aucune erreur de modèle.

Résultat : 44 lames sur 50 correctes, soit 88 %. Les 17 micro et les 17 macro sont toutes justes. Les 6 erreurs sont des ITC classées micro, sur 16 ITC.

**Le piège de la résolution.** Le code WILDS place les patchs sur une grille de 32 pixels au niveau 2 de la lame. Chaque patch fait 96 pixels, mais son label ne dépend que de ses 32 pixels centraux. Une cellule de la grille vaut donc 32 pixels, soit environ 31 micromètres. En comptant 96 pixels par cellule, la concordance tombe à 50 %.

**Attention.** La valeur de 0,24 µm par pixel est un ordre de grandeur à confirmer (tâche 10). L'audit lit aussi les labels de H2, pour décrire les données. Aucun réglage ne doit s'appuyer dessus.

**Si le jury demande pourquoi on a fait cela :** pour savoir si une erreur future vient du modèle ou de la méthode. Une erreur qui existe déjà dans l'oracle ne vient pas du modèle.

## 2. `sorties/audit_lames.csv` et `sorties/audit_oracle.csv`

Ce sont les résultats écrits par le script. On peut les ouvrir dans un tableur.

- Dans `audit_lames.csv`, `n` est le nombre de patchs de la lame, `tum` le nombre de patchs tumoraux, `n_sub` et `tum_sub` les mêmes chiffres dans le sous-échantillon, `classe` la classe officielle.
- Dans `audit_oracle.csv`, `n_amas` est le nombre de zones tumorales séparées, `axe_max_mm` la plus grande longueur mesurée, `oracle` la classe déduite.

## 3. `creer_split.py` et `sorties/splits_par_patch.csv`

**Le problème.** Des patchs d'une même lame se ressemblent beaucoup. Si certains servent à l'entraînement et d'autres à la validation, le modèle semble excellent alors qu'il reconnaît la lame. C'est une fuite de données. Or `id_val` partage ses 30 lames avec `train`. Le notebook de démarrage le signale d'ailleurs lui-même.

**La solution.** On réserve des patientes entières de H0, H3 et H4 pour une validation interne.

| Choix | Justification | Limite |
|---|---|---|
| Découper par patiente et non par lame | Deux ganglions d'une même patiente partagent la biologie de sa tumeur. Le sujet accepte les deux, celui-ci est le plus strict | Moins de combinaisons possibles |
| Réserver une patiente par classe et par hôpital | Mesurer le modèle sur des ITC, micro et macro dans chaque hôpital | Peu de lames par classe : résultats à lire comme des indications |
| Ne pas réserver la lame dominante d'un hôpital | Sinon l'entraînement perdrait la majorité des patchs tumoraux de cet hôpital | La validation ne contient pas les plus grosses tumeurs |
| Exclure les lames sans patch tumoral | Elles ne permettraient pas de calculer un rappel | Certaines lames ITC de H3 ne sont jamais en validation |
| Fixer la graine à 42 | Tout le monde obtient exactement le même découpage | Un seul tirage : à faire varier pour tester la stabilité |

**Résultat.** 10 lames réservées, et non 9, car la patiente 17 a trois lames qui partent ensemble.

| Jeu | Patchs | Tumoral |
|---|---|---|
| `train_interne` | 19 483 | 10 518 |
| `val_interne` | 7 517 | 2 982 |
| `val_ood` (H1) | 4 000 | 2 000 |
| `test` (H2) | 8 000 | 4 000 |

**Le prix à payer.** L'entraînement passe de 24 000 à 19 483 patchs, et n'est plus exactement à 50/50 : sa part tumorale atteint 57 % pour H0.

## 4. `preprocessing.py`

C'est le livrable que la feuille de route attend de P1. Il contient trois familles d'outils.

### Les contrôles de découpage

- `faire_split_interne` applique les choix décrits plus haut.
- `controler_fuite` lève une erreur si une lame ou une patiente se retrouve dans deux jeux, ou si H1 ou H2 entre dans l'entraînement. Sur le découpage WILDS d'origine, il signale bien les 30 lames partagées.

### Les techniques de couleur

| Technique | Principe | Avantage | Limite |
|---|---|---|---|
| `aucun` | Rien | Point de comparaison indispensable | Sensible aux couleurs |
| `colorjitter` | Modifie au hasard luminosité, contraste, saturation, teinte | Simple, fourni par torchvision | Modifications sans lien avec la chimie de la coloration |
| `hed` | Modifie au hasard la quantité de chaque colorant | Imite une vraie variation de coloration | À régler : trop fort, l'image devient irréaliste |
| `reinhard` | Aligne la moyenne et l'écart des couleurs sur une référence | Rapide | Dépend de la part de tissu et de blanc dans le patch |
| `macenko` | Sépare les deux colorants, puis recolore vers une référence | Suit la physique de la coloration | Plus lent, instable sur un petit patch peu coloré |
| `gris` | Retire la couleur | Empêche d'apprendre la teinte | Perd une information peut-être utile |

Les colorants sont l'hématoxyline, qui marque les noyaux en violet, et l'éosine, qui marque le reste en rose.

**Normalisation ou augmentation.** La normalisation rend les images semblables, à l'entraînement comme à l'évaluation. L'augmentation montre au modèle des couleurs variées pendant l'entraînement seulement, pour qu'il apprenne à ne pas s'y fier.

**Sécurité.** `Reinhard.fit` et `Macenko.fit` refusent une référence issue de H1 ou H2. Apprendre la référence sur ces hôpitaux ferait entrer dans le prétraitement une information sur les données censées rester inconnues.

**Patchs à problème.** Si `Macenko` ne trouve pas assez de tissu, il rend le patch intact et l'ajoute au compteur `n_echecs`, au lieu de planter ou de produire une image fausse. Ce taux est à rapporter.

### L'interface pour P3

`get_transforms(mode, reference)` renvoie les transformations d'entraînement et d'évaluation. P3 change un seul mot pour tester une variante, ce qui garantit que seule la couleur diffère d'une expérience à l'autre.

### La mesure de l'écart

`stats_lab` et `distances_lab` calculent l'écart de couleur entre hôpitaux. L'espace Lab est utilisé parce qu'une distance y correspond à peu près à une différence perçue, ce qui n'est pas le cas en RGB.

## 5. `test_preprocessing.py`

**Pourquoi.** P3 et P4 s'appuieront sur ce module. Une erreur silencieuse fausserait tous les résultats.

**Ce qui est vérifié, en 22 tests :** les conversions de couleur sont justes, une normalisation rapproche bien deux images décalées, un patch blanc ne fait rien planter, une référence interdite est refusée, le contrôle détecte chaque type de fuite, le découpage est reproductible.

**Ce qui n'est pas vérifié.** Les tests utilisent des images fabriquées, et `get_transforms` n'a jamais tourné. Rien ne dit encore que ces techniques améliorent le modèle sur les vraies images. C'est l'objet des tâches 3 et 9.

**Pour les lancer :**

```bash
python -m pytest p1_data -q
```

## 6. Ce qu'il faut savoir expliquer seul

1. Pourquoi ne pas découper les patchs au hasard ?
2. Pourquoi ne pas utiliser `id_val` pour juger le modèle ?
3. Pourquoi H2 ne sert-il qu'une fois ?
4. Pourquoi un patch de 96 pixels compte-t-il pour 32 pixels sur la carte ?
5. Pourquoi une accuracy de 90 % sur ce jeu ne dit-elle pas grand-chose en clinique ?
6. Que fait Macenko, et en quoi diffère-t-il d'une simple augmentation ?
7. Pourquoi apprendre la référence de couleur sur H0, H3 et H4 uniquement ?
8. Que nous apprend le test oracle ?
