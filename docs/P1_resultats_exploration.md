# Résultats de l'exploration (P1)

Notebook : `notebooks/p1_exploration_couleur.ipynb`, exécuté sur Colab le 13 novembre 2026, sur les vraies images du sous-échantillon. Figures et tableaux : `Mon Drive/SPE1_resultats_P1` (compte de P1).

## 1. Les données (tâche 4)

| Jeu | H0 normal | H0 tumoral | H1 | H2 | H3 normal | H3 tumoral | H4 normal | H4 tumoral |
|---|---|---|---|---|---|---|---|---|
| train_interne | 2 365 | 3 085 | 0 | 0 | 3 528 | 3 575 | 3 072 | 3 858 |
| val_interne | 2 135 | 1 415 | 0 | 0 | 972 | 925 | 1 428 | 642 |
| val_ood | 0 | 0 | 2 000 / 2 000 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 4 000 / 4 000 | 0 | 0 | 0 | 0 |

Part des patchs tumoraux qui viennent de la plus grosse lame de l'hôpital :

| Hôpital | Lame | Classe | Part |
|---|---|---|---|
| H0 | 3 | macro | 42 % |
| H1 | 13 | macro | 49 % |
| H2 | 28 | macro | 64 % |
| H3 | 38 | macro | 75 % |
| H4 | 48 | macro | 78 % |

## 2. Les patchs par hôpital (tâche 5)

Figure `patchs_par_hopital.png` : quatre patchs normaux et quatre tumoraux par hôpital. À l'œil, H2 est nettement plus bleu-violet et plus saturé, et H4 plus pâle.

## 3. Les écarts de couleur (tâche 6)

Échantillon : 500 patchs par hôpital et par classe, couleur moyenne de chaque patch.

**Couleur médiane par hôpital**

| Hôpital | L* (clarté) | a* (vers le rouge) | b* (négatif = vers le bleu) | Saturation |
|---|---|---|---|---|
| H0 | 66,9 | 17,3 | environ -9 | 51,6 |
| H1 | 51,7 | 21,6 | environ -12 | 79,5 |
| H2 | 57,9 | 33,8 | environ -29 | 101,1 |
| H3 | 57,0 | 26,3 | environ -10 | 87,1 |
| H4 | 75,8 | 16,7 | environ -13 | 45,1 |

Les valeurs de b* sont lues sur la figure `couleur_par_hopital.png`. Les autres viennent de `couleur_mediane_par_hopital.csv`.

**Écart de couleur Delta-E entre hôpitaux** (environ 2 à 3 est à peine visible)

| | H0 | H1 | H2 | H3 | H4 |
|---|---|---|---|---|---|
| H0 | 0 | 13,8 | 24,7 | 12,4 | 9,7 |
| H1 | 13,8 | 0 | 18,4 | 5,9 | 19,8 |
| H2 | 24,7 | 18,4 | 0 | 17,6 | 22,8 |
| H3 | 12,4 | 5,9 | 17,6 | 0 | 17,8 |
| H4 | 9,7 | 19,8 | 22,8 | 17,8 | 0 |

Écart moyen entre hôpitaux : 16,0. Le plus faible : H1 et H3 (5,9). Le plus fort : H0 et H2 (24,7).

**La couleur moyenne trahit-elle l'hôpital ?**

Une régression logistique ne reçoit que trois nombres par patch (L*, a* et b* moyens). Les lames testées ne servent jamais à son entraînement.

| Question | Résultat | Hasard |
|---|---|---|
| Reconnaître l'hôpital (5 choix) | accuracy 51,7 % | 20 % |
| Reconnaître tumoral ou non (H2 exclu) | ROC-AUC 0,63 | 0,50 |

## 4. Normalisation sur vraies images (tâche 3)

Références apprises sur 100 patchs par hôpital de `train_interne` (H0, H3, H4). Test sur 1 000 patchs (100 par hôpital et par classe).

| Technique | Delta-E moyen entre hôpitaux | Delta-E max | Temps par patch | Échecs |
|---|---|---|---|---|
| Origine | 16,0 | 24,9 | | |
| Reinhard | 0,07 | 0,12 | 7,7 ms | 0 |
| Macenko | 3,4 | 5,4 | 5,5 ms | 0 |

**Attention à la ligne Reinhard.** Reinhard impose à chaque patch la moyenne de couleur de la référence. Un écart presque nul sur les moyennes est donc garanti par construction : ce tableau ne permet pas de juger Reinhard. Il ne sert qu'à vérifier que le code fait ce qu'il annonce.

**Ce que montre la figure `normalisation_avant_apres.png` :**
- Reinhard teinte aussi le fond blanc en gris rosé (patchs de H1 et H3 avec du vide). C'est sa limite connue : il traite le tissu et le fond de la même façon.
- Macenko garde le fond blanc et rapproche les teintes, mais rend certains patchs plus magenta que l'original.
- Macenko n'a échoué sur aucun des 1 000 patchs. À 5,5 ms par patch, il ajoute environ 2 minutes par époque de 24 000 patchs, sur un seul processus.

**Ce que ces résultats ne disent pas encore :** si une normalisation améliore le modèle. Cela se mesure avec P3 sur `val_interne` et `val_ood` (tâche 9).

## 5. Les trois constats pour le follow-up

1. **Le 50/50 cache peu de diversité.** Une seule lame fournit 78 % des patchs tumoraux de H4, 75 % pour H3, 64 % pour H2. Le modèle voit peu de tumeurs différentes.
2. **Les petites métastases sont presque absentes.** 97 % des patchs tumoraux viennent de lames macro, 0,2 % de lames ITC. Un bon rappel par patch ne dit presque rien sur les petites lésions.
3. **La couleur trahit l'hôpital plus que la tumeur.** Avec seulement trois nombres de couleur, on reconnaît l'hôpital une fois sur deux (hasard : une sur cinq), alors que tumoral et normal se séparent mal (AUC 0,63). Un modèle peut donc s'appuyer sur la teinte de l'hôpital. H2, l'hôpital de test, est le plus éloigné de tous (Delta-E de 17,6 à 24,7).

## 6. Une conséquence pour toute l'équipe

L'hôpital de validation H1 est proche de H3 (Delta-E 5,9), qui sert à l'entraînement. H2 est loin de tous les hôpitaux. Un modèle réglé sur H1 risque donc de paraître plus robuste qu'il ne le sera sur H2. Il faut s'attendre à une baisse sur le test final, et le dire dans le rapport plutôt que d'en être surpris.

## 7. Ce qui reste à vérifier

- L'écart de couleur entre normal et tumoral à l'intérieur d'un hôpital, et l'essai des 6 modes de `get_transforms`, ont été affichés dans le notebook mais pas enregistrés. Leur sortie est à relire dans Colab.
- La moyenne d'un patch mêle tissu et fond clair. Une mesure sur le tissu seul serait plus précise.
