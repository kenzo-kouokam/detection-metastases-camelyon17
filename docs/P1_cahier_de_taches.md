# Cahier de tâches de P1 : données et couleur

Dernière mise à jour : lundi 9 novembre 2026.

## Ce que P1 doit prouver

1. Le découpage est sain : aucune lame ni patiente n'est partagée entre entraînement et validation.
2. Les couleurs changent d'un hôpital à l'autre, et on sait de combien.
3. Une technique réduit cet écart, ou n'y parvient pas. Dans les deux cas, on sait le montrer et l'expliquer.

Cela porte directement le critère « Compréhension et gestion des données » (25 % de la note) et une partie du critère « Interprétation » (10 %).

## Vue d'ensemble

| N° | Tâche | Quand | Qui attend ce résultat | État |
|---|---|---|---|---|
| 1 | Fabriquer le fichier parquet et le partager sur Drive | Aujourd'hui | Toute l'équipe | Fabriqué et partagé avec 3 membres, reste le 4e |
| 2 | Comprendre le découpage et le transmettre | Aujourd'hui | P2, P3 | Fichier prêt, à transmettre |
| 3 | Essayer `preprocessing.py` sur de vraies images | Aujourd'hui ou mardi | P3 | Testé sur images synthétiques seulement |
| 4 | Décrire les données | Mardi 10 | Follow-up | Chiffres prêts, à mettre en forme |
| 5 | Montrer des patchs par hôpital et par classe | Mardi 10 | Follow-up | À faire |
| 6 | Mesurer les écarts de couleur | Mardi 10 | Follow-up, tâche 9 | À faire |
| 7 | Rédiger les trois constats d'exploration | Mercredi 11 | Follow-up | Deux pistes prêtes |
| 8 | Préparer les diapositives de P1 | Jeudi 12 | Follow-up | À faire |
| 9 | Comparer les techniques de couleur sur H1 | 14 au 18 nov. | P5, rapport | À faire |
| 10 | Confirmer la résolution par scanner avec P4 | 14 au 16 nov. | P4 | À faire |
| 11 | Aider P5 sur Grad-CAM et l'équité entre hôpitaux | 18 au 22 nov. | P5 | À faire |
| 12 | Rédiger la partie données du rapport | 19 au 23 nov. | Rapport | À faire |
| 13 | Nettoyer le code, préparer l'oral | 23 au 26 nov. | Dépôt, soutenance | À faire |

## Aujourd'hui, dans cet ordre

### Tâche 1 : fabriquer le fichier parquet

**Pourquoi d'abord :** sans images, ni P1, ni P2, ni P3 ne peuvent avancer.

**Comment :**
1. Personne dans l'équipe ne l'a encore fait : P1 s'en charge.
2. Ouvrir `notebooks/00_preparer_parquet_colab.ipynb` dans Colab et lancer Tout exécuter.
3. Partager le dossier `SPE1` avec le rôle Lecteur et envoyer le lien à l'équipe.

**Résultat :** un dossier `SPE1` partagé sur Drive, contenant un fichier vérifié de 39 000 patchs.

**État au 09/11 :** fichier fabriqué et vérifié sur Colab, en moins de 3 minutes. Il pèse 0,91 Go et se trouve dans `Mon Drive/SPE1`, partagé en Lecteur avec 3 membres. Il reste à ajouter le 4e membre et à envoyer le lien à l'équipe.

### Tâche 2 : comprendre et transmettre le découpage

**Pourquoi maintenant :** si P2 et P3 démarrent sur `id_val`, leurs scores seront gonflés et il faudra tout relancer.

**Comment :**
1. Lire `docs/P1_explication_fichiers.md`, sections 1 et 3.
2. Ouvrir `p1_data/sorties/audit_lames.csv` dans un tableur et repérer une lame ITC et une lame macro.
3. Envoyer à P2 et P3 : « Utilisez la colonne `split_equipe`, voir `GUIDE_EQUIPE.md`, section 5. »

**Résultat :** tout le monde entraîne et valide sur les mêmes patchs.

### Tâche 3 : essayer `preprocessing.py` sur de vraies images

**Pourquoi maintenant :** P3 doit pouvoir l'utiliser sans surprise. Le module n'a jamais vu de vraies images ni tourné avec torchvision.

**Comment, sur Colab :**
1. Cloner le dépôt et charger le parquet.
2. Ajuster `Reinhard` et `Macenko` sur une centaine de patchs de H0, H3 et H4.
3. Afficher des patchs de chaque hôpital avant et après.
4. Appeler `get_transforms` pour chaque mode.
5. Passer `Macenko` sur 1 000 patchs et noter `n_echecs` ainsi que le temps de calcul.

**Résultat :** une figure avant/après, un taux d'échec, un temps par patch. En cas de problème, on corrige avant de transmettre.

## D'ici au follow-up du vendredi 13

Le sujet attend pour ce jour-là : l'échantillon et le découpage, des exemples d'images par hôpital, trois constats d'exploration. Ces trois points sont pour P1. Le mercredi 11 novembre étant férié, mieux vaut fixer les disponibilités de l'équipe dès maintenant.

### Tâche 4 : décrire les données

**Pourquoi :** le sujet demande la répartition par hôpital, par lame et par classe.

**Comment :** mettre en forme les tableaux de `audit_lames.csv` : nombre de patchs par jeu et par hôpital, nombre de lames par classe, part de la plus grosse lame par hôpital.

### Tâche 5 : montrer des patchs

**Pourquoi :** c'est explicitement demandé, et c'est ce qui rend le problème visible à l'oral.

**Comment :** reprendre la section 3 du notebook de démarrage. Une ligne par hôpital, des patchs normaux puis tumoraux. Garder une seule figure, la plus lisible.

### Tâche 6 : mesurer les écarts de couleur

**Pourquoi :** un histogramme montre que les couleurs diffèrent. Un chiffre permet de dire de combien et de comparer avant/après normalisation.

**Comment :**
1. Prendre environ 500 patchs par hôpital et par classe.
2. Calculer les moyennes RGB, HSV et Lab, et tracer les distributions par hôpital.
3. Utiliser `stats_lab` et `distances_lab` pour obtenir un tableau des écarts entre hôpitaux.
4. Vérifier si l'écart de couleur entre hôpitaux est plus grand que l'écart entre normal et tumoral. Si oui, cela explique pourquoi un modèle peut se laisser tromper.

**Résultat :** une figure et un tableau, avec une phrase d'interprétation pour chacun.

### Tâche 7 : les trois constats

Il faut trois constats appuyés sur des chiffres. Le troisième dépend de la tâche 6.

1. **Le 50/50 cache peu de diversité.** Une seule lame fournit 78 % des patchs tumoraux de H4, 75 % pour H3, 64 % pour H2. Le modèle voit donc peu de tumeurs différentes.
2. **Les petites métastases sont presque absentes.** Les lames macro représentent 97 % des patchs tumoraux, les ITC 0,2 %. Un bon rappel par patch ne garantit donc rien sur les petites lésions, qui sont les plus difficiles.
3. **Les couleurs diffèrent entre hôpitaux.** À chiffrer avec la tâche 6.

### Tâche 8 : les diapositives

**Contenu :** une diapositive sur l'échantillon et le découpage (pourquoi pas `id_val`), une sur les exemples et la couleur, une sur les trois constats. Se préparer à répondre à la question : pourquoi découper par patiente ?

## Après le follow-up

### Tâche 9 : comparer les techniques de couleur

**Pourquoi :** le sujet impose d'en tester au moins une. C'est le cœur du rôle de P1.

**Comment, avec P3 :**
1. Partir de la même recette d'entraînement et ne changer que le mode de `get_transforms`.
2. Tester `aucun`, `colorjitter`, `hed`, `reinhard`, `macenko`, `gris`. Avec peu de GPU, commencer par `aucun`, `colorjitter` et `macenko`.
3. Répéter avec au moins trois graines aléatoires, sinon un écart peut n'être que du hasard.
4. Comparer sur `val_interne` et `val_ood` : ROC-AUC et rappel, par hôpital.
5. Retenir une technique pour le modèle final. Ne jamais comparer sur H2.

**Résultat :** un tableau avec moyenne et écart entre graines, et un choix justifié. Si aucune technique n'aide nettement, on le dit : c'est un résultat valable.

### Tâche 10 : confirmer la résolution

**Pourquoi :** P4 convertit des cellules en millimètres. L'erreur sur le seuil change la classe des lames, en particulier ITC contre micro.

**Comment :** chercher la résolution des scanners dans l'article de Bándi et al. (2019) et dans la documentation WILDS, puis remplacer la valeur 0,24 µm par pixel si besoin.

### Tâche 11 : aider P5

**Comment :** regarder avec P5 des patchs bien et mal classés dans plusieurs hôpitaux. Le modèle regarde-t-il les cellules ou la teinte générale ? Comparer les performances entre hôpitaux pour la partie équité.

### Tâche 12 : la partie données du rapport

**Contenu, 3 à 4 pages sur les 15 :** description des données, limites (lames toutes positives, jeu équilibré, concentration sur quelques lames), découpage et sa justification, couleur et résultats de la tâche 9.

### Tâche 13 : finir proprement

Vérifier que le dépôt se relance depuis zéro, compléter `docs/usage_ia.md`, répéter ses 2 minutes d'oral, et revoir les parties des autres : les questions du jury peuvent viser n'importe qui.

## Questions à poser

**À l'équipe :**
- Qui a déjà lancé un entraînement, et sur quel découpage ?
- Où en sont réellement les phases 1 à 3 de la feuille de route ?
- Qui est P2, P3, P4, P5 ?

**Au professeur :**
- Pour classer les lames, peut-on prédire sur tous les patchs de WILDS (455 954) plutôt que sur les 39 000 ? Le sous-échantillon laisse certaines lames ITC avec 0 ou 1 patch tumoral.
- Sur H2, faire au même moment l'évaluation des patchs et celle des lames respecte-t-il bien la règle du test unique ?
