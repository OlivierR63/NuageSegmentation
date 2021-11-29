# Projet NuageSegmentation 

Différents scripts et notebooks pour le projet NuageSegmentation:

Attention, pour éviter les problèmes de compatibilité, ces fichiers doivent être exécuté en utilisant la version 2.3.1 de Keras et la version 2.0 de Tensorflow

* 2 scripts python pour généraliser les fonctions utiles à tous les notebooks: 

  * clouds_graph_functions: script python avec toutes les fonctions utiles à l'affichage
  * clouds_utilities_functions: script python avec toutes les fonctions utiles non liées à l'affichage
  
* 6 notebooks:
ces 6 notebooks nécessitent l'ajout des 2 scripts ci dessus en utility script et l'ajout des données d'input du projet Kaggle 
https://www.kaggle.com/c/understanding_cloud_organization/ (soit 2 répertoires d'images test_images, train_images et un fichier csv train.csv)
  
    1. Partie analyse des données et data visualisation:
          * clouds-init-analyse-data : notebook initial de prise en main et analyse des données + dataviz
          
    2. Partie classification multilabels:
          * clouds-classification : notebook de classification multilabels (différents modèles peuvent être testés)
          
    3. Partie segmentation
          * clouds-segmentation-unet : notebook de segmentation avec le modèle U-Net
          * cloud-segmentation-fpn : notebook de segmentation avec le modèle FPN
          * cloud-segmentation-segnet : notebook de segmentation avec le modèle SegNet
          
    4. Partie détection objet:
          * understanding-clouds-eda-mask-rcnn-v2 : notebook de détection d'objets avec le modèle mask R-CNN
    
