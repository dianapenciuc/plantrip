# Plantrip
L’objectif du projet est la création d’une application permettant de proposer un itinéraire optimisé selon certains critères (Durée du séjour, lieu de visite).
Le dossier data contient actuellement des échantillons de données, le dataset complet est disponible sur un stockage externe ainsi que les fichiers de données extraites et transformées.
Le fichier requirements.txt contient les dépendances pour l'environnement python; le contenu actuel est donné à titre d'exemple.

---------------------------------------------------------------------------------------------------------------------
Dans l'ordre chronologique de lancement pour un fonctionnement optimal:

1 - Task_Scheduler.bat:
  - Permet de télécharger les flux de données au format jsonls provenant de DATATourisme dans un dossier ./data/jsonld/
		=> 4 fichiers seront téléchargés correspondant aux différents types de POI définis sur DATATourisme (Lieux, Events, Tours et Produits)
  - Déplace les anciens fichiers téléchargés dans un dossier ./data/jsonld/previous_versions.
  - Il faut lancer le script en étant dans le dossier principal du projet. Ce fichier a été testé à partir d'un processus Windows Task Scheduler réalisé toutes les semaines.

2 - jsonld_extract:
  - Permet d'extraire les données brutes des fichiers jsonld et de les stocker dans un fichier csv par jsonld.
  	=> Données extraites pour chauqe POI : id, nom, type(s), theme(s), startdate, enddate, street, postalcode, city, insee, region, latitude, longitude, email, web, tel, lastupdate, comment
		=> 4 fichiers csv seront créés car 4 jsonld téléchargés à chaque update.
  - Prend automatiquement en entrée tous les fichiers se trouvant dans le dossier ./data/jsonld/
  
