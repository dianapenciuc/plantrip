# Plantrip
L’objectif du projet est la création d’une application permettant de proposer un itinéraire optimisé selon certains critères (Durée du séjour, lieu de visite).
Le dossier data contient actuellement des échantillons de données, le dataset complet est disponible sur un stockage externe ainsi que les fichiers de données extraites et transformées.
Le fichier requirements.txt contient les dépendances pour l'environnement python; le contenu actuel est donné à titre d'exemple.

---------------------------------------------------------------------------------------------------------------------
Dans l'ordre chronologique de lancement pour un fonctionnement optimal:
  
## 1 - Task_Scheduler.bat:
  - Permet de télécharger les flux de données au format jsonls provenant de DATATourisme dans un dossier ./data/jsonld/
		=> 4 fichiers seront téléchargés correspondant aux différents types de POI définis sur DATATourisme (Lieux, Events, Tours et Produits)
  - Déplace les anciens fichiers téléchargés dans un dossier ./data/jsonld/previous_versions.
  - Il faut lancer le script en étant dans le dossier principal du projet. Ce fichier a été testé à partir d'un processus Windows Task Scheduler réalisé toutes les semaines.
  

## 2 - Extract_jsonld.py:
  - Permet d'extraire les données brutes des fichiers jsonld et de les stocker dans un fichier csv par jsonld.
    
  	=> Données extraites pour chauqe POI : id, nom, type(s), theme(s), startdate, enddate, street, postalcode, city, insee, latitude, longitude, email, web, tel, lastupdate, comment

	=> Nettoie les données type et thèmes

  	=> 4 fichiers csv seront créés car 4 jsonld téléchargés à chaque update.
  - Prend automatiquement en entrée tous les fichiers se trouvant dans le dossier ./data/jsonld/
  - Ajout d'une colonne région
  
  
## 3 - Transform_csv.py:
  - Dans les colonnes "postalcode" et "latitude", certaines données sont sous une forme particulière ("data:xxxxxxx"). Cette forme est en fait une partie d'url ("https://data.datatourisme.fr/xxxxxxx) qui permet de récupérer les données d'intérêts.
    
 	  Exemple 1: https://data.datatourisme.fr/6de52089-2ef6-310c-a4ef-ae310b273cc7 nous amène sur une page permettant de récupérer _via_ du webscrapping les données d'adresse
    
	  Exemple 2: https://data.datatourisme.fr/26871bb2-1294-3c82-9c39-99b16e4247d8 nous amène sur une page permettant de récupérer _via_ du webscrapping les données de géolocalisation

- Rempli la colonne région en se basant sur le code postal
