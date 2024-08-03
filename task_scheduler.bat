### Etapes de préparation
# On crée les dossiers nécessaires:
mkdir .\\data\\jsonld\\previous_versions\\
mkdir .\\data\\csv\\raw_extract\\previous_versions\\
mkdir .\\data\\csv\\transformed\\previous_versions\\

# On supprime les anciennes versions téléchargées
del /Q .\\data\\jsonld\\previous_versions\\*

# On déplace les fichiers anciennement téléchargés
move .\\data\\jsonld\\*.jsonld .\\data\\jsonld\\previous_versions

# On déplace les fichiers d'extraction s'ils existent déjà
move .\\data\\csv\\raw_extract\\*.csv .\\data\\csv\\raw_extract\\previous_versions
move .\\data\\csv\\transformed\\*.csv .\\data\\csv\\transformed\\previous_versions

# On crée une sauvegarde de notre base de données présente dans le container itineraire/mysql-1
docker exec mysql-1 mysqldump --default-character-set=utf8mb4 -uroot -prootadmin itineraire -r /projet/data/mysql_db/save_db.sql

### Etapes de lancement / actualisation
# On récupère la date qui nous servira à renommer nos fichiers
set mydate=%date:~6,4%%date:~3,2%%date:~0,2%

# On récupère nos flux compréssés qu'on va décomprésser et sauvegarder dans des fichiers spécifiques portant la date pour le versioning
curl --compressed https://diffuseur.datatourisme.fr/webservice/c229282874eb94445dac9450649a6130/2197898d-3506-444f-bc6d-6789d6cf0888 --output .\\data\\jsonld\\lieux_1-%mydate%.jsonld
curl --compressed https://diffuseur.datatourisme.fr/webservice/6ec7691f90a10a04e04732b250d35575/2197898d-3506-444f-bc6d-6789d6cf0888 --output .\\data\\jsonld\\lieux_2-%mydate%.jsonld
curl --compressed https://diffuseur.datatourisme.fr/webservice/8435363fa282ec0c8bbf851335536325/2197898d-3506-444f-bc6d-6789d6cf0888 --output .\\data\\jsonld\\lieux_3-%mydate%.jsonld
curl --compressed https://diffuseur.datatourisme.fr/webservice/340629804e573eb5fd00491062f7b2d6/2197898d-3506-444f-bc6d-6789d6cf0888 --output .\\data\\jsonld\\lieux_4-%mydate%.jsonld
curl --compressed https://diffuseur.datatourisme.fr/webservice/1db5576a244ab8adff34886e9c746172/2197898d-3506-444f-bc6d-6789d6cf0888 --output .\\data\\jsonld\\produits-%mydate%.jsonld
curl --compressed https://diffuseur.datatourisme.fr/webservice/e647c3e07526fb416f16ee3fd25ce334/2197898d-3506-444f-bc6d-6789d6cf0888 --output .\\data\\jsonld\\tours-%mydate%.jsonld
curl --compressed https://diffuseur.datatourisme.fr/webservice/5dc3fe2b1b00a009b9b2e34a85750bf4/2197898d-3506-444f-bc6d-6789d6cf0888 --output .\\data\\jsonld\\events-%mydate%.jsonld

# On lance le docker; cette étape, si déjà lancée une 1ère fois, relancera le container etl-1 pour l'extraction des nouvelles données, la création des fichiers csv et l'alimentation de la base de données mysql
docker-compose up -d