#On crée les dossiers nécessaires:
mkdir .\\data\\jsonld\\
mkdir .\\data\\csv\\raw_extract\\
mkdir .\\data\\csv\\transformed\\

# On déplace les fichiers anciennement téléchargés
move .\\data\\jsonld\\*.jsonld .\\data\\jsonld\\previous_versions

# On récupère la date qui nous servira à renommer nos fichiers
set mydate=%date:~6,4%%date:~3,2%%date:~0,2%
# On récupère nos flux compréssés qu'on va décomprésser et sauvegarder dans des fichiers spécifiques portant la date pour le versioning
curl --compressed https://diffuseur.datatourisme.fr/webservice/2d170ab3322c5d5ade969c185d7aa1f7/2197898d-3506-444f-bc6d-6789d6cf0888 --output .\\data\\jsonld\\lieux-%mydate%.jsonld
curl --compressed https://diffuseur.datatourisme.fr/webservice/1db5576a244ab8adff34886e9c746172/2197898d-3506-444f-bc6d-6789d6cf0888 --output .\\data\\jsonld\\produits-%mydate%.jsonld
curl --compressed https://diffuseur.datatourisme.fr/webservice/e647c3e07526fb416f16ee3fd25ce334/2197898d-3506-444f-bc6d-6789d6cf0888 --output .\\data\\jsonld\\tours-%mydate%.jsonld
curl --compressed https://diffuseur.datatourisme.fr/webservice/5dc3fe2b1b00a009b9b2e34a85750bf4/2197898d-3506-444f-bc6d-6789d6cf0888 --output .\\data\\jsonld\\events-%mydate%.jsonld