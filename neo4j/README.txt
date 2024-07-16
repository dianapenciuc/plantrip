Image docker :
https://hub.docker.com/_/neo4j

Documentation:
https://neo4j.com/docs/operations-manual/current/docker/introduction/?utm_source=Google&utm_medium=PaidSearch&utm_campaign=Evergreen&utm_content=EMEA-Search-SEMCE-DSA-None-SEM-SEM-NonABM&utm_term=&utm_adgroup=DSA&gad_source=1

https://github.com/neo4j-labs/neosemantics/releases

Documentation import RDF
https://neo4j.com/labs/neosemantics/4.1/import/

Installation:
https://github.com/neo4j-labs/neosemantics/blob/4.0/README.md

Première commande à lancer lors du lancement du neo4j:
CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE

call n10s.graphconfig.init()
CALL n10s.rdf.import.fetch("file:///data/na_19_06.rdf", "RDF/XML");

Important! the csv file needs to be placed in the import directory, otherwise it will not load

docker run -d \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/your_password \
    --env NEO4J_PLUGINS='["graph-data-science"]' \
    --volume /media/diana/DATA/2024/datatourisme/region_idf:/data \
    --volume /media/diana/DATA/2024/pluginsneo4j:/plugins \
    --env dbms.unmanaged_extension_classes=n10s.endpoint=/rdf \
    --env dbms.security.allow_csv_import_from_file_urls=true\
    --env NEO4J_server_memory_pagecache_size=8G \
    neo4j:5.20.


