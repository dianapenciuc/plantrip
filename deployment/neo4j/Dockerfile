FROM neo4j:5.20.0
ADD pluginsneo4j/ /plugins
ENV NEO4J_AUTH=neo4j/your_password
ENV NEO4J_PLUGINS='["graph-data-science"]'
ENV dbms.unmanaged_extension_classes=n10s.endpoint=/rdf
ENV dbms.security.allow_csv_import_from_file_urls=true
ENV NEO4J_server_memory_pagecache_size=8G
