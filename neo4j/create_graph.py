from neo4j import GraphDatabase
from pprint import pprint
from extract_distance import get_ors_matrix
import pandas as pd
import os

URL = 'bolt://0.0.0.0:7687'
AUTH = ('neo4j', 'your_password')
#AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))


driver = GraphDatabase.driver(os.getenv("NEO4J_URI"),
                              auth=AUTH)

def get_connection():
    return GraphDatabase.driver(URL, auth=AUTH)

def close_connection():
    driver.close()

def clean_database():
    # deleting data
    query = ['''
    MATCH (n) 
    DETACH DELETE n;
    '''
    ]
    with driver.session() as session:
        session.run(query)

def create_index():
    query = '''
    CREATE INDEX index_category FOR (p:POI) ON (p.category);
    CREATE INDEX index_distance FOR ()-[r:ONFOOT]-() ON (r.distance)
    '''
    
def load_pois_from_csv(filename,region_name='idf'):
    # loading POI data
    query = ''' 
    LOAD CSV WITH HEADERS FROM \'''' + filename + '''\' AS row
    MERGE (:POI {
        name: row.Nom_du_POI,
        id: row.URI_ID_du_POI,
        city: row.Commune,
        postalCode: row.Code_postal,
        category: row.Categories_de_POI,
        coordinates:[toFloat(row.Latitude),toFloat(row.Longitude)]
        });
    '''
    with driver.session() as session:
        session.run(query)
    
    graph_name = 'pois'.join(region_name)
    query_check_projection = '''
    CALL gds.graph.exists(\'''' + graph_name + '''\') YIELD
    graphName,
    exists
    return exists
    '''
    with driver.session() as session:
        exists = session.run(query_check_projection)
    #create the graph projection
    if not exists:
        query_create_projection = '''
        CALL gds.graph.project(
        \'''' + graph_name + '''\',
        {
        POI: {
            properties: 'coordinates'
        }
        },
        '*'
        );
        '''
    with driver.session() as session:
        session.run(query_create_projection)
        
def load_cities_from_csv(filename):
    # loading cities data
    query = ''' 
    LOAD CSV WITH HEADERS FROM \'''' + filename + '''\' AS row
    MERGE (:City {
        name: row.label,
        cityCode: row.city_code,
        id: row.insee_code,
        zipCode: row.zip_code,
        regionName: row.region_name,
        departementId: row.department_number,
        departementName: row.department_name,
        coordinates:[toFloat(row.latitude),toFloat(row.longitude)]
        });
    '''
    with driver.session() as session:
        session.run(query)   

def load_ranking(filename):
    #On récupère le ranking à partir d'une source externe
    query = ''' 
    LOAD CSV WITH HEADERS FROM \'''' + filename + '''\' AS row
    MATCH (c:City{cityCode: row.cityCode})
    SET c.rank = row.rank
    '''
    with driver.session() as session:
        session.run(query)
    
def get_poi_by_id(ids):
    query = '''
     MATCH (p:POI)
     WHERE p.id in ''' + str(ids) + '''
     RETURN p
    '''
    print(query)
    with driver.session() as session:
        result = session.run(query)
    return result

#Je donne en paramètre les coordonnées des villes le plus populaires (avec le plus de POIs)
#Params: 
#coordinates - [[48.86287791,2.3599986], [48.802292474,2.117410101],[48.885771198,2.793406447]]
def get_clusters(coordinates):
    #Le clustering avec centroids sur les coordonnées données
    query = '''
    CALL{
    CALL gds.kmeans.stream('poisidf', {
    nodeProperty: 'coordinates',
    k: 3,
    seedCentroids: ''' + coordinates + '''
    })
    YIELD nodeId, communityId
    RETURN gds.util.asNode(nodeId) AS node, communityId
    ORDER BY communityId
    } with node,communityId where communityId = 0 AND node.category CONTAINS 'placeofinterest' return node
    '''
    with driver.session() as session:
        session.run(query)

def get_geo_distance(coordinates1,coordinates2):
    query = '''
    WITH
    point({latitude: ''' + coordinates1[0] + ''', longitude: ''' + coordinates1[1] + ''', height: 100}) AS p1,
    point({latitude: ''' + coordinates2[0] + ''', longitude: ''' + coordinates2[1] + ''', height: 100}) AS p2
    RETURN round(point.distance(p1, p2)) AS dist
    '''
    with driver.session() as session:
            result = session.run(query)
    return result[0]["_fields"]

def set_geo_distance(ids):
    if len(ids) == 0:
        return
    query = '''
     MATCH (p:POI)
     WHERE p.id in ''' + str(ids) + '''
     RETURN p.coordinates as item
    '''
    
    records, summary,_keys = driver.execute_query(query)
    print("Query `{query}` returned {records_count} records in {time} ms.".format(
    query=summary.query, records_count=len(records),
    time=summary.result_available_after
    ))
    
#On calcule les distances entre les POIs donnés en paramètre          
def set_distance(ids):
    if len(ids) == 0:
        print("Liste d'identifiants vide")
        return
    query = '''
     MATCH (p:POI)
     WHERE p.id in ''' + str(ids) + '''
     RETURN p.coordinates as item
    '''
    
    records, summary,_keys = driver.execute_query(query)
    print("Query `{query}` returned {records_count} records in {time} ms.".format(
    query=summary.query, records_count=len(records),
    time=summary.result_available_after
    ))
    #Ajouter les distances entre les POIs donnés en paramètre
    locations = []
    
    for record in records:
        item = record.data()['item']
        #On inverse les coordonnées, car l'API prend en entrée (long,lat)
        locations.append([item[1],item[0]])
        
    distances_json = get_ors_matrix(locations,metrics=['distance'])
    
    #On vérifie que l'API n'a pas envoyé une erreur
    if 'error' not in distances_json.keys():
        distances = distances_json['distances']
        sources_snapped_distances = distances_json['sources']
        destinations_snapped_distances = distances_json['destinations']
        for i,idstart in enumerate(ids):
            for j,idstop in enumerate(ids):
                if (j > i) & (distances[i][j] != None):
                    if 'snapped_distance' in sources_snapped_distances[i].keys():
                        source_distance = sources_snapped_distances[i]['snapped_distance']
                    else:
                        source_distance = 0
                        print("Index without snapped distance for source: ",i)
                    if 'snapped_distance' in destinations_snapped_distances[j].keys():
                        destination_distance = destinations_snapped_distances[j]['snapped_distance']
                    else:
                        destination_distance = 0
                        print("Index without snapped distance for destination: ",j)
                    distance = source_distance + distances[i][j] + destination_distance
                   
                    query = '''
                    MATCH (start:POI{id:\'''' + str(idstart) + '''\'})
                    MATCH (stop:POI{id:\'''' + str(idstop) + '''\'})
                    MERGE (start)-[r1:ONFOOT]-(stop)
                    SET r1.distance = ''' + str(distance) + '''
                    '''
                    with driver.session() as session:
                        session.run(query)
                        
def set_poi_cluster(id,community):
    query = '''
        MATCH (n:POI{id:\'''' + id + '''\'})
        SET n.cluster = ''' + str(community) + '''
        '''
    with driver.session() as session:
        session.run(query)

async def set_clusters(tx):
    filename = "file:///output_with_clusters.csv"
    query = '''
        LOAD CSV WITH HEADERS FROM \'''' + filename + '''\' AS row
        MATCH (n:POI{id:row.URI})
        SET n.cluster = row.Cluster
        '''
    result = await tx.run(query)
    records = await result.values()
    print(len(records))

#Créer des clusters avec kmeans et centroids
#Après intégration ces points seront déterminés de manière automatique
#Je donne une liste de points situés dans les endroits les plus populaires de la région (ici des arrondissements à Paris)
''' ['75008#paris', '75004#paris', '75001#paris', '75018#paris',
'75009#paris', '75006#paris', '75012#paris', '75007#paris',
'75011#paris', '75010#paris'] ''' 
def create_clusters(graphname='poisidf'):
    
    best_places = [[48.872725,2.312558],[48.854351,2.357626],[48.862550,2.336419],[48.892570,2.348177],\
    [48.877165,2.337456],[48.849121,2.332884]]
    query = '''
    CALL{
    CALL gds.kmeans.stream( \'''' + graphname + '''\', {
    nodeProperty: 'coordinates',
    k: ''' + str(len(best_places)) + ''',
    seedCentroids: ''' + str(best_places) + '''
    })
    YIELD nodeId, communityId
    RETURN gds.util.asNode(nodeId) AS node, communityId
    ORDER BY communityId
    } with node,communityId where node.city='paris' AND node.category CONTAINS 'placeofinterest' return node,communityId
    '''
    print(query)
    
    records, summary,_keys = driver.execute_query(query)
    print("Query `{query}` returned {records_count} records in {time} ms.".format(
    query=summary.query, records_count=len(records),
    time=summary.result_available_after
    ))
    print(records[0].data())
    query = ''
    for record in records:
        item = record.data()['node']
        community = record.data()['communityId']
        set_poi_cluster(item['id'],community)
    
     
          
def init_graph(queries):
    print('Deleting previous data')
    print('Inserting POIs')
    with driver.session() as session:
        for query in queries:
            session.run(query)
    print('done')
    
if __name__ == '__main__':
    ''' filename_poi='file:///graph-reg-idf.csv'
    load_pois_from_csv(filename_poi)
    filename_cities = 'file:///cities_clean.csv'
    load_cities_from_csv(filename_cities) '''
    filename_ranking = 'file:///cities_ranking.csv' 
    load_ranking(filename_ranking)
    create_index()