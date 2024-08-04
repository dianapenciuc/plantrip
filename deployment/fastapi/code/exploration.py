from neo4j import GraphDatabase
import neo4j
from pprint import pprint
from create_graph import init_graph
import math

driver = GraphDatabase.driver('bolt://0.0.0.0:7687',
                              auth=('neo4j', 'your_password'))

#Je crée une projection sur un cluster donné pour sélectionner un sous-ensemble du graphe complet
#ensuite j'effectuerai ma recherche d'itinéraire dans le graphe de la projection
#J'élimine des catégories qui ne m'intéresse pas
def create_projection(cluster_nr):
    if not cluster_nr.isnumeric():
        return
    graph_name = 'graph' + cluster_nr
    if graph_exists(graph_name):
        return 
    query = '''
    MATCH (source: POI)
    WHERE source.cluster in [\'''' + cluster_nr + '''\'] and
    none(x in split(source.category,' ') where x in ['serviceprovider','foodestablishment','accommodation','restaurant','store'])
    MATCH (source)-[r:ONFOOT]-(target)
    WHERE none(x in split(target.category,' ') where x in ['serviceprovider','foodestablishment','accommodation','restaurant','store'])
    WITH gds.graph.project(\'''' + graph_name + '''\',source,target,
    { sourceNodeLabels: labels(source),
        targetNodeLabels: labels(target),
        relationshipProperties: r { .distance }

    },  {undirectedRelationshipTypes: ['*']}
    ) as g
    RETURN g.graphName AS graph, g.nodeCount AS nodes, g.relationshipCount AS rels
    '''
    
    with driver.session() as session:
        result = session.run(query).data()
    return result[0]['graph']

def graph_exists(graph_name):
    query = '''
    RETURN gds.graph.exists(\'''' + graph_name + '''\') as response
    '''
    with driver.session() as session:
        result = session.run(query).data()
        
    return result[0]['response']
    
def delete_projection(graph_name):
    if graph_exists(graph_name):
        query = '''
        CALL gds.graph.drop(\'''' + graph_name + '''\') YIELD graphName;
        '''
        with driver.session() as session:
            session.run(query)
    else:
        return

def delete_relations():
    query = '''
    MATCH path = (n:POI)-[r:MINST]-() delete r;
    '''
    with driver.session() as session:
            session.run(query)
            
#Je cherche le minimum spanning tree sur la projection
#Params: source_id : identifiant du POI initial
def min_spanning_tree(graph_name,source_id):
    if not graph_exists(graph_name):
        return 
    find_tree = '''
    MATCH (n:POI {id:  \'''' + source_id + '''\'})
    CALL gds.spanningTree.write(\'''' + graph_name + '''\', {
    sourceNode: n,
    relationshipWeightProperty: 'distance',
    writeProperty: 'writeCost',
    writeRelationshipType: 'MINST'
    })
    YIELD preProcessingMillis, computeMillis, writeMillis, effectiveNodeCount
    RETURN preProcessingMillis, computeMillis, writeMillis, effectiveNodeCount;
    '''
    
    with driver.session() as session:
        session.run(find_tree)
    
    query = '''
    MATCH path = (n:POI {id:  \'''' + source_id + '''\'})-[:MINST*]-()
    WITH relationships(path) AS rels
    UNWIND rels AS rel
    WITH DISTINCT rel AS rel
    RETURN startNode(rel).id AS Source, endNode(rel).id AS Destination, rel.writeCost AS Cost
    '''
    with driver.session() as session:
        result = session.run(query).data()
    
    sum = 0
    path = []
    for record in result:
        sum += record['Cost']
        path.append(record['Source'])
    print('Distance totale : ',sum)
    print('Trajet : ',path)
    return result

#Je cherche le poi le plus proche d'un poi donné
def nearest_point(id_start,minutes):
    #Je considère un temps de marche de 4km/h
    distance = 4000*minutes/60
    query = '''
    CALL{
        MATCH (start:POI{id:$id_start})  
        MATCH p = (start)-[:ONFOOT]-(point)
        with *,relationships(p) as r
        return DISTINCT point.id as poi,reduce(sum=0, x in r|sum+x.distance) as distance
        order by distance desc
    }
    with poi,distance
    where distance < ''' + str(distance) + '''
    return poi,distance ORDER by distance ASC LIMIT 1
    '''
    with driver.session() as session:
        result = session.run(query).data()

    print("POI se trouvant à moins de ",minutes," du départ : ", result[0]['poi'])

def find_path(start_id,cluster_nr):
    graph_name = create_projection(cluster_nr)
    min_spanning_tree(graph_name,start_id)
    delete_relations()
    delete_projection(graph_name)
    
if __name__ == '__main__':
    cluster_nr = "0"
    start = 'https://data.datatourisme.fr/19/8e2fd5c2-be9d-3aed-a618-6759b72b0a36'
    find_path(start,cluster_nr)
    
   

