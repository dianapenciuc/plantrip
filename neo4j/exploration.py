from neo4j import GraphDatabase
import neo4j
from pprint import pprint
from create_graph import init_graph
import math
import time
import pandas as pd

driver = GraphDatabase.driver('bolt://0.0.0.0:7687',
                              auth=('neo4j', 'your_password'))

#Je crée une projection sur un cluster donné pour sélectionner un sous-ensemble du graphe complet
#ensuite j'effectuerai ma recherche d'itinéraire dans le graphe de la projection
#J'élimine des catégories qui ne m'intéresse pas
def create_projection(cluster_nr):
    if not cluster_nr.isnumeric():
        print("Invalid cluster number")
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
    print(result)
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

#Creates a graph from a tree
def get_graph_from_tree_result(result):
    graph = {}
    for node in result:
        if node['Source']['id'] in graph.keys():
            graph[node['Source']['id']].append(node['Destination']['id'])
        else:
            graph[node['Source']['id']] = [node['Destination']['id']]
        if node['Destination']['id'] in graph.keys():
            graph[node['Destination']['id']].append(node['Source']['id'])
        else:
            graph[node['Destination']['id']] = [node['Source']['id']]
    
    return graph

#Crée un itinéraire à partir du résultat de minimum spanning tree 
def DFS(graph,start):
    visited = set()  # keep track of the visited nodes
    stack = [start]  # use a stack to keep track of nodes to visit next

    while stack:
        node = stack.pop()  # get the next node to visit
        if node not in visited:
            visited.add(node)  # mark the node as visited
            stack.extend(graph[node])  # add the node's neighbors to the stack
    
    return visited

#Créer un dictionnaire à partir de l'arbre retourné
def get_tree_dict(result):
    tree_dict = {}
    for record in result:
        item = {
            "source" : {
                "coordinates":record['Source']['coordinates'],
                "category": record['Source']['category'],
                "name":record['Source']['name']
            },
            "destination" : {
                "coordinates":record['Destination']['coordinates'],
                "category": record['Destination']['category'],
                "name":record['Destination']['name']
            },
            "cost":record["Cost"]
        }
        tree_dict[(record['Source']['id'],record['Destination']['id'])] = item
    return tree_dict

def save_to_csv(path,coordinates,scores):
    df = pd.DataFrame(data=[pd.Series(path),pd.Series(coordinates),pd.Series(scores)])
    df.to_csv("itineraire_cluster_249.csv",header=None)
    return df

def depth_first_traversal(graph_name,source_id):
    #Depth first traversal of the tree
    query = '''
    MATCH (source:POI)-[r:MINST'''+ graph_name + ''']-(target:POI) RETURN gds.graph.project(
    'MINST'''+ graph_name + '''\',
    source,
    target
    );
    '''
    with driver.session() as session:
        session.run(query)
    traversal = '''
    MATCH (source:POI{id:\'''' + source_id + '''\'})
    CALL gds.dfs.stream('MINST'''+ graph_name + '''\', {
    sourceNode: source
    })
    YIELD path
    RETURN path;
    '''
    
    with driver.session() as session:
        result = session.run(traversal).data()
        
    #Je supprime la projection utilisée pour le depth first search
    delete_projection('MINST'+ graph_name)
    
    result = result[0]['path']
    result = list(filter(lambda x: x != "NEXT", result))
    return result

#Calcule un score pour le noeud en fonction des thèmes qui correspondent à la demande de l'utilisateur
def get_node_score(node_categories,themes):
        categories = node_categories.split(' ')
        categories_matched = [value for value in categories if value in themes]
        return len(categories_matched) 

def get_paths_from_tree(tree,themes,duration):
    sum = 0
    path = []
    names = []
    coordinates = []
    paths = []
    first = True
    #Je calcule le temps total à pied (en secondes), pour une vitesse de 4km/h; je considère un temps de 1h passé sur chaque POI
    path.append(tree[0]['Source']['id'])
    names.append(tree[0]['Source']['name'])
    coordinates.append(tree[0]['Source']['coordinates'])
    score = get_node_score(tree[0]['Source']['category'],themes)
    #Temps d'arrêt pour visiter le premier objectif touristique
    total_time = 3600
    for record in tree:
        if total_time > duration:
            break
        if first:
            first = False
            cost = record["Cost"]
            sum += cost
            score += get_node_score(record['Destination']['category'],themes)
            path.append(record['Destination']['id'])
            names.append(record['Destination']['name'])
            coordinates.append(record['Destination']['coordinates'])
            total_time += cost * 0.9 + 3600
        #Vérifie si le noeud est le point de départ
        elif record['Source']['id'] == tree[0]['Source']['id']:
            paths.append({"path":path,"names":names,"score":score,"time":total_time,"distance":sum,"coordinates":coordinates})
            sum = record["Cost"]
            path = [record['Source']['id'],record['Destination']['id']]
            names = [record['Source']['name'],record['Destination']['name']]
            coordinates = [record['Source']['coordinates'],record['Destination']['coordinates']]
            score = get_node_score(record['Destination']['category'],themes) + get_node_score(record['Source']['category'],themes)
            total_time = record["Cost"] * 0.9 + 3600
        elif record['Source']['id'] != record['Destination']['id']:
            cost = record["Cost"]
            sum += cost
            score += get_node_score(record['Destination']['category'],themes)
            path.append(record['Destination']['id'])
            names.append(record['Destination']['name'])
            coordinates.append(record['Destination']['coordinates'])
            total_time += cost * 0.9 + 3600
    
    paths.append({"path":path,"names":names,"score":score,"time":total_time,"distance":sum,"coordinates":coordinates})   
    for path in paths:
        print(path)
    print("--") 
    return paths

def get_best_path(paths):
    max_score = 0
    best_path = paths[0]
    if len(paths) > 1:
        for path in paths:
            if path["score"] > max_score:
                max_score = path["score"]
                best_path = {
                    "path":path["path"],
                    "names":path["names"],
                    "score":path["score"],
                    "time":path["time"],
                    "distance":path["distance"],
                    "coordinates":path["coordinates"]
                }
    
    print('Temps trajet : ',best_path["time"])
    print('Trajet : ',best_path["names"])
    
#Je cherche le minimum spanning tree sur la projection
#Params: source_id : identifiant du POI initial
def min_spanning_tree(graph_name,source_id,themes,duration):
    if not graph_exists(graph_name):
        return {}
    find_tree = '''
    MATCH (n:POI {id:  \'''' + source_id + '''\'})
    CALL gds.spanningTree.write(\'''' + graph_name + '''\', {
    sourceNode: n,
    relationshipWeightProperty: 'distance',
    writeProperty: 'writeCost',
    writeRelationshipType: 'MINST'''+ graph_name + '''\'
    })
    YIELD preProcessingMillis, computeMillis, writeMillis, effectiveNodeCount
    RETURN preProcessingMillis, computeMillis, writeMillis, effectiveNodeCount;
    '''
    
    with driver.session() as session:
        session.run(find_tree)
    
    query = '''
    MATCH path = (n:POI {id:  \'''' + source_id + '''\'})-[:MINST'''+ graph_name + '''*]-()
    WITH relationships(path) AS rels
    UNWIND rels AS rel
    WITH DISTINCT rel AS rel
    RETURN startNode(rel) AS Source, endNode(rel) AS Destination, rel.writeCost AS Cost
    '''
    with driver.session() as session:
        tree = session.run(query).data()
    
    #graph = get_graph_from_tree_result(tree)
    if len(tree) < 2:
        print("No tree for POI with id ", source_id)
        return {}
    paths = get_paths_from_tree(tree,themes,duration)
    best_path =  get_best_path(paths)
    return best_path

#Je cherche le poi le plus proche d'un poi donné
def nearest_point(id_start,minutes):
    #Je considère un temps de marche de 4km/h
    distance = 4000*minutes/60
    query = '''
    CALL{
        MATCH (start:POI{id:''' + id_start + '''})  
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

#Par défaut, la durée du trajet est de 10h
def find_path(start_id,themes=[],duration=36000):
    query = '''
    MATCH path = (n:POI {id:  \'''' + start_id + '''\'}) return n.cluster as cluster_nr
    '''
    with driver.session() as session:
        result = session.run(query).data()
    
    cluster_nr = result[0]['cluster_nr']
    graph_name = create_projection(cluster_nr)
    result = min_spanning_tree(graph_name,start_id,themes,duration)
    print(result)
    delete_relations()
    delete_projection(graph_name)

def find_best_route(cluster_nr,themes=[],duration=36000):
    query = '''
    MATCH (p:POI) WHERE p.cluster = \'''' + cluster_nr + '''\' and
    none(x in split(p.category,' ') where x in ['serviceprovider','foodestablishment','accommodation','restaurant','store']) RETURN p as poi
    '''
    with driver.session() as session:
        result = session.run(query).data()
    
    paths = []
    for poi in result:
        path = find_path(poi['poi']['id'],themes,duration)
        if path != {}:
            paths.append(path)
    
    print(paths)   
    return
   
if __name__ == '__main__':
    ''' cluster_nr = "0"
    delete_projection("graph0")
    start = 'https://data.datatourisme.fr/19/8e2fd5c2-be9d-3aed-a618-6759b72b0a36'
    themes = ['parkandgarden','culturalsite'] '''
    #La tour Eiffel
    start = 'https://data.datatourisme.fr/19/4b69d406-412a-3136-8cb8-cf8bac872956'
    themes = ['parkandgarden','culturalsite','museum']
    heure_debut = time.time()
    find_path(start,themes,duration=30000)
    heure_fin = time.time()
    temps = heure_fin - heure_debut
    print("Temps execution recherche itinéraire:",temps)
    
    
   

