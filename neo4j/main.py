from neo4j import AsyncGraphDatabase
import neo4j 
from create_graph import create_index, load_ranking, set_clusters, set_distance, driver
import asyncio
import time

URI_NEO4J = 'bolt://0.0.0.0:7687'
AUTH = neo4j.basic_auth("neo4j", "your_password")

#Crée des relations de type ONFOOT avec une propriété "distance" pour chaque pair
#de POI d'un cluster donné en paramètre
def create_distances(cluster_nr):
    query = '''
    MATCH path = (n:POI {cluster:  \'''' + cluster_nr + '''\'}) return collect(n.id) as ids
    '''
    with driver.session() as session:
        result = session.run(query).data()
    
    print(result[0]['ids'])
    set_distance(result[0]['ids'])


async def main():
    async with AsyncGraphDatabase.driver(URI_NEO4J, auth=AUTH) as driver:
        async with driver.session(database="neo4j") as session:
            await session.execute_write(set_clusters)
            print("Clusters added to the graph database")


     
if __name__ == '__main__':
    if __name__ == "__main__":
        ''' cluster_nr = "249"
        create_distances(cluster_nr) '''
        heure_debut = time.time()
        #asyncio.run(main())
        heure_fin = time.time()
        temps = heure_fin - heure_debut
        print("Cette fonction s'est exécutée en {} s".format(temps))
        filename_ranking = 'file:///cities_ranking.csv' 
        load_ranking(filename_ranking)
        #create_index()
   