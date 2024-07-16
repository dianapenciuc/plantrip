from neo4j import AsyncGraphDatabase
import neo4j 
from create_graph import set_clusters, set_distance
import asyncio
import time

URI_NEO4J = 'bolt://0.0.0.0:7687'
AUTH = neo4j.basic_auth("neo4j", "your_password")

def create_distances():
    ids = []
    with open('data/selected_pois.txt', 'r',encoding='utf-8-sig') as file:
        for line in file:
            ids.append(line.rstrip('\n'))
    print(ids)
    set_distance(ids)


async def main():
    async with AsyncGraphDatabase.driver(URI_NEO4J, auth=AUTH) as driver:
        async with driver.session(database="neo4j") as session:
            await session.execute_write(set_clusters)
            print("Clusters added to the graph database")

    
     
if __name__ == '__main__':
    if __name__ == "__main__":
        heure_debut = time.time()
        asyncio.run(main())
        heure_fin = time.time()
        temps = heure_fin - heure_debut
        print("Cette fonction s'est exécutée en {} s".format(temps))
   