from neo4j import GraphDatabase
from pprint import pprint

driver = GraphDatabase.driver('bolt://0.0.0.0:7687',
                              auth=('neo4j', 'your_password'))

def clean_database():
    # deleting data
    query = ['''
    MATCH (n) 
    DETACH DELETE n;
    '''
    ]
    with driver.session() as session:
        session.run(query)

def load_pois_from_csv(filename):
    # loading POI data
    query = ''' 
    LOAD CSV WITH HEADERS FROM \'''' + filename + '''\' AS row
    CREATE (:POI {
        name: row.Nom_du_POI,
        id: row.URI_ID_du_POI,
        city: row.Commune,
        category: row.Categories_de_POI,
        coordinates:[toFloat(row.Latitude),toFloat(row.Longitude)]
        });
    '''
    with driver.session() as session:
        session.run(query)
 
def load_cities_from_csv(filename):
    # loading cities data
    query = ''' 
    LOAD CSV WITH HEADERS FROM \'''' + filename + '''\' AS row
    CREATE (:City {
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
    
def get_distance(start,stop):
    #Calculer les distances entre POIs
    query = '''
    
    '''
    with driver.session() as session:
        session.run(query)
            
def init_graph():
    print('Deleting previous data')
    print('Inserting POIs')
    with driver.session() as session:
        for query in queries:
            session.run(query)
    print('done')
    
if __name__ == '__main__':
    filename_poi='file:///graph-reg-idf.csv'
    load_pois_from_csv(filename_poi)
    filename_cities = 'file:///cities_clean.csv'
    load_cities_from_csv(filename_cities)