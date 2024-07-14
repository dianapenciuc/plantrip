# Chargement des packages
import requests
import pandas as pd
from io import StringIO
import csv
import numpy as np
from py2neo import Graph, Node, Relationship
from geopy.distance import geodesic
from neo4j import GraphDatabase
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import time
from geopy.distance import geodesic
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import squareform, pdist
from concurrent.futures import ThreadPoolExecutor, as_completed
from k_means_constrained import KMeansConstrained
from sklearn.preprocessing import MinMaxScaler
import folium
from folium.plugins import MarkerCluster
import matplotlib.colors as mcolors

###########################
# 1/ EXTRACTION DES DONNEES
###########################

# 1/1 Ingestion des données de DATAToutisme dans un dataframe


class DataLoader:
    def __init__(self, urls, chunk_size=100000, max_workers=10):
        self.urls = urls
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.data_frames = []

    def load_single_url(self, sigle, region, url):
        data_frames = []
        try:
            response = requests.get(url)
            response.raise_for_status()
            csv_data = StringIO(response.text)
            chunks = pd.read_csv(csv_data, chunksize=self.chunk_size, low_memory=False)
            for chunk in chunks:
                chunk['Sigle_Region'] = sigle
                chunk['Region'] = region
                data_frames.append(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors du chargement depuis {url}: {str(e)}")
        except pd.errors.ParserError as e:
            print(f"Erreur lors de l'analyse du CSV depuis {url}: {str(e)}")
        return data_frames

    def load_data(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.load_single_url, sigle, region, url) for sigle, region, url in self.urls]
            for future in as_completed(futures):
                self.data_frames.extend(future.result())
        return pd.concat(self.data_frames, join='inner', ignore_index=True)


# 1/2 Application de l'ingestion des données

sigles_regions_urls = [
    ("REU", "Ile de la Réunion", "https://www.data.gouv.fr/fr/datasets/r/2b52bb1f-8676-43f2-b883-f673e7015ed9"),
    ("PDL", "Pays de la Loire", "https://www.data.gouv.fr/fr/datasets/r/56d437a7-eb0c-4c31-9138-539be94bc490"),
    ("PAC", "Provence Alpes Côte d’Azur", "https://www.data.gouv.fr/fr/datasets/r/83a1f131-9e23-4c3b-b1c6-e58f33fe7b80"),
    ("OCC", "Occitanie", "https://www.data.gouv.fr/fr/datasets/r/0c463ef6-c00a-48e2-b50a-d17cfe998b84"),
    ("NOR", "Normandie", "https://www.data.gouv.fr/fr/datasets/r/cb1ebc9c-73fb-43e8-9386-a7b3cf83a642"),
    ("NAQ", "Nouvelle Aquitaine", "https://www.data.gouv.fr/fr/datasets/r/734a4e86-d571-48f6-bdc0-596914066606"),
    ("MYT", "Mayote", "https://www.data.gouv.fr/fr/datasets/r/77114c00-9928-49c1-9a1c-5545c10c7101"),
    ("MTQ", "Martinique", "https://www.data.gouv.fr/fr/datasets/r/4f95a530-d106-4ecd-aa39-1b9d639ee45c"),
    ("IDF", "Ile de France", "https://www.data.gouv.fr/fr/datasets/r/b31a1eca-f2ff-495a-9b67-7c0bc281ea57"),
    ("HDF", "Hauts de France", "https://www.data.gouv.fr/fr/datasets/r/838b6af3-74e5-4d51-873d-d359af3f1855"),
    ("GUF", "Guyane", "https://www.data.gouv.fr/fr/datasets/r/338bb298-cc1f-4bfd-adfd-a3c13fbfa393"),
    ("GLP", "Guadeloupe", "https://www.data.gouv.fr/fr/datasets/r/f73506d6-a336-4743-827b-64a39d891158"),
    ("GDE", "Grand Est", "https://www.data.gouv.fr/fr/datasets/r/59956d74-969b-4c42-8ea4-9348f6a70f7a"),
    ("CVL", "Centre Val de Loire", "https://www.data.gouv.fr/fr/datasets/r/6063e108-f8bd-4541-ba67-a5cadac804fb"),
    ("COR", "Corse", "https://www.data.gouv.fr/fr/datasets/r/2aefffc5-42f5-4e68-ba85-ba19c13fcb4c"),
    ("BRE", "Bretagne", "https://www.data.gouv.fr/fr/datasets/r/ab746af8-d21a-42d1-acae-fdfb2e52ecd5"),
    ("BFC", "Bourgogne Franche Comté", "https://www.data.gouv.fr/fr/datasets/r/d92f0184-e9cb-4bc9-81b0-b43fbcf2a0d2"),
    ("ARA", "Auvergne Rhône Alpes", "https://www.data.gouv.fr/fr/datasets/r/5b3c2cee-44b7-48bd-b4e8-439a03ff6cd2")
]

data_loader = DataLoader(sigles_regions_urls)
concatenated_df = data_loader.load_data()
concatenated_df.head()




###############################
# 2/ TRANSFORMATION DES DONNEES
###############################

# 2/1 Transformation pour identifier les catégories et les thèmes

class DataTransformer:
    def __init__(self, df):
        self.df = df

    def transform_data(self):
        self.df[['code_postal', 'commune']] = self.df['Code_postal_et_commune'].str.split('#', expand=True)
        self.df_cleaned = self.df[['URI_ID_du_POI', 'Nom_du_POI', 'Description', 'Categories_de_POI', 'Latitude', 'Longitude', 'Adresse_postale', 'code_postal', 'commune', 'Region', 'Sigle_Region', 'Date_de_mise_a_jour', 'Contacts_du_POI']]
        self.clean_categories()
        self.assign_categories()
        self.identify_theme()
        return self.df_cleaned

    def clean_categories(self):
        to_remove = ["https://www.datatourisme.fr/ontology/core#", "http://schema.org/", "http://purl.org/ontology/olo/core#"]
        self.df_cleaned["Categories_de_POI_new"] = self.df_cleaned['Categories_de_POI'].apply(lambda x: self.clean_string(x, to_remove))

    def clean_string(self, input_string, to_remove):
        for item in to_remove:
            input_string = input_string.replace(item, "")
        input_string = input_string.replace("||", "|").strip("|")
        return input_string

    def assign_categories(self):
        target_categories = ['PlaceOfInterest', 'Product', '%Event%', '%Tour%']
        self.df_cleaned['Category_List'] = self.df_cleaned['Categories_de_POI_new'].str.split('|')
        self.df_cleaned['Catégorie_OK'] = self.df_cleaned['Category_List'].apply(lambda x: self.get_categorie_ok(x, target_categories))
        mapping = {'PlaceOfInterest': 'Lieu', '%Event%': 'Evènement et manifestation', '%Tour%': 'Itinéraire touristique', 'Product': 'Produit'}
        self.df_cleaned['Catégorie_OK'] = self.df_cleaned['Catégorie_OK'].replace(mapping)

    def get_categorie_ok(self, categories, target_categories):
        for category in target_categories:
            regex_category = category.replace('%', '.*')
            if any(re.search(regex_category, cat) for cat in categories):
                return category
        return categories[0] if categories else None

    def identify_theme(self):
        self.df_cleaned['Categories_de_POI_temp'] = self.df_cleaned['Categories_de_POI_new'].str.strip().fillna('')
        categories_to_remove = ['PlaceOfInterest', 'PointOfInterest', 'Product', 'Event', 'Tour']
        self.df_cleaned['Categorie_temp'] = self.df_cleaned['Categories_de_POI_temp'].apply(lambda x: self.remove_categories_and_deduplicate(x, categories_to_remove))
        self.df_cleaned['Thème_OK'] = self.df_cleaned['Categorie_temp'].apply(lambda x: x.split('|')[-1])

    def remove_categories_and_deduplicate(self, categories, to_remove):
        categories = categories.split('|')
        filtered_categories = [category for category in categories if not any(rem in category for rem in to_remove)]
        unique_categories = list(dict.fromkeys(filtered_categories))
        return '|'.join(unique_categories)

# 2/2 Application des transformations sur les données

data_transformer = DataTransformer(concatenated_df)
concatenated_df_cleaned_util = data_transformer.transform_data()
print(concatenated_df_cleaned_util.isnull().sum())
print(concatenated_df_cleaned_util["Catégorie_OK"].value_counts())
concatenated_df_cleaned_util['Thème_OK'].value_counts()


#########################
# 3/ CALCUL DES DISTANCES
#########################


# 3/1 Classe pour calculer les distances géodésiques entre POIs

class DistanceCalculator:
    def __init__(self, df):
        self.df = df

    def calculate_distance(self, poi1, poi2):
        lat1, lon1 = poi1['Latitude'], poi1['Longitude']
        lat2, lon2 = poi2['Latitude'], poi2['Longitude']
        return poi1['Nom_du_POI'], poi2['Nom_du_POI'], geodesic((lat1, lon1), (lat2, lon2)).kilometers

    def calculate_all_distances(self, num_threads=4):
        poi_pairs = []
        distances = []
        chunk_size = len(self.df) // num_threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.process_chunk, i, min(i + chunk_size, len(self.df))) for i in range(0, len(self.df), chunk_size)]
            for future in futures:
                local_pairs, local_distances = future.result()
                poi_pairs.extend(local_pairs)
                distances.extend(local_distances)
        return poi_pairs, distances

    def process_chunk(self, start_idx, end_idx):
        local_pairs = []
        local_distances = []
        for i in range(start_idx, end_idx):
            for j in range(i + 1, len(self.df)):
                poi1 = self.df.iloc[i]
                poi2 = self.df.iloc[j]
                poi1_name, poi2_name, distance = self.calculate_distance(poi1, poi2)
                local_pairs.append((poi1_name, poi2_name))
                local_distances.append(distance)
        return local_pairs, local_distances

    def normalize_distances(self, distances):
        scaler = MinMaxScaler()
        normalized_distances = scaler.fit_transform(np.array(distances).reshape(-1, 1)).flatten()
        return normalized_distances

# 3/2 Calcul des distances géodésiques

    # 3/2/1 Sélection des variables qui serviront d'attributs pour les noeuds du graphe
concatenated_df_cleaned_util_pour_graph = concatenated_df_cleaned_util[['Nom_du_POI', 'Description', 'Catégorie_OK', 'Adresse_postale', 'code_postal', 'commune', 'Region', 'Sigle_Region', 'Thème_OK', 'Latitude','Longitude']]

    # 3/2/2 Sélection des données de la Corse (COR) qui représenteront les noeuds du graphe
filtered_df_Lieu_Corse = concatenated_df_cleaned_util_pour_graph[(concatenated_df_cleaned_util_pour_graph['Region'] == 'Corse') & (concatenated_df_cleaned_util_pour_graph['Catégorie_OK'] == 'Lieu')]
print(filtered_df_Lieu_Corse.head())

    # 3/2/3 Créer une instance de la classe DistanceCalculator
distance_calculator = DistanceCalculator(filtered_df_Lieu_Corse)

    # 3/2/4 Calculer toutes les distances entre les paires de POIs
start_time_distances = time.time()
poi_pairs, distances = distance_calculator.calculate_all_distances(num_threads=4)
end_time_distances = time.time()

    # 3/2/5 Normaliser les distances
start_time_normalization = time.time()
normalized_distances = distance_calculator.normalize_distances(distances)
end_time_normalization = time.time()

    # 3/2/6 Créer un DataFrame avec les résultats
distance_df = pd.DataFrame({
    'POI1': [pair[0] for pair in poi_pairs],
    'POI2': [pair[1] for pair in poi_pairs],
    'Distance_km': distances,
    'Normalized_Distance': normalized_distances
})

    # 3/2/7 Afficher le DataFrame final avec les distances calculées et normalisées
print(distance_df.head())

    # 3/2/8 Affichage des temps pris
print(f"Temps pris pour calculer les distances: {end_time_distances - start_time_distances} secondes")
print(f"Temps pris pour la normalisation des distances: {end_time_normalization - start_time_normalization} secondes")
print(f"Temps total pris pour le traitement: {end_time_normalization - start_time_distances} secondes")


    # 3/2/9 Distribution des distances géodésiques entre POIs

           # Définir le style de seaborn pour les graphiques
sns.set(style="whitegrid")

           # Créer l'histogramme des scores
plt.figure(figsize=(10, 6))
sns.histplot(distance_df['Distance_km'], bins=20, kde=True, color='skyblue')
plt.title('Distribution des distances entre POIs en Corse')
plt.xlabel('Distance')
plt.ylabel('Nombre de couples de POIs')
plt.tight_layout()

           # Afficher le graphique
plt.show()




############################################################
# 4/ CHARGEMENT DES DONNEES DE LA REGION DE CORSE DANS NEO4J
############################################################

# 4/1 Classe pour charger les données dans Neo4j
class Neo4jLoader:
    def __init__(self, graph_uri, user, password):
        self.graph = GraphDatabase.driver(graph_uri, auth=(user, password))

    def create_node(self, tx, row):
        query = (
            'MERGE (poi:POI {name: $name, description: $description, '
            'category: $category, address: $address, '
            'postal_code: $postal_code, commune: $commune, '
            'region: $region, region_sigle: $region_sigle, '
            'theme: $theme, latitude: $latitude, longitude: $longitude})'
        )
        tx.run(query, 
               name=row["Nom_du_POI"], 
               description=row["Description"],
               category=row["Catégorie_OK"],
               address=row["Adresse_postale"],
               postal_code=row["code_postal"],
               commune=row["commune"],
               region=row["Region"],
               region_sigle=row["Sigle_Region"],
               theme=row["Thème_OK"],
               latitude=row["Latitude"],
               longitude=row["Longitude"])

    def create_relationship(self, tx, row):
        query = (
            'MATCH (poi1:POI {name: $poi1}), (poi2:POI {name: $poi2}) '
            'MERGE (poi1)-[:EST_DISTANT_DE {distance: $distance, distance_normalized: $distance_normalized}]->(poi2)'
        )
        tx.run(query, 
               poi1=row["POI1"], 
               poi2=row["POI2"],
               distance=row["distance_POI1_POI2"], 
               distance_normalized=row["normalized_distance"])

    def load_nodes(self, df):
        with self.graph.session() as session:
            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(lambda row: session.write_transaction(self.create_node, row), [row for _, row in df.iterrows()])

    def load_relationships(self, df):
        with self.graph.session() as session:
            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(lambda row: session.write_transaction(self.create_relationship, row), [row for _, row in df.iterrows()])


# 4/2 Chargement des POIs et des relations dans Neo4j

loader = Neo4jLoader(graph_uri="bolt://localhost:7687", user="neo4j", password="neo4j")
loader.load_nodes(filtered_df_Lieu_Corse)
loader.load_relationships(distance_df)



################
# 5/ CLUSTERING
################


# 5/1 Classe pour générer les clusters

class GeodesicDistanceCalculator:
    def __init__(self, df):
        self.df = df
        self.distances = []
        self.poi_pairs = [(df.iloc[i], df.iloc[j]) for i in range(len(df)) for j in range(i + 1, len(df))]
        
    def calculate_geodesic_distance(self, point1, point2):
        return geodesic(point1, point2).kilometers

    def calculate_distances(self):
        for poi1, poi2 in self.poi_pairs:
            distance = self.calculate_geodesic_distance((poi1['Latitude'], poi1['Longitude']), (poi2['Latitude'], (poi2['Longitude'])))
            self.distances.append(distance)
        return self.distances

    def create_distance_matrix(self):
        return squareform(self.calculate_distances())

    def get_coordinates(self):
        return self.df[['Latitude', 'Longitude']].values

    def apply_kmeans_constrained(self, n_clusters=65, size_min=None, size_max=20, random_state=0, n_jobs=-1):
        X = self.get_coordinates()
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            random_state=random_state,
            n_jobs=n_jobs
        )
        labels = clf.fit_predict(X)
        self.df['label_des_clusters'] = labels
        return self.df



# 5/2 Utilisation de la classe pour calculer les distances géodésiques entre POIs

      # 5/2/1 Sélection des données de la Corse (COR) qui représenteront les noeuds du graphe
filtered_df_Lieu_Corse = concatenated_df_cleaned_util_pour_graph[(concatenated_df_cleaned_util_pour_graph['Region'] == 'Corse') & (concatenated_df_cleaned_util_pour_graph['Catégorie_OK'] == 'Lieu')]

print(filtered_df_Lieu_Corse.head())

      # 5/2/2 Créer une instance de la classe GeodesicDistanceCalculator
distance_calculator = GeodesicDistanceCalculator(filtered_df_Lieu_Corse)

      # 5/2/3 Calculer les distances et créer la matrice de distances
distance_matrix = distance_calculator.create_distance_matrix()

      # 5/2/4 Appliquer K-Means contraint sur les données
filtered_df_Lieu_Corse_with_labels = distance_calculator.apply_kmeans_constrained(n_clusters=65, size_min=None, size_max=20, random_state=0, n_jobs=-1)

      # 5/2/5 Afficher le DataFrame final avec les labels de clusters
print(filtered_df_Lieu_Corse_with_labels.head())







