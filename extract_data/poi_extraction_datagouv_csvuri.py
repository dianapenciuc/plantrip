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
        self.graph = Graph(graph_uri, auth=(user, password))

    def create_node(self, row):
        query = (
            f'CREATE (poi:POI {{name: "{row["Nom_du_POI"]}", description: "{row["Description"]}", '
            f'category: "{row["Catégorie_OK"]}", address: "{row["Adresse_postale"]}", '
            f'postal_code: "{row["code_postal"]}", commune: "{row["commune"]}", '
            f'region: "{row["Region"]}", region_sigle: "{row["Sigle_Region"]}", '
            f'theme: "{row["Thème_OK"]}", latitude: {row["Latitude"]}, longitude: {row["Longitude"]}}})'
        )
        self.graph.run(query)

    def create_relationship(self, row):
        query = (
            f'MATCH (poi1:POI {{name: "{row["POI1"]}"}}), (poi2:POI {{name: "{row["POI2"]}"}}) '
            f'MERGE (poi1)-[:EST_DISTANT_DE {{distance: {row["distance_POI1_POI2"]}, distance_normalized: {row["normalized_distance"]}}}]->(poi2)'
        )
        self.graph.run(query)

    def load_nodes(self, df):
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(self.create_node, [row for _, row in df.iterrows()])

    def load_relationships(self, df):
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(self.create_relationship, [row for _, row in df.iterrows()])

# 4/2 Connection  au contenaire Neo4j

client = DockerClient()

client.containers.run(image='datascientest/neo4j:latest', name='my_neo4j', detach=True, auto_remove=True, ports={'7474/tcp': 7474, '7687/tcp':7687}, network='bridge')


# 4/3 Chargement des POIs et des relations dans le contenaire Neo4j

loader = Neo4jLoader(graph_uri="bolt://localhost:7687", user="neo4j", password="test")
loader.load_nodes(filtered_df_Lieu_Corse)
loader.load_relationships(distance_df)


################
# 5/ CLUSTERING
################


# 5/1 Identification du meilleur modèle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
import networkx as nx
import community as community_louvain

# Charger les données
df = filtered_df_Lieu_Corse_reset  # Remplacez 'filtered_df_Lieu_Corse_reset' par votre dataframe

# Utiliser les colonnes Latitude et Longitude pour le clustering
data = df[['Latitude', 'Longitude']].values

# Initialisation des résultats
results = []

# KMeans
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(data)
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    results.append(('KMeans', n_clusters, silhouette, davies_bouldin))

# Gaussian Mixture Model
for n_clusters in range(2, 10):
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    labels = gmm.fit_predict(data)
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    results.append(('GMM', n_clusters, silhouette, davies_bouldin))

# HDBSCAN
hdbscan = HDBSCAN(min_cluster_size=10)
labels = hdbscan.fit_predict(data)
silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else float('inf')
davies_bouldin = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else float('inf')
results.append(('HDBSCAN', len(set(labels)), silhouette, davies_bouldin))

# DBSCAN
for eps in np.arange(0.1, 1.0, 0.1):
    dbscan = DBSCAN(eps=eps)
    labels = dbscan.fit_predict(data)
    if len(set(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        results.append(('DBSCAN', len(set(labels)), silhouette, davies_bouldin))

# Spectral Clustering
for n_clusters in range(2, 10):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)
    labels = spectral.fit_predict(data)
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    results.append(('SpectralClustering', n_clusters, silhouette, davies_bouldin))

# MeanShift
bandwidth_list = estimate_bandwidth(data) * np.linspace(0.5, 1.5, 5)
for bandwidth in bandwidth_list:
    meanshift = MeanShift(bandwidth=bandwidth)
    labels = meanshift.fit_predict(data)
    if len(set(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        results.append(('MeanShift', len(set(labels)), silhouette, davies_bouldin))

# Agglomerative Clustering
for n_clusters in range(2, 10):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglomerative.fit_predict(data)
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    results.append(('AgglomerativeClustering', n_clusters, silhouette, davies_bouldin))

# Affichage des résultats sous forme de tableau
df_results = pd.DataFrame(results, columns=['Model', 'Clusters', 'Silhouette Score', 'Davies-Bouldin Score'])
print(df_results)

# Graphiques comparatifs
fig, ax = plt.subplots(3, 1, figsize=(12, 18))

# Silhouette Score
df_results.groupby('Model').mean()['Silhouette Score'].plot(kind='bar', ax=ax[0], color='skyblue')
ax[0].set_title('Mean Silhouette Score by Model')
ax[0].set_ylabel('Silhouette Score')

# Davies-Bouldin Score
df_results.groupby('Model').mean()['Davies-Bouldin Score'].plot(kind='bar', ax=ax[1], color='salmon')
ax[1].set_title('Mean Davies-Bouldin Score by Model')
ax[1].set_ylabel('Davies-Bouldin Score')

# Nombre de clusters
df_results.groupby('Model').mean()['Clusters'].plot(kind='bar', ax=ax[2], color='lightgreen')
ax[2].set_title('Mean Number of Clusters by Model')
ax[2].set_ylabel('Number of Clusters')

plt.tight_layout()
plt.show()



# 5/2 Clustering



# Score silhouette

from sklearn.metrics import silhouette_score

best_score = -1
best_clusters = None
best_bandwidth = None

for i, clusters in enumerate(results):
    score = silhouette_score(data, clusters)
    print(f"Silhouette Score for bandwidth={bandwidth_list[i]}: {score}")
    
    if score > best_score:
        best_score = score
        best_clusters = clusters
        best_bandwidth = bandwidth_list[i]

print(f"\nBest bandwidth: {best_bandwidth} with Silhouette Score: {best_score}")
print(f"Clusters: {best_clusters}")


# Score Bouldin

from sklearn.metrics import davies_bouldin_score

best_score = float('inf')
best_clusters = None
best_bandwidth = None

for i, clusters in enumerate(results):
    score = davies_bouldin_score(data, clusters)
    print(f"Davies-Bouldin Index for bandwidth={bandwidth_list[i]}: {score}")
    
    if score < best_score:
        best_score = score
        best_clusters = clusters
        best_bandwidth = bandwidth_list[i]

print(f"\nBest bandwidth: {best_bandwidth} with Davies-Bouldin Index: {best_score}")
print(f"Clusters: {best_clusters}")


filtered_df_Lieu_Corse['best_clusters']=best_clusters



# Affichage des meilleurs clusters


import folium
import pandas as pd
import numpy as np
from itertools import cycle

    # Créer une carte centrée sur la Corse
map_center = [42.0, 9.0]  # Coordonnées du centre de la Corse
folium_map = folium.Map(location=map_center, zoom_start=8)

    # Palette de couleurs disponibles dans matplotlib
colors = [
    'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'lightgreen', 'cadetblue',
    'yellow', 'cyan', 'magenta', 'lime', 'navy', 'teal', 'gold', 'olive', 'chocolate', 'indigo'
]

    # Créer un cycle de couleurs pour attribuer des couleurs uniques à chaque cluster
color_cycle = cycle(colors)

    # Générer le dictionnaire de mapping des couleurs pour les clusters
commune_color_map = {i: next(color_cycle) for i in range(20)}  # Changer le nombre 20 par le nombre de clusters maximum

    # Ajouter des marqueurs pour chaque POI sur la carte, colorés par commune
for index, row in filtered_df_Lieu_Corse.iterrows():
    cluster = row['best_clusters']  # Utiliser le cluster basé sur le meilleur score Davies-Bouldin ou silhouette
    color = commune_color_map.get(cluster, 'blue')  # Utiliser bleu par défaut si la commune n'est pas dans la map
    popup_text = f"Nom: {row['Nom_du_POI']}<br>ville: {row['ville']}<br>Cluster: {cluster}"
    folium.Marker(
        [row['Latitude'], row['Longitude']],
        popup=popup_text,
        icon=folium.Icon(color=color)
    ).add_to(folium_map)

    # Afficher la carte
folium_map.save('clustered_pois_map.html')  # Sauvegarder la carte au format HTML
folium_map


    # Plot clusters vs Ville

import matplotlib.pyplot as plt

    # Créer le crosstab entre les clusters et les villes
crosstab_result = pd.crosstab(filtered_df_Lieu_Corse['best_clusters'], filtered_df_Lieu_Corse['ville'])

    # Créer une figure pour le heatmap
plt.figure(figsize=(12, 8))

    # Générer le heatmap avec une palette de couleurs
sns.heatmap(crosstab_result, annot=True, fmt='d', cmap='YlGnBu', linewidths=0.5, linecolor='black')

    # Ajouter des titres et labels
plt.title('Heatmap des POI par Cluster et Ville', fontsize=16)
plt.xlabel('Ville', fontsize=12)
plt.ylabel('Cluster', fontsize=12)

    # Afficher le heatmap
plt.show()


    # Manipulation des clusters

             # Utilisation de la classe
             # Sélection des données de la Corse (COR) qui représenteront les noeuds du graphe
filtered_df_Lieu_Corse_Cluster1 = filtered_df_Lieu_Corse[(filtered_df_Lieu_Corse['best_clusters'] == 1)]
filtered_df_Lieu_Corse_Cluster0 = filtered_df_Lieu_Corse[(filtered_df_Lieu_Corse['best_clusters'] == 0)]
filtered_df_Lieu_Corse_Cluster2 = filtered_df_Lieu_Corse[(filtered_df_Lieu_Corse['best_clusters'] == 2)]

             # Créer une instance de la classe GeodesicDistanceCalculator
distance_calculator_Lieu_Corse_Cluster1 = GeodesicDistanceCalculator(filtered_df_Lieu_Corse_Cluster1)
distance_calculator_Lieu_Corse_Cluster0 = GeodesicDistanceCalculator(filtered_df_Lieu_Corse_Cluster0)
distance_calculator_Lieu_Corse_Cluster2 = GeodesicDistanceCalculator(filtered_df_Lieu_Corse_Cluster2)

             # Calculer les distances et créer la matrice de distances
distance_matrix_Lieu_Corse_Cluster1 = distance_calculator_Lieu_Corse_Cluster1.create_distance_matrix()
distance_matrix_Lieu_Corse_Cluster0 = distance_calculator_Lieu_Corse_Cluster0.create_distance_matrix()
distance_matrix_Lieu_Corse_Cluster2 = distance_calculator_Lieu_Corse_Cluster2.create_distance_matrix()

             # Appliquer K-Means contraint sur les données
filtered_df_Lieu_Corse_Cluster1_with_labels = distance_calculator_Lieu_Corse_Cluster1.apply_kmeans_constrained(n_clusters=10, size_min=None, size_max=20, random_state=0, n_jobs=-1)
filtered_df_Lieu_Corse_Cluster0_with_labels = distance_calculator_Lieu_Corse_Cluster0.apply_kmeans_constrained(n_clusters=10, size_min=None, size_max=20, random_state=0, n_jobs=-1)
filtered_df_Lieu_Corse_Cluster2_with_labels = distance_calculator_Lieu_Corse_Cluster2.apply_kmeans_constrained(n_clusters=10, size_min=None, size_max=20, random_state=0, n_jobs=-1)

             # Nombre de clusters
print(len(filtered_df_Lieu_Corse_Cluster1_with_labels['label_des_clusters'].unique()))
print(len(filtered_df_Lieu_Corse_Cluster0_with_labels['label_des_clusters'].unique()))
print(len(filtered_df_Lieu_Corse_Cluster2_with_labels['label_des_clusters'].unique()))

             # labels des clusters
print(filtered_df_Lieu_Corse_Cluster1_with_labels['label_des_clusters'].unique())
print(filtered_df_Lieu_Corse_Cluster0_with_labels['label_des_clusters'].unique())
print(filtered_df_Lieu_Corse_Cluster2_with_labels['label_des_clusters'].unique())

             # POIs par cluster
clusters = filtered_df_Lieu_Corse_Cluster1_with_labels.groupby('label_des_clusters')['Nom_du_POI'].apply(list).to_dict()
for cluster, pois in clusters.items():
    print(f"Cluster {cluster}: {pois}")
clusters = filtered_df_Lieu_Corse_Cluster0_with_labels.groupby('label_des_clusters')['Nom_du_POI'].apply(list).to_dict()
for cluster, pois in clusters.items():
    print(f"Cluster {cluster}: {pois}")
clusters = filtered_df_Lieu_Corse_Cluster2_with_labels.groupby('label_des_clusters')['Nom_du_POI'].apply(list).to_dict()
for cluster, pois in clusters.items():
    print(f"Cluster {cluster}: {pois}")

             # Effectif des POIs par subcluster
poi_count_per_cluster_Lieu_Corse_Cluster1 = filtered_df_Lieu_Corse_Cluster1_with_labels['label_des_clusters'].value_counts()
print("\nNombre de POIs par cluster:")
poi_count_per_cluster_Lieu_Corse_Cluster1
poi_count_per_cluster_Lieu_Corse_Cluster0 = filtered_df_Lieu_Corse_Cluster0_with_labels['label_des_clusters'].value_counts()
print("\nNombre de POIs par cluster:")
poi_count_per_cluster_Lieu_Corse_Cluster0
poi_count_per_cluster_Lieu_Corse_Cluster2 = filtered_df_Lieu_Corse_Cluster1_with_labels['label_des_clusters'].value_counts()
print("\nNombre de POIs par cluster:")
poi_count_per_cluster_Lieu_Corse_Cluster2

             #
filtered_df_Lieu_Corse_Cluster1_with_labels['label_des_subclusters'] = (
    filtered_df_Lieu_Corse_Cluster1_with_labels['best_clusters'].astype(str) + '.' + 
    filtered_df_Lieu_Corse_Cluster1_with_labels['label_des_clusters'].astype(str)
)
filtered_df_Lieu_Corse_Cluster0_with_labels['label_des_subclusters'] = (
    filtered_df_Lieu_Corse_Cluster0_with_labels['best_clusters'].astype(str) + '.' + 
    filtered_df_Lieu_Corse_Cluster0_with_labels['label_des_clusters'].astype(str)
)
filtered_df_Lieu_Corse_Cluster2_with_labels['label_des_subclusters'] = (
    filtered_df_Lieu_Corse_Cluster2_with_labels['best_clusters'].astype(str) + '.' + 
    filtered_df_Lieu_Corse_Cluster2_with_labels['label_des_clusters'].astype(str)
)
filtered_df_Lieu_Corse_Cluster0_with_labels


filtered_df_Lieu_Corse_Cluster_Other = filtered_df_Lieu_Corse.loc[~filtered_df_Lieu_Corse['best_clusters'].isin([0, 1, 2])]
filtered_df_Lieu_Corse_Cluster_Other['label_des_clusters']=filtered_df_Lieu_Corse['best_clusters']
filtered_df_Lieu_Corse_Cluster_Other['label_des_subclusters']=filtered_df_Lieu_Corse['best_clusters'].astype(str)
filtered_df_Lieu_Corse_Cluster_Other.info()


filtered_df_Lieu_Corse_Cluster_Other = filtered_df_Lieu_Corse[filtered_df_Lieu_Corse['best_clusters'].isin([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])]
filtered_df_Lieu_Corse_Cluster_Other['label_des_clusters']=filtered_df_Lieu_Corse['best_clusters']
filtered_df_Lieu_Corse_Cluster_Other['label_des_subclusters']=filtered_df_Lieu_Corse['best_clusters'].astype(str)
filtered_df_Lieu_Corse_Cluster_Other.info()


        # Concaténation des DataFrames pour la formation des subclusters

filtered_df_Lieu_Corse_reset = pd.concat([
    filtered_df_Lieu_Corse_Cluster1_with_labels,
    filtered_df_Lieu_Corse_Cluster0_with_labels,
    filtered_df_Lieu_Corse_Cluster2_with_labels,
    filtered_df_Lieu_Corse_Cluster_Other
])

        # Réinitialisation de l'index si nécessaire
filtered_df_Lieu_Corse_reset.reset_index(drop=True, inplace=True)

        # Plot des subclusters

import matplotlib.pyplot as plt

                   # Créer le crosstab entre les clusters et les villes
crosstab_result = pd.crosstab(filtered_df_Lieu_Corse_reset['best_clusters'], filtered_df_Lieu_Corse_reset['label_des_clusters'])

                   # Créer une figure pour le heatmap
plt.figure(figsize=(12, 8))

                   # Générer le heatmap avec une palette de couleurs
sns.heatmap(crosstab_result, annot=True, fmt='d', cmap='YlGnBu', linewidths=0.5, linecolor='black')

                   # Ajouter des titres et labels
plt.title('Heatmap des POI par Cluster et sous-clusters', fontsize=16)
plt.xlabel('Sous-clusters', fontsize=12)
plt.ylabel('Cluster', fontsize=12)

                   # Afficher le heatmap
plt.show()


         # Enregistrement du DataFrame pour streamlit

joblib.dump(filtered_df_Lieu_Corse_reset, 'filtered_df_Lieu_Corse_reset.joblib')
filtered_df_Lieu_Corse_reset.to_parquet('filtered_df_Lieu_Corse_reset.parquet')

         # Chargement du DataFrame avec joblib
filtered_df_Lieu_Corse_reset_loaded_joblib = joblib.load('filtered_df_Lieu_Corse_reset.joblib')
filtered_df_Lieu_Corse_reset_loaded_parquet = pd.read_parquet('filtered_df_Lieu_Corse_reset.parquet')


# 5/3 Chargement dans Neo4j

        #5/3/1 Classe pour le chargement des clusters dans  Neo4j

from py2neo import Graph
from concurrent.futures import ThreadPoolExecutor

class Neo4jLoader:
    def __init__(self, graph_uri, user, password):
        self.graph = Graph(graph_uri, auth=(user, password))

    def create_node(self, row):
        query = (
            f'CREATE (poi:POI {{name: "{row["Nom_du_POI"]}", description: "{row["Description"]}", '
            f'category: "{row["Catégorie_OK"]}", address: "{row["Adresse_postale"]}", '
            f'postal_code: "{row["code_postal"]}", commune: "{row["commune"]}", commune: "{row["commune"]}", '
            f'region: "{row["Region"]}", region_sigle: "{row["Sigle_Region"]}", '
            f'theme: "{row["Thème_OK"]}", latitude: {row["Latitude"]}, longitude: "{row["Longitude"]}", '
            f'cluster: "{row["best_clusters"]}", subcluster: "{row["label_des_subclusters"]}"}})'
        )
        self.graph.run(query)

    def create_relationship(self, row):
        query = (
            f'MATCH (poi1:POI {{name: "{row["POI1"]}"}}), (poi2:POI {{name: "{row["POI2"]}"}}) '
            f'MERGE (poi1)-[:EST_DISTANT_DE {{distance: {row["distance_POI1_POI2"]}, distance_normalized: {row["normalized_distance"]}}}]->(poi2)'
        )
        self.graph.run(query)

    def load_nodes(self, df):
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(self.create_node, [row for _, row in df.iterrows()])

    def load_relationships(self, df):
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(self.create_relationship, [row for _, row in df.iterrows()])

            # 5/3/3 Chargement des Noeuds et des relations

# Connexion à Neo4j
loader = Neo4jLoader(graph_uri="bolt://localhost:7687", user="neo4j", password="test")
# Chargement des noeuds
loader.load_nodes(filtered_df_Lieu_Corse_reset)
#Chargement des relations
loader.load_relationships(distance_df)
