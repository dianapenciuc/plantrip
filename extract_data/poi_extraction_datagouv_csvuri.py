# Chargement des packages
import requests
import pandas as pd
from io import StringIO
import csv
import numpy as np

# Liste URLs par région des CSV de DATATourisme
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


# Fonction pour charger les données d'une seule URL
def load_single_url(sigle, region, url, chunk_size=100000):
    data_frames = []
    try:
        response = requests.get(url)
        response.raise_for_status()  # Vérifie si la requête a réussi (statut HTTP 200)
        
        # Utilisation de StringIO pour lire le contenu de la réponse comme un fichier CSV
        csv_data = StringIO(response.text)
        
        # Charger les données par morceaux (chunking)
        chunks = pd.read_csv(csv_data, chunksize=chunk_size, low_memory=False)
        
        for chunk in chunks:
            # Ajouter les colonnes Sigle_Region et Region
            chunk['Sigle_Region'] = sigle
            chunk['Region'] = region
            data_frames.append(chunk)
        
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du chargement depuis {url}: {str(e)}")
        
    except pd.errors.ParserError as e:
        print(f"Erreur lors de l'analyse du CSV depuis {url}: {str(e)}")
    
    return data_frames

# Fonction pour charger les données de toutes les URL en utilisant le multithreading
def load_data_from_urls(sigles_regions_urls, chunk_size=100000, max_workers=10):
    all_data_frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre les tâches au thread pool
        futures = [executor.submit(load_single_url, sigle, region, url, chunk_size) for sigle, region, url in sigles_regions_urls]
        
        # Attendre que toutes les tâches soient terminées
        for future in as_completed(futures):
            all_data_frames.extend(future.result())
    
    # Concaténer tous les DataFrames en un seul
    concatenated_df = pd.concat(all_data_frames, join='inner', ignore_index=True)
    
    return concatenated_df

concatenated_df = load_data_from_urls(sigles_regions_urls)
