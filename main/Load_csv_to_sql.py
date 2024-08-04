#!/usr/bin/env python3
#Librairies utiles pour le script
from os import listdir
from os.path import isfile, join
import mysql.connector
import numpy as np
import pandas as pd

# Définir les types de colonnes et la conversion des types pour le SQL
types_colonnes = {
    'id': 'object','label': 'object','type': 'object','themes': 'object',    
    'startdate' : 'object', 'enddate' : 'object',
    'street' : 'object','postalcode': 'int64','city' : 'object','insee':'object','region' : 'object',
    'latitude' : 'float64','longitude' : 'float64',
    'email' : 'object','web':'object','tel':'object',
    'lastupdate' : 'object', 'comment':'object'
    }

# Tableau de conversion des types de colonnes
convert={
    'object' :'MEDIUMTEXT',
    'int64' : 'INT',
    'float64' : 'FLOAT',
    'bool' : 'BOOL'
    }

# Définition d'une fonction pour vérifier si la table existe déjà dans la base de données
def check_table(nom):
    
    query = f"""
    SELECT *
    FROM information_schema.tables
    WHERE table_name = '{nom}'
    """
    cursor.execute(query)
    result = cursor.fetchone()

    if result:
        print(f"La table '{nom}' existe.")
        return True
    else:
        print(f"La table '{nom}' n'existe pas.")
        return False

# Fonction pour la création des tables dans SQLserver
def manage_table(dataframe, nom_table,fonction):
    df=dataframe
    nom=nom_table
    
    if fonction == "create" :
        create_table_query =f"CREATE TABLE {nom} ("
        for column in df.columns:
            col_type=types_colonnes[column]
            col_type=convert[col_type]
            create_table_query+=f"{column} {col_type},"
        create_table_query=create_table_query[:-1]
        create_table_query+=")"
        print(create_table_query)
    
        cursor.execute(create_table_query)
    conn.commit()

    df = pd.DataFrame(df)
    df.replace({np.nan: None}, inplace=True)

    # Check what Data already is in the table with id
    check_query = f"SELECT id FROM {nom}"
    existing_ids = pd.read_sql(check_query, conn)
    new_df = df[~df['id'].isin(existing_ids['id'])]
    new_df=new_df.drop(columns=['comment'])
    print("Nouelles Données:", new_df.shape)
    count=0
    
    # Insérer les données du DataFrame dans la table
    for index, row in new_df.iterrows():
        sql = f'''
        INSERT INTO {nom} (id,label,type,themes,startdate,enddate,street,postalcode,city,insee,region,latitude,longitude,email,web,tel,lastupdate) 
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        '''
        cursor.execute(sql, tuple(row))
        count+=1
    conn.commit()

### Script principal
## Fera une first task si les fichiers de sortie n'ont jamais été créés, sinon fera une update de ces fichiers
# Connexion au server SQL qui doit être activé

conn = mysql.connector.connect(user="root", password='rootadmin', host='mysql', database="itineraire",port=3306)
print("DataBase Itineraire Connected")
cursor = conn.cursor()

# Récupération du nom des différents fichiers dans le dossier data\csv\raw_extract
fichiers = [f for f in listdir("/projet/data/csv/transformed/") if isfile(join("/projet/data/csv/transformed/", f))]

for fichier in fichiers:
    print("ouverture du fichier :",fichier)
    path_trs="/projet/data/csv/transformed/"+fichier
    try:
        df_trs=pd.read_csv(path_trs, sep=';', low_memory=False)
    except:
        df_trs=pd.read_csv(path_trs, sep=';', engine='python')
    df_trs = df_trs.where(pd.notnull(df_trs), None)

    fichier_deb=fichier.split(".")
    try:
        fichier_deb2=fichier_deb[0].split("_")
        nom_table=fichier_deb2[0]
    except:
        nom_table=fichier_deb[0]
        
    print("Données du fichier:", df_trs.shape)
    
    if check_table(nom_table):
        manage_table(df_trs,nom_table,'update')
    else:
        manage_table(df_trs,nom_table,'create')
        manage_table(df_trs,nom_table,'update')

cursor.close()
conn.close()
print("Exécution de Load_csv_to_sql.py terminée avec succés.")
print()