# %%
#!/usr/bin/env python3
#Librairies utiles pour le script
from os import listdir
from os.path import isfile, join
import pyodbc
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
    'object' :'NVARCHAR(700)',
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
    df = df.fillna(value = None, method='ffill')
    df = df.fillna(value = None, method='bfill')

    # Check what Data already is in the table with id
    check_query = f"SELECT id FROM {nom}"
    existing_ids = pd.read_sql(check_query, conn)
    new_df = df[~df['id'].isin(existing_ids['id'])]

    # Insérer les données du DataFrame dans la table
    for index, row in new_df.iterrows():
        insert_query = f'''
        INSERT INTO {nom} (id,label,type,themes,startdate,enddate,street,postalcode,city,insee,region,latitude,longitude,email,web,tel,lastupdate)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        '''

        print(row)
        cursor.execute(insert_query, row['id'],row['label'],row['type'],row['themes'],row['startdate'],row['enddate'],row['street'],row['postalcode'],row['city'],row['insee'],row['region'],
            row['latitude'],row['longitude'],row['email'],row['web'],row['tel'],row['lastupdate'])
        conn.commit()

### Script principal
## Fera une first task si les fichiers de sortie n'ont jamais été créés, sinon fera une update de ces fichiers
# Connexion au server SQL qui doit être activé

conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=MSI-ROUDEL;"
    "Database=itineraire_test;"
    "Trusted_Connection=yes;"
)
cursor = conn.cursor()

# Récupération du nom des différents fichiers dans le dossier data\csv\raw_extract
fichiers = [f for f in listdir(".\\data\\csv\\transformed\\") if isfile(join(".\\data\\csv\\transformed\\", f))]

for fichier in fichiers:
    path_trs=".\\data\\csv\\transformed\\"+fichier
    df_trs=pd.read_csv(path_trs, sep=';', low_memory=False)
    df_trs = df_trs.where(pd.notnull(df_trs), None)

    fichier_deb=fichier.split(".")
    nom_table=fichier_deb[0]

    if check_table(nom_table):
        manage_table(df_trs,nom_table,'update')
    else:
        manage_table(df_trs,nom_table,'create')
        manage_table(df_trs,nom_table,'update')

cursor.close()
conn.close()