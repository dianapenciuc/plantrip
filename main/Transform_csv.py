#!/usr/bin/env python3

#Librairies utiles pour le script
from os import listdir
from os.path import isfile, join
import re
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup as bs
from unidecode import unidecode

# Définition des headers pour le webscrapping 
headers={ 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',}

# Définitions d'un dictionnaire de région pour ajouter une colonne à notre fichier de sortie
REGIONS = {
            'Auvergne-Rhône-Alpes': ['01', '03', '07', '15', '26', '38', '42', '43', '63', '69', '73', '74'],
            'Bourgogne-Franche-Comté': ['21', '25', '39', '58', '70', '71', '89', '90'],
            'Bretagne': ['35', '22', '56', '29'],
            'Centre-Val de Loire': ['18', '28', '36', '37', '41', '45'],
            'Corse': ['2A', '2B'],
            'Grand Est': ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88'],
            'Guadeloupe': ['971'],
            'Guyane': ['973'],
            'Hauts-de-France': ['02', '59', '60', '62', '80'],
            'Île-de-France': ['75', '77', '78', '91', '92', '93', '94', '95'],
            'La Réunion': ['974'],
            'Martinique': ['972'],
            'Normandie': ['14', '27', '50', '61', '76'],
            'Nouvelle-Aquitaine': ['16', '17', '19', '23', '24', '33', '40', '47', '64', '79', '86', '87'],
            'Occitanie': ['09', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82'],
            'Pays de la Loire': ['44', '49', '53', '72', '85'],
            'Provence-Alpes-Côte d\'Azur': ['04', '05', '06', '13', '83', '84'],
            'Terres australes et antarctiques françaises' : ['984'],
            'Wallis-et-Futuna' : ['986'],
            'Polynésie française' : ['987'],
            'Nouvelle-Calédonie' : ['988'],
            'Île de Clipperton' : ['989']
        }

# Définition du web scrapping sur les données de géoloc
def web_scrapping_geoloc(dataframe):
    resultat = dataframe[dataframe['latitude'].str.contains('data:', na=False)]
    cpt_max=resultat.shape[0]
    count=0
    for index in resultat.index:
        value=dataframe.loc[index]['latitude']
        # Etape de webscrapping sur les données de géolocalisation
        value_tab=[]                                                        # on la formate de telle façon à pouvoir l'utiliser
        value_tab=value.split(':')
        url="https://data.datatourisme.fr/"+value_tab[1]                    # en tant que partie d'url (car geo peut contenir un morceau d'url pointant vers les infos manquantes)
        count+=1
        time.sleep(0.4)
        page=requests.get(url, headers=headers, timeout=60)                  # on va donc demander la page
        bspage=bs(page.content,'html.parser')                                # en faire une Beautiful Soup
        lonlat=bspage.find('div', class_="col-sm-6 p-l-10 p-r-10 list-group")# et aller chercher notre composante avec la latitude et la longitude
        if lonlat:                                                           #si on la trouve
            lonlat_clean=lonlat.find('div').text.strip()                     # on traite l'information pour au final avoir la lat et la lon
            lonlat_tab=str(lonlat_clean).split('#')
            lat=lonlat_tab[0]
            lon=lonlat_tab[1]
        dataframe.loc[index,'latitude']=lat
        dataframe.loc[index,'longitude']=lon
    print(f'Compteur scrapping géolocalisation: {count}/{cpt_max}')
    return dataframe

# Définition du web scrapping sur les données d'adresse
def web_scrapping_address(dataframe):
    resultat = dataframe[dataframe['postalcode'].str.contains('data:', na=False)]
    cpt_max=resultat.shape[0]
    count=0
    for index in resultat.index:
        value=dataframe.loc[index]['postalcode']
        # Etape de webscrapping sur les données de géolocalisation
        value_tab=[]                                                        # on la formate de telle façon à pouvoir l'utiliser
        value_tab=value.split(':')
        url="https://data.datatourisme.fr/"+value_tab[1]                    # en tant que partie d'url (car addrr peut contenir un morceau d'url pointant vers les infos manquantes)
        count+=1
        time.sleep(0.4)
        page=requests.get(url, headers=headers, timeout=60)                  # on va donc demander la page
        bspage=bs(page.content,'html.parser')                               # en faire une Beautiful Soup
        adresse1=bspage.find('div', class_="prop-list list-group-item no-border")
        adresse2=bspage.find('div', class_="prop-list list-group-item")
        if adresse1 != None and adresse2 != None:
            adresse1=adresse1.text.strip()
            adresse2=adresse2.text.strip()
            adresse=bspage.find_all('div', class_="col-sm-6 p-l-10 p-r-10")           # et aller chercher notre composante avec l'adresse
            codepostal=adresse[3].text.strip()
            ville=adresse[1].text.strip()
            ville=ville.replace("Français","")
            ville=ville.replace("(France)","")
            ville=ville.strip()
            #print(f'Compteur scrapping adresse: {count}/{cpt_max} : {adresse1},{adresse2} {codepostal} {ville}', end='\r') 
            dataframe.loc[index,'street']=adresse1+"/"+adresse2
        else:
            adresse=bspage.find_all('div', class_="col-sm-6 p-l-10 p-r-10")           # et aller chercher notre composante avec l'adresse
            if len(adresse) > 0:
                try:
                    codepostal=adresse[3].text.strip()
                    ville=adresse[7].text.strip()
                    adresse0=adresse[5].text.strip()
                    dataframe.loc[index,'street']=adresse0
                    #print(f'Compteur scrapping adresse: {count}/{cpt_max} : {adresse0} {codepostal} {ville}',end='\r')
                except:
                    codepostal=adresse[3].text.strip()
                    ville=adresse[1].text.strip()
                    ville=ville.replace("Français","")
                    ville=ville.replace("(France)","")
                    ville=ville.strip()
                    #print(f'Compteur scrapping adresse: {count}/{cpt_max} : no street adress, {codepostal} {ville}', end='\r')
        dataframe.loc[index,'postalcode']=codepostal
        dataframe.loc[index,'city']=ville
    (f'Compteur scrapping adresse: {count}/{cpt_max}')
    return dataframe

# Définition d'une fonction d'harmonisation du code postal
def postalcode_rework(dataframe):
    df=dataframe
    cpt_max=df.shape[0]
    count=0
    for index in df.index:
        count+=1
        cp=df.loc[index]['postalcode']
        cp=str(cp)
        if "." in cp:
            cp_tab=cp.split(".")
            cp=cp_tab[0]
        pattern= r'^\d{5}$'     
        if cp != None and cp != "nan" and "data" not in cp and re.match(pattern, cp):
            cp1=re.sub(r'[^\d]+', '', cp)
            cp1=cp1.replace(" ","")
            df.loc[index]['postalcode']=cp1
    print(f'Compteur rework code postal: {count}/{cpt_max} : {cp} => {cp1}', end='\r') 
    return df
        
# Définition de la fonction d'implémentation de la colonne région en se basant sur le code postal
def region(dataframe):
    resultat = dataframe[(dataframe['region'].isna()) & (dataframe['postalcode'].notna())]
    cpt_max=resultat.shape[0]
    count=0
    for index in resultat.index:
        cp=dataframe.loc[index]['postalcode']  
        region=None
        if cp != None and cp != "nan" and "data" not in cp :
            cp=re.sub(r'[^\d]+', '', cp)
            cp=cp.replace(" ","")
            if type(cp) is str:
                cp=int(cp)
            if type(cp) is int:
                if cp < 100 :
                    value=str(cp)
                    region_list= [reg for reg, code in REGIONS.items() if value in code]
                    region=region_list[0]
                elif cp > 99999:
                    cp=cp//10
                elif cp > 97000:
                    value=cp//100
                    value=str(value)
                    region_list= [reg for reg, code in REGIONS.items() if value in code]
                    region=region_list[0]
                else:
                    value=cp//1000
                    if value != 20:
                        value=str(value)
                        if len(value) ==1:
                            value='0'+value
                        region_list= [reg for reg, code in REGIONS.items() if value in code]
                        region=region_list[0]
                    else:
                        region="Corse"
        dataframe.loc[index,'region']=region
    print(f'Compteur régions: {count}/{cpt_max} : {cp} {region}', end='\r')
    return dataframe

# Fonction pour nettoyer les caractères spéciaux
def nettoyer_chaine(chaine):
    if type(chaine) == str:
        chaine=unidecode(chaine)
        return re.sub(r'[^a-zA-Z0-9\s]', '', chaine)
    else :
        return chaine

### Script principal
## Fera une first task si les fichiers de sortie n'ont jamais été créés, sinon fera une update de ces fichiers
# Récupération du nom des différents fichiers dans le dossier data\csv\raw_extract
fichiers = [f for f in listdir("/projet/data/csv/raw_extract/") if isfile(join("/projet/data/csv/raw_extract/", f))]

# Récupération 
for fichier in fichiers:
    print("Ouverture du fichier : ",fichier)
    path_raw="/projet/data/csv/raw_extract/"+fichier
    df_raw=pd.read_csv(path_raw, sep=';', encoding='utf-8', low_memory=False)
    path1="/projet/data/csv/transformed/previous_versions/"+fichier
    print("Recherche du fichier",path1)
    if isfile(path1):
        print("Le fichier existe déjà, les données seront mises à jour.")
        df_file=pd.read_csv(path1, sep=';', encoding='utf-8', low_memory=False)
        df_to_add= df_raw[~df_raw['id'].isin(df_file['id'])]
        df_filtered = df_file[df_file['id'].isin(df_raw['id'])]
        df_data = pd.concat([df_filtered, df_to_add], ignore_index=True)
    else:
        print("Ce fichier n'existe pas, il sera créé.")
        df_data=df_raw
    print(df_data.shape)
    
    # Nombre de lignes dont le postalcode contient data et qu'il va falloir scrapper
    df_data['postalcode'] = df_data['postalcode'].astype(str)
    print("Données de code postal à scrapper:")
    print(df_data.loc[df_data['postalcode'].str.contains('data:', na=False),["postalcode"]].count())
    # Nombre de lignes dont latitude contient data et qu'il va falloir scrapper
    df_data['latitude'] = df_data['latitude'].astype(str)
    print("Données de géolocalisation à scrapper:")
    print(df_data.loc[df_data['latitude'].str.contains('data:', na=False),["latitude"]].count())
    
    # Lancement des scrapping successifs
    web_scrapping_geoloc(df_data)
    print()
    web_scrapping_address(df_data)
    print()
    
    # Ajustement des code postaux
    postalcode_rework(df_data)
    print()
    
    # Ajout de la région
    region(df_data)
    print()
    
    # Nettoyage des caractères spéciaux
    df_data['label'] = df_data['label'].apply(nettoyer_chaine)
    df_data['street'] = df_data['street'].apply(nettoyer_chaine)
    
    # Affichage des informations finales
    print(df_data[(df_data['latitude'].isna()) & (df_data['longitude'].isna()) & (df_data['city'].isna()) & (df_data['postalcode'].isna())].shape)
    
    path2="/projet/data/csv/transformed/"+fichier
    with open(path2,'w',newline='', encoding="UTF-8", errors='replace') as csvfile:
        df_data.to_csv(csvfile, index=False, sep=';')
        csvfile.close()
print("Exécution de Transform.py terminée avec succés.")
print()