#!/usr/bin/env python3

#Librairies utiles pour le script
from os import listdir
from os.path import isfile, join
import re
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup as bs

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
    resultat = dataframe[(dataframe['latitude'].notna()) & (dataframe['longitude'].isna())]
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
            lon=lonlat_tab[1]
            lat=lonlat_tab[0]
        #print(count,"/",cpt_max,"lat:",lat,"lon:",lon)
        print(f'Compteur géoloc: {count}/{cpt_max} : lat={lat}, lon={lon}', end='\r')
        dataframe.loc[index,'latitude']=lat
        dataframe.loc[index,'longitude']=lat
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
            #print(count,":",adresse1,adresse2,codepostal,ville)
            print(f'Compteur adresse: {count}/{cpt_max} : {adresse1},{adresse2} {codepostal} {ville}', end='\r') 
            dataframe.loc[index,'street']=adresse1+"/"+adresse2
        else:
            adresse=bspage.find_all('div', class_="col-sm-6 p-l-10 p-r-10")           # et aller chercher notre composante avec l'adresse
            codepostal=adresse[3].text.strip()
            try:
                ville=adresse[7].text.strip()
                adresse0=adresse[5].text.strip()
                dataframe.loc[index,'street']=adresse0
                #print(count,":",adresse0,codepostal,ville)
                print(f'Compteur adresse: {count}/{cpt_max} : {adresse0} {codepostal} {ville}', end='\r')
            except:
                ville=adresse[1].text.strip()
                ville=ville.replace("Français","")
                ville=ville.replace("(France)","")
                ville=ville.strip()
                #print(count,":","no street address",codepostal,ville)
                print(f'Compteur adresse: {count}/{cpt_max} : no street adress, {codepostal} {ville}', end='\r')
        dataframe.loc[index,'postalcode']=codepostal
        dataframe.loc[index,'city']=ville
    return dataframe

# Définition de la fonction d'implémentation de la colonne région en se basant sur le code postal
def region(dataframe):
    resultat = dataframe[(dataframe['region'].isna()) & (dataframe['postalcode'].notna())]
    cpt_max=resultat.shape[0]
    count=0
    for index in resultat.index:
        cp=dataframe.loc[index]['postalcode']  
        region=None
        if cp != None and "data" not in cp:
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

### Script principal
## Fera une first task si les fichiers de sortie n'ont jamais été créés, sinon fera une update de ces fichiers
# Récupération du nom des différents fichiers dans le dossier data\csv\raw_extract
fichiers = [f for f in listdir(".\\data\\csv\\raw_extract\\") if isfile(join(".\\data\\csv\\raw_extract\\", f))]

# Récupération 
for fichier in fichiers:
    path_raw=".\\data\\csv\\raw_extract\\"+fichier
    df_raw=pd.read_csv(path_raw, sep=';', low_memory=False)
    
    path1=".\\data\\csv\\transformed\\"+fichier
    print("Recherche du fichier",path1)
    if isfile(path1):
        print("Ce fichier existe déjà, il subira une update")
        mode="a+"
        df_file=pd.read_csv(path1, sep=';', usecols=[0],low_memory=False)
        df_data = df_raw[~df_raw['id'].isin(df_file['id'])]
    else:
        print("Ce fichier n'existe pas, il sera créé.")
        mode="w"
        df_data=df_raw
    print(df_data.shape)
    
    # Nombre de lignes dont le postalcode contient data et qu'il va falloir scrapper
    print(df_data.loc[df_data['postalcode'].str.contains('data:', na=False),["postalcode"]].count())
    # Nombre de lignes dont latitude contient data et qu'il va falloir scrapper
    print(df_data.loc[df_data['latitude'].str.contains('data:', na=False),["latitude"]].count())
    
    # Lancement des scrapping successifs
    #web_scrapping_geoloc(df_data)
    #web_scrapping_address(df_data)

    # Ajout de la région
    #region(df_data)

    # Affichage des informations finales
    print()
    print(df_data[(df_data['latitude'].isna()) & (df_data['longitude'].isna()) & (df_data['city'].isna()) & (df_data['postalcode'].isna())].shape)
    print(df_data.isnull().sum())

    with open(path1,mode,newline='', encoding="UTF-8" ) as csvfile:
        df_data.to_csv(csvfile, index=False, sep=';')
        csvfile.close()



