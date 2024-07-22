#!/usr/bin/env python3
#################################################################################################################################
# Script permettant la création de fichiers CSV à partir de fichiers JSONLD, téléchargés via le webservice de  sur DATATourisme #
#################################################################################################################################
## 1- Les fichiers sont pré-téléchargés via un CronTab au format JSONLD. Ce sont des fichiers basés sur le principe d'une condensation de multiples fichiers JSON en un graph qui respecte le format JSON.
## 2- Pour chaque fichier dans le dossier data/jsonld, un fichier csv sera créé ou mis à jour en se basant sur le nom (à un fichier lieux.jsonld correspondra un fichier lieux.csv)
## 3- Une colonne région sera rajoutée en se basant sur le code postal qui est systématiquement renseigné dans les données DataTourisme
## 4- Certaines données de latitude et longitude sont représentées sous la forme d'un morceau d'URL, un webscrapping est effectué afin de les récupérer au même format que les autres données WGS84

### Importations des librairies utiles pour le script
import json
import re
import pandas as pd
from os import listdir
from os.path import isfile, join

# Définition des headers pour le webscrapping 
headers={ 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',}

# Définition d'une méthode de descente dans le graph de données d'un fichier JSONLD
def ldget(obj, path, default=None):
    "Descente dans le graphe de données"
    if obj is not None and path[0] in obj:
        if len(path) == 1:
            return obj[path[0]]
        else:
            return ldget(obj[path[0]], path[1:], default=default)
    else:
        return default

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

# Récupération du nom des différents fichiers dans le dossier de stockage, ce dossier doit être au même endroit que ce code python
fichiers = [f for f in listdir("C:\\Users\\roude\\Documents\\itineraire\\data\\jsonld\\") if isfile(join("C:\\Users\\roude\\Documents\\itineraire\\data\\jsonld\\", f))]

### Script principal
for fichier in fichiers:                            # Pour chaque fichier se trouvant dans le dossier :
    fichier_deb=fichier.split(".")
    nom_fichier_date=fichier_deb[0]   # On va créer un fichier csv de sortie comportant le même nom (pour le reconnaître facilement)
    fichier_deb=nom_fichier_date.split("-")
    nom_fichier=fichier_deb[0]+".csv" 

    # Ouverture du fichier CSV de sortie, s'il existe déjà, on passe en mode a+, s'il n'existe pas, on passe en mode w
    path="C:\\Users\\roude\\Documents\\itineraire\\data\\csv\\raw_extract\\"+nom_fichier
    print("Recherche du fichier",path)
    if isfile(path):
        print("Ce fichier existe déjà, il subira une update")
        mode="a+"
        df_file=pd.read_csv(path, sep=';', usecols=[0])
    else:
        print("Ce fichier n'existe pas, il sera créé.")
        mode="w"
    with open(re.sub(r'json.*', 'csv', path), mode, newline='', encoding="UTF-8" ) as csvfile:
        
    #Initialisation d'une liste qui contiendra les données récupérées de chaque POI (sous forme d'une ligne/une entrée par POI)
        extraction=[]

        # Ouverture du fichier JSONLD en entrée
        path2="C:\\Users\\roude\\Documents\\itineraire\\data\\jsonld\\"+fichier
        with open(path2, encoding="UTF-8") as json_file:
            data = json.load(json_file)
            count=0
            for e in data['@graph']:
                # extraction des différenst éléments voulus dans le graphe
                uri = ldget(e, ['@id'])
                if mode=="a+" and df_file['id'].isin([uri]).any():
                    continue
                label = ldget(e, ['rdfs:label','@value'])
                comment = ldget(e, ['rdfs:comment','@value']) 
                startdate = ldget(e, ['takesPlaceAt', 'startDate', '@value'])
                enddate = ldget(e, ['takesPlaceAt', 'endDate', '@value'])
                geo = ldget(e, ['isLocatedAt', 'schema:geo'])
                lat = ldget(geo, ['schema:latitude', '@value'])
                lon = ldget(geo, ['schema:longitude', '@value'])
                # Si on n'a pas de latitude et de longitude, il se peut qu'on ait quand même une information dans geo
                if lat == None and lon == None:
                    if geo != None:                                     # on regarde donc si on a une info dans geo, si oui :
                        value=str(geo['@id'])                           # on recupère cette valeur, elle nous servira à faire du webscrapping 
                        lat=value
                addr = ldget(e, ['isLocatedAt', 'schema:address'])
                street = ldget(addr, ['schema:streetAddress'])
                cp = ldget(addr, ['schema:postalCode'], '')
                # Récupération de la ville car elle peut se trouver à deux endroits différents selon comment l'utilisateur a rentré les informations
                city = ldget(addr, ['hasAddressCity', 'rdfs:label'])
                if city:
                    city = city['@value']
                if city == None:
                    city = ldget(addr, ['schema:addressLocality'])
                insee = ldget(addr, ['hasAddressCity', 'insee'])
                last_update = ldget(e, ['lastUpdate',"@value"])
                # Traintement particulier des types d'évènements car beaucoup de redondance dans les données due à une structuration particulière
                event_type = '/'.join(ldget(e, ['@type']))
                event_types=event_type.split('/')
                
                for mot in ['schema','PointOfInterest','urn:resource','olo:OrderedList','Product']:
                    event_types = [valeur for valeur in event_types if mot not in valeur]
                if "Tour" in event_types:
                    event_types.remove("Tour")
                if "Product" in event_types:
                    event_types.remove("Product")
                event_type='/'.join(event_types)
                if event_type == "u/r/n/:/r/e/s/o/u/r/c/e":
                    event_type = "Notype"
                email = ldget(e, ['hasContact', 'schema:email'])
                web = ldget(e, ['hasContact', 'foaf:homepage'])
                tel = ldget(e, ['hasContact', 'schema:telephone'])
                # Récupération des thèmes : organisation particulière nécéssitant de fouiller car cet étage peut prendre différentes formes
                themes = ldget(e, ['hasTheme', 'rdfs:label'], None)
                event_theme = ''
                if themes:
                    if themes['@language'] == 'fr':
                        event_theme = event_theme + themes['@value'] + '/'
                    event_theme = event_theme[:-2]
                if event_theme == '' :
                    themes= ldget(e,['hasTheme'])
                    if type(themes) is list:
                        for theme in themes:
                            event_theme_temp=theme['@id']
                            event_theme_temp=event_theme_temp[3:]
                            event_theme = event_theme + event_theme_temp + '/'
                        event_theme=event_theme[:-1]
                    elif type(themes) is dict:
                        event_theme=themes['@id']
                        event_theme=event_theme[3:]
                    else:
                        event_theme=None

                #Implémentation de la colonne région en se basant sur le code postal (celui-ci est toujours renseigné)
                region=''
                if cp != '':
                    cp=re.sub(r'[^\d]+', '', cp)
                    cp=cp.replace(" ","")
                    if type(cp) is str:
                        cp=int(cp)
                    if type(cp) is int:
                        if cp < 100 :
                            index=str(cp)
                            region_list= [reg for reg, code in REGIONS.items() if index in code]
                            region=region_list[0]
                        elif cp > 99999:
                            cp=cp//10
                        elif cp > 97000:
                            index=cp//100
                            index=str(index)
                            region_list= [reg for reg, code in REGIONS.items() if index in code]
                            region=region_list[0]
                        else:
                            index=cp//1000
                            if index != 20:
                                index=str(index)
                                if len(index) ==1:
                                    index='0'+index
                                region_list= [reg for reg, code in REGIONS.items() if index in code]
                                region=region_list[0]
                            else:
                                region="Corse"

                # Rassemblement des informations en une ligne, et vérification si ce POI existe déjà dans notre fichier csv (si ce dernier existe déjà)
                row={
                    'id':uri,'label':label,'type':event_type,'themes':event_theme,
                    'startdate':startdate, 'enddate': enddate,
                    'street': street,'postalcode' : cp,'city' : city, 'insee' : insee,'region' : region,
                    'latitude': lat,'longitude' : lon,
                    'email' : email,'web' : web,'tel': tel,
                    'lastupdate':last_update,'comment': comment
                    }
                extraction.append(row)
            
            # Création des colonnes, avec utilisation des DataFrames pour afficher les données manquantes et finalement, écrire les fichiers csv de sortie:
            # Colonnes :
            fields = ['id', 'label', 'type', 'theme', 'startdate', 'enddate',
                'street', 'postalcode', 'city', 'insee','region',
                'latitude', 'longitude', 'email', 'web', 'tel',
                'lastupdate', 'comment']
            
            # Dataframe de l'extraction:
            df_extr=pd.DataFrame(extraction, columns=fields)
            print("Voici le nombre de données qui seront ajoutées à la base de donnes:",df_extr.shape)
            print("Voici les informations concernant les données ajoutées au fichier:",path)
            print(df_extr.count())
            print("et les informations concernant les données manquantes:")
            print(df_extr.isna().sum())
                                    
            # écriture dans le fichier CSV
            df_extr.to_csv(csvfile, index=False, sep=';')
            json_file.close()
            csvfile.close()