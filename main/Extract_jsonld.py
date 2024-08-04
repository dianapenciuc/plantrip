#!/usr/bin/env python3
#################################################################################################################################
# Script permettant la création de fichiers CSV à partir de fichiers JSONLD, téléchargés via le webservice de  sur DATATourisme #
#################################################################################################################################
## 1- Les fichiers sont pré-téléchargés via un CronTab au format JSONLD. Ce sont des fichiers basés sur le principe d'une condensation de multiples fichiers JSON en un graph qui respecte le format JSON.
## 2- Pour chaque fichier dans le dossier data/jsonld, un fichier csv sera créé ou mis à jour en se basant sur le nom (à un fichier lieux.jsonld correspondra un fichier lieux.csv)

### Importations des librairies utiles pour le script
import json
import re
import pandas as pd
from os import listdir
from os.path import isfile, join
from unidecode import unidecode

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

# Récupération du nom des différents fichiers dans le dossier de stockage, ce dossier doit être au même endroit que ce code python
fichiers = [f for f in listdir("/projet/data/jsonld/") if isfile(join("/projet/data/jsonld/", f))]

### Script principal
for fichier in fichiers:                            # Pour chaque fichier se trouvant dans le dossier :
    fichier_deb=fichier.split(".")
    nom_fichier_date=fichier_deb[0]   # On va créer un fichier csv de sortie comportant le même nom (pour le reconnaître facilement)
    fichier_deb=nom_fichier_date.split("-")
    nom_fichier=fichier_deb[0]+".csv" 
    path="/projet/data/csv/raw_extract/"+nom_fichier
    print("Ouverture du fichier :",fichier)
    #Initialisation d'une liste qui contiendra les données récupérées de chaque POI (sous forme d'une ligne/une entrée par POI)
    extraction=[]

    # Ouverture du fichier JSONLD en entrée
    path2="/projet/data/jsonld/"+fichier
    with open(path2, encoding="UTF-8") as json_file:
        print("Extraction des données en cours : Merci de patienter...")
        data = json.load(json_file)
        count=0
        for e in data['@graph']:
            # extraction des différenst éléments voulus dans le graphe
            uri = ldget(e, ['@id'])
            label = ldget(e, ['rdfs:label','@value'])
            comment = ldget(e, ['rdfs:comment','@value'])
            comment=str(comment)
            comment=unidecode(comment)
            comment=re.sub(r'[^a-zA-Z0-9\s]', '', comment)
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
            if cp == "":
                try :
                    cp = addr['@id']
                except:
                    cp = None
            # Récupération de la ville car elle peut se trouver à deux endroits différents selon comment l'utilisateur a rentré les informations
            city = ldget(addr, ['hasAddressCity', 'rdfs:label'])
            if city:
                city = str(city['@value'])
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
                event_type = None
            if event_type == "":
                event_type = None
            email = ldget(e, ['hasContact', 'schema:email'])
            web = ldget(e, ['hasContact', 'foaf:homepage'])
            tel = ldget(e, ['hasContact', 'schema:telephone'])
            # Récupération des thèmes : organisation particulière nécéssitant de fouiller car cet étage peut prendre différentes formes
            themes = ldget(e, ['hasTheme'], None)
            event_theme = ""
            if themes:
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
                    event_theme = None
            else:
                event_theme = None

            # Rassemblement des informations en une ligne, et vérification si ce POI existe déjà dans notre fichier csv (si ce dernier existe déjà)
            row={
                'id':uri,'label':label,'type':event_type,'themes':event_theme,
                'startdate':startdate, 'enddate': enddate,
                'street': street,'postalcode' : cp,'city' : city, 'insee' : insee, 'region' : None,
                'latitude': lat,'longitude' : lon,
                'email' : email,'web' : web,'tel': tel,
                'lastupdate':last_update,'comment': comment
                }
            extraction.append(row)
        json_file.close()
                
    # Création des colonnes, avec utilisation des DataFrames pour afficher les données manquantes et finalement, écrire les fichiers csv de sortie:
    # Colonnes :
    fields = ['id', 'label', 'type', 'themes', 'startdate', 'enddate',
            'street', 'postalcode', 'city', 'insee', 'region',
            'latitude', 'longitude', 'email', 'web', 'tel',
            'lastupdate', 'comment']
            
    # Dataframe de l'extraction:
    df_extr=pd.DataFrame(extraction, columns=fields)
    print("Nombre de données ajoutées :",df_extr.shape)
    print("Voici les informations concernant les données ajoutées au fichier:",path)
    print(df_extr.count())
    print()
    print("et les informations concernant les données manquantes:")
    print(df_extr.isna().sum())
    print()
        
    # écriture dans le fichier CSV
    with open(re.sub(r'json.*', 'csv', path), "w", newline='', encoding="UTF-8", errors='replace') as csvfile:
        df_extr.to_csv(csvfile, index=False, sep=';')
        csvfile.close()
        
print("Exécution de Extract_jsonld.py terminée avec succés.")
print()