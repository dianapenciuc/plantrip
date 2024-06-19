#!/usr/bin/env python3
#######################################################################################################################
# Script permettant la création de fichiers CSV à partir de fichiers JSONLD, téléchargés via des flux sur DATATourisme #
#######################################################################################################################

#Importations des librairies utiles
import sys
import json
import csv
import re
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

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

# Récupération du nom des différents fichiers du dossier de stockage, ce dossier doit être au même endroit que ce code python  => A FINALISER
fichiers = [f for f in listdir(".\\jsonld\\") if isfile(join(".\\jsonld\\", f))]
i=0

for fichier in fichiers:
    i+=1
    fichier_deb=fichier.split(".")
    nom_fichier=fichier_deb[0]+"_"+str(i)+".csv"
    
# Ouverture du fichier CSV de sortie
    with open(re.sub(r'json.*', 'csv', nom_fichier), 'w', newline='', encoding="UTF-8" ) as csvfile:
        fields = ['id', 'label', 'type', 'theme', 'startdate', 'enddate',
                'street', 'postalcode', 'city', 'insee',
                'latitude', 'longitude', 'email', 'web', 'tel',
                'lastupdate', 'comment']
        
    #Initialisation d'une liste qui contiendra les données récupérées de chaque POI (sous forme d'une ligne/une entrée par POI)
        extraction=[]

        # Ouverture du fichier JSONLD en entrée
        path="jsonld\\"+fichier
        with open(path, encoding="UTF-8") as json_file:
            data = json.load(json_file)
            for e in data['@graph']:
                # extraction des différenst éléments voulus dans le graphe
                uri = ldget(e, ['@id'])
                label = ldget(e, ['rdfs:label','@value'])
                comment = ldget(e, ['rdfs:comment','@value']) 
                startdate = ldget(e, ['takesPlaceAt', 'startDate', '@value'])
                enddate = ldget(e, ['takesPlaceAt', 'endDate', '@value'])
                geo = ldget(e, ['isLocatedAt', 'schema:geo'])
                lat = ldget(geo, ['schema:latitude', '@value'])
                lon = ldget(geo, ['schema:longitude', '@value'])
                addr = ldget(e, ['isLocatedAt', 'schema:address'])
                street = ldget(addr, ['schema:streetAddress'])
                cp = ldget(addr, ['schema:postalCode'], '')
                city = ldget(addr, ['hasAddressCity', 'rdfs:label'])
                if city:
                    city = city['@value']
                if city == None:
                    city = ldget(addr, ['schema:addressLocality'])
                insee = ldget(addr, ['hasAddressCity', 'insee'])
                last_update = ldget(e, ['lastUpdate',"@value"])
                event_type = '/'.join(ldget(e, ['@type']))
                email = ldget(e, ['hasContact', 'schema:email'])
                web = ldget(e, ['hasContact', 'foaf:homepage'])
                tel = ldget(e, ['hasContact', 'schema:telephone'])

                themes = ldget(e, ['hasTheme', 'rdfs:label'], None)
                event_theme = ''
                if themes:
                    if themes['@language'] == 'fr':
                        event_theme = event_theme + themes['@value'] + ', '
                    event_theme = event_theme[:-2]

                # Corrections sur les informations
                row={'id':uri, 
                    'label':label,
                    'type':event_type,
                    'theme':event_theme,
                    'startdate':startdate, 
                    'enddate': enddate,
                    'street': street, 
                    'postalcode' : cp, 
                    'city' : city, 
                    'insee' : insee,
                    'latitude': lat,
                    'longitude' : lon,
                    'email' : email,
                    'web' : web,
                    'tel': tel,
                    'lastupdate':last_update,
                    'comment': comment}
                extraction.append(row)
            
            # Utilisation des DataFrames et affichage des données manquantes:
            df=pd.DataFrame(extraction, columns=fields)
            print(df.head(20))
            print(df.count())
            print(df.isna().sum())
            
            # écriture dans le fichier CSV
            df.to_csv(csvfile, index=False, sep=';')
            json_file.close()
            csvfile.close()