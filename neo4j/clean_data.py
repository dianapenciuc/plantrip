import pandas as pd
import re

def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def get_categories(text):
    text.replace('PointOfInterest','')
    raw_list = re.findall(r'\#((\w)+)\|', text)
    categ_list = [category[0] for category in raw_list]
    return ' '.join(categ_list)

def get_statistics(df,columns=[]):
    print(df.head())
    print(df.columns)
    print(df.isna().sum())
    print(df.shape)
    for col in columns:
        print(df[col].value_counts())

def get_city_ranking(cities):
    city_rank = cities.value_counts()
    city_rank = city_rank.rename(lambda index: index.lower())
    df_city_ranks = pd.DataFrame(city_rank)
    df_city_ranks.reset_index(inplace=True)
    df_city_ranks.columns = ['city','rank']
    df_city_ranks.head()
    return df_city_ranks

def get_best_places(df):
    city_rank = df["city"].value_counts()
    city_rank.sort_values(inplace=True,ascending=False)
    return city_rank[0:20]

#In a future implementation we will read the file directly from our source by webscraping
def process_datatourisme(filename):
    df = pd.read_csv(filename)
    df["Categories_de_POI"] = df["Categories_de_POI"].apply(lambda text: get_categories(text))
    df["Code_postal"] = df["Code_postal_et_commune"].apply(lambda text: text.split("#")[0])
    df["Commune"] = df["Code_postal_et_commune"].apply(lambda text: text.split("#")[1])
    graph_columns = ['Nom_du_POI','Code_postal','Commune','Categories_de_POI','Latitude','Longitude','URI_ID_du_POI','Description']
    df_graph = df[graph_columns]
    cat_cols = ['Nom_du_POI','Commune','Categories_de_POI','Description']
    #put categorical columns to lowercase to standardize representation between datasets
    for col in cat_cols:
        df_graph.loc[:,col] = df_graph[col].str.lower()
    print(df_graph.head())
    print("Categories de POI")
    print(df_graph['Categories_de_POI'].value_counts())
    df_city_ranks = get_city_ranking(df['Code_postal_et_commune'])
    df_graph.to_csv("graph_reg_idf.csv",index=False)
    return df_city_ranks
    
def process_cities(filename):
    df = pd.read_csv(filename)
    df = df.drop_duplicates()
    df['city'] = cities['zip_code'].astype(str) + "#" + cities['label']
    return df.dropna()

def process_cities_region(filename):
    df = pd.read_csv("best_idf.csv",sep=";",on_bad_lines='warn')
    #Some statistics
    print("Thèmes :")
    print(df["typo_niv3"].unique())
    df['city'] = df['nomcom'].str.lower()
    #Because Paris has "arrondissements" like "paris 01" I want a uniform representation
    #to compute the ranking for 'paris'
    df['city'] = df['city'].apply(lambda x: 'paris' if 'paris' in str(x) else x)
    df['city']
    city_rank = df["city"].value_counts()
    city_rank.sort_values(inplace=True,ascending=False)
    #print(city_rank)
    norm_arr = normalize(city_rank.values)
    city_ranks = pd.Series(norm_arr,index=city_rank.index)
    return city_ranks
    
def set_city_rankings(cities,df_city_ranks):
    cities = cities.merge(right=df_city_ranks,on="city",how='left')
    cities.fillna({"rank":0},inplace=True)
    return cities

def save_cities_ranking(cities,filename="cities_ranking.csv"):
    cities.to_csv(filename)
    
if __name__ == '__main__':
    cities_idf_filename = "best_idf.csv"
    cities = pd.read_csv("../../cours_ml/cities_clean.csv")
    df_city_ranks = process_datatourisme("datatourisme-reg-idf.csv")
    set_city_rankings(cities,df_city_ranks)
    cities.head()
    
    
   
    
