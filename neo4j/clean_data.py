import pandas as pd
import re

def get_categories(text):
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

#in a future implementation we will read the file directly from our source
def process_datatourisme(filename):
    df = pd.read_csv(filename)
    df["Categories_de_POI"] = df["Categories_de_POI"].apply(lambda text: get_categories(text))
    df["Code_postal"] = df["Code_postal_et_commune"].apply(lambda text: text.split("#")[0])
    df["Commune"] = df["Code_postal_et_commune"].apply(lambda text: text.split("#")[1])
    graph_columns = ['Nom_du_POI','Code_postal','Commune','Categories_de_POI','Latitude','Longitude','URI_ID_du_POI','Description']
    df_graph = df[graph_columns]
    cat_cols = ['Nom_du_POI','Commune','Categories_de_POI','Description']
    #put categorical columns to lowercase to standardize representations between datasets
    for col in cat_cols:
        df_graph[col] = df_graph[col].str.lower()
    print(df_graph.head())
    print("Categories de POI")
    print(df_graph['Categories_de_POI'].value_counts())
    df_graph.to_csv("graph-reg-idf.csv",index=False)
    
def process_cities(filename):
    df = pd.read_csv(filename)
    get_statistics(df)
    return df.dropna()

if __name__ == '__main__':
    process_datatourisme("datatourisme-reg-idf.csv")
    ''' df = process_cities("cities.csv")
    get_statistics(df)
    df.to_csv("cities_clean.csv") '''
