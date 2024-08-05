import pandas as pd
import re

df = pd.read_csv('../data/output_with_clusters_2.csv')
classes_fr = pd.read_csv('../data/classes_fr.csv')

def clean_classes_fr(df):
    for col in df.columns:
        df[col] = df[col].str.replace('https://www.datatourisme.fr/ontology/core#', '', regex=False)
        df[col] = df[col].str.replace('<', '').str.replace('>', '')
    return df

def clean_data_idf(df):
    df['EventType'] = df['EventType'].str.replace('schema:', '', regex=False)
    df['EventType'] = df['EventType'].str.replace('PointOfInterest', '', regex=False)
    df['EventType'] = df['EventType'].str.replace('olo:OrderedList', '', regex=False)
    return df

def replace_event_types(event_types, mapping):
    for eng_term, fr_term in mapping.items():
        pattern = r'\b' + re.escape(eng_term) + r'\b'
        event_types = event_types.str.replace(pattern, fr_term, regex=True)
    return event_types

classes_fr_cleaned = clean_classes_fr(classes_fr)
data_cleaned = clean_data_idf(df)

mapping = pd.Series(classes_fr_cleaned.iloc[:, 1].values, index=classes_fr_cleaned.iloc[:, 0]).to_dict()

data_cleaned['EventType'] = replace_event_types(data_cleaned['EventType'], mapping)

cat = {
    "Lieu": [
        "Commerce de détail",
        "Site naturel",
        "Site sportif, récréatif et de loisirs",
        "Service d'information touristique",
        "Hébergement",
        "Site d'affaires",
        "Prestataire d'activité",
        "Lieu de santé",
        "Fournisseur de dégustation",
        "Service pratique",
        "Transport",
        "Site culturel",
        "Restauration",
        "Prestataire de service"
    ],
    "Itinéraire touristique": [
        "Itinéraire sous-marin",
        "Itinéraire fluvial ou maritime",
        "Itinéraire cyclable",
        "Itinéraire pédestre",
        "Itinéraire routier",
        "Itinéraire équestre"
    ],
    "Fête et manifestation": [
        "Évènement sports et loisirs",
        "Evènement professionnel d'entreprise",
        "Evènement social",
        "Évènement culturel",
        "Evènement commercial"
    ],
    "Produit": [
        "Visite",
        "Produit d'hébergement",
        "Pratique",
        "Location"
    ]
}

category_columns = []
for category_list in cat.values():
    for category in category_list:
        normalized_category = " ".join(category.lower().strip().split())
        data_cleaned[normalized_category] = data_cleaned['EventType'].apply(lambda x: 1 if normalized_category in " ".join(x.lower().strip().split()) else 0)

        category_columns.append(normalized_category)

for key in cat.keys():
    normalized_category = " ".join(key.lower().strip().split())
    data_cleaned[normalized_category] = data_cleaned['EventType'].apply(lambda x: 1 if normalized_category in " ".join(x.lower().strip().split()) else 0)

    category_columns.append(normalized_category)


data_idf_with_indicators = data_cleaned.copy()
print(data_idf_with_indicators.columns)

data_idf_with_indicators.to_csv('../data/data_w_indicators.csv', index=False)

def verify_categories_in_columns(cat, df):
    missing_elements = []
    for category in category_columns:
        if category not in df.columns:
            missing_elements.append(category)
    if missing_elements:
        print(f"Missing elements: {missing_elements}")
        return False
    else:
        return True

if verify_categories_in_columns(cat, data_idf_with_indicators):
    print("All elements are present in the dataframe columns.")
else:
    print("Some elements are missing from the dataframe columns.")

print(data_idf_with_indicators.columns)
print(data_idf_with_indicators['site sportif, récréatif et de loisirs'].nunique())
