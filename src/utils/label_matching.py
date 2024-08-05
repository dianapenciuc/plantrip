import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_best_match(data, user_labels):
    user_labels_lower = [label.lower() for label in user_labels]

    data['Label_lower'] = data['Label'].str.lower()
    all_labels = data['Label_lower'].tolist()
    all_labels.extend(user_labels_lower)

    vectorizer = TfidfVectorizer().fit_transform(all_labels)
    vectors = vectorizer.toarray()

    cosine_similarities = cosine_similarity(vectors[-len(user_labels_lower):], vectors[:-len(user_labels_lower)])

    best_matches = cosine_similarities.argmax(axis=1)
    best_match_labels = [data['Label'].iloc[i] for i in best_matches]

    matching_observations = data.iloc[best_matches].drop(columns=['Label_lower'])

    return matching_observations
