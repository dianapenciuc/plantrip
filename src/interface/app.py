import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import openrouteservice
import networkx as nx
import sys
import os
import folium
import tempfile
import cachetools
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.label_matching import find_best_match
from utils.cat import cat, poi

categories = cat
points_interet = poi

api_key = "5b3ce3597851110001cf62483a064f210f254131b3cb5b034c97d3a5"
client = openrouteservice.Client(key=api_key)

data = pd.read_csv("../data/data_w_indicators.csv")

cache = cachetools.TTLCache(maxsize=100, ttl=300) 

# Compteur global pour les requêtes API
api_request_count = 0

G = nx.Graph()

for idx, row in data.iterrows():
    G.add_node(row['Label'], pos=(row['Longitude'], row['Latitude']), cluster=row['Cluster'], comment=row.get('Comment', ''))

def ajouter_aretes_dans_cluster(graphe):
    noeuds = list(graphe.nodes(data=True))
    for i, (noeud1, data1) in enumerate(noeuds):
        for j, (noeud2, data2) in enumerate(noeuds):
            if i != j and data1['cluster'] == data2['cluster']:
                distance = ((data1['pos'][0] - data2['pos'][0])**2 + (data1['pos'][1] - data2['pos'][1])**2)**0.5
                graphe.add_edge(noeud1, noeud2, weight=distance)

ajouter_aretes_dans_cluster(G)

def generate_options_with_counts(categories, data, cluster_id=None):
    options = []
    if cluster_id is not None:
        cluster_data = data[data['Cluster'] == cluster_id]
    else:
        cluster_data = data
    for category, subcategories in categories.items():
        cat_key = category.lower().replace(' ', '_')
        cat_count = cluster_data[cat_key].sum() if cat_key in cluster_data.columns else 0
        options.append({"label": html.Span(f"{category} ({cat_count})", style={"font-weight": "bold"}), "value": cat_key})
        for subcategory in subcategories:
            subcat_value = subcategory.lower().replace(' ', '_')
            subcat_count = cluster_data[subcat_value].sum() if subcat_value in cluster_data.columns else 0
            options.append({"label": html.Span(f"  - {subcategory} ({subcat_count})", style={"margin-left": "20px"}), "value": subcat_value})
    return options

def tsp_voisin_le_plus_proche(graphe, point_depart, point_arrive):
    cluster = graphe.nodes[point_depart]['cluster']
    noeuds_cluster = [noeud for noeud, data in graphe.nodes(data=True) if data['cluster'] == cluster]
    chemin = [point_depart]
    noeuds_a_visiter = set(noeuds_cluster)
    noeuds_a_visiter.remove(point_depart)
    noeuds_a_visiter.remove(point_arrive)

    current_node = point_depart
    while noeuds_a_visiter:
        nearest_neighbor = min(noeuds_a_visiter, key=lambda node: graphe[current_node][node]['weight'])
        chemin.append(nearest_neighbor)
        noeuds_a_visiter.remove(nearest_neighbor)
        current_node = nearest_neighbor

    chemin.append(point_arrive)
    return chemin

def get_route(client, coord1, coord2):
    global api_request_count
    try:
        route = client.directions(coordinates=[coord1, coord2], profile='foot-walking', format='geojson')
        api_request_count += 1
        print(f"Nombre de requêtes API effectuées : {api_request_count}")
        return route['features'][0]['geometry']['coordinates']
    except Exception as e:
        print(f"Erreur lors de la requête ORS pour l'itinéraire : {e}")
        return []

def create_map_with_folium(chemin=None, positions_dict=None):
    center_location = [48.8566, 2.3522] if not chemin else positions_dict[chemin[0]]
    folium_map = folium.Map(location=center_location, zoom_start=13)
    
    if chemin and positions_dict:
        for i, label in enumerate(chemin):
            location = positions_dict[label]
            comment = data[data['Label'] == label]['Comment'].values[0] if 'Comment' in data.columns else ''
            folium.Marker(
                location=location,
                tooltip=f"{i+1}. {label}",
                popup=f"{i+1}. {label}<br>{comment}",
                icon=folium.DivIcon(html=f'<div style="font-size: 24pt; color: red;">{i+1}</div>')
            ).add_to(folium_map)

            if i > 0:
                prev_label = chemin[i-1]
                prev_location = positions_dict[prev_label]
                route = get_route(client, (prev_location[1], prev_location[0]), (location[1], location[0]))
                if route:
                    folium.PolyLine([(lat, lon) for lon, lat in route], color="blue").add_to(folium_map)
    
    return folium_map

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

initial_map = create_map_with_folium()

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Itinéraire de vacances", className="text-center text-primary mb-4"), width=12)
    ]),
    dcc.Store(id='selected-cluster', data=None),  # Store pour stocker le cluster sélectionné
    dcc.Store(id='positions-dict', data=None),  # Store pour stocker les positions des points
    dcc.Store(id='itineraries', data={}),  # Store pour stocker les itinéraires par jour
    dbc.Row([
        dbc.Col([
            html.Label('Jour', className="font-weight-bold"),
            dcc.Dropdown(
                id='day-dropdown',
                options=[{'label': f'Jour {i}', 'value': i} for i in range(1, 8)],  # Supposons une semaine de vacances
                placeholder='Sélectionner un jour',
                className="mb-3"
            ),
            html.Label('Points majeurs', className="font-weight-bold"),
            dcc.Dropdown(
                id='category-dropdown',
                options=[{'label': k, 'value': k} for k in points_interet.keys()],
                placeholder='Sélectionner une catégorie',
                className="mb-3"
            ),
            html.Label('Points d\'intérêt', className="font-weight-bold"),
            dcc.Dropdown(
                id='poi-dropdown',
                placeholder='Sélectionner un point',
                className="mb-3"
            ),
            html.Label('Catégories', className="font-weight-bold", style={'marginTop': '10px'}),
            dcc.Dropdown(
                id='multi-dropdown',
                options=generate_options_with_counts(categories, data),
                multi=True,
                placeholder="Sélectionner des catégories et sous-catégories",
                className="mb-3"
            ),
            html.Label('Point de départ', className="font-weight-bold"),
            dcc.Dropdown(
                id='start-point-dropdown',
                placeholder='Sélectionner un point de départ',
                className="mb-3"
            ),
            html.Label('Point d\'arrivée', className="font-weight-bold"),
            dcc.Dropdown(
                id='end-point-dropdown',
                placeholder='Sélectionner un point d\'arrivée',
                className="mb-3"
            ),
            dbc.Button("Calculer l'itinéraire", id="calculate-route", color="primary", className="mt-3"),
            dbc.Button("Exporter la carte", id="export-map", color="secondary", className="mt-3"),
            dcc.Download(id="download-map"),
            dbc.Alert(id='rate-limit-alert', color='danger', is_open=False, duration=4000)
        ], width=3, style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRight': '2px solid #e9ecef'}),
        dbc.Col([
            html.Iframe(id='map', srcDoc=initial_map._repr_html_(), width='100%', height='600', style={"border": "2px solid #e9ecef", "border-radius": "8px"})
        ], width=9)
    ], style={'marginBottom': '30px'})
], fluid=True, style={'backgroundColor': '#f0f2f5', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})

@app.callback(
    Output('poi-dropdown', 'options'),
    Input('category-dropdown', 'value')
)
def set_points_interet(selected_category):
    if (selected_category is None) or (selected_category not in points_interet):
        return []
    return [{'label': poi, 'value': poi} for poi in points_interet[selected_category]]

@app.callback(
    Output('selected-cluster', 'data'),
    Output('positions-dict', 'data'),
    Output('start-point-dropdown', 'options'),
    Output('end-point-dropdown', 'options'),
    Input('poi-dropdown', 'value')
)
def store_selected_cluster(selected_poi):
    if not selected_poi:
        return None, None, [], []
    matching_observations = find_best_match(data, [selected_poi])
    if matching_observations.empty:
        return None, None, [], []
    cluster_id = matching_observations['Cluster'].values[0]
    cluster_data = data[data['Cluster'] == cluster_id]
    labels = cluster_data['Label'].values
    positions_dict = {label: [row['Latitude'], row['Longitude']] for label, row in cluster_data.set_index('Label').iterrows()}
    options = [{'label': label, 'value': label} for label in labels]
    return cluster_id, positions_dict, options, options

@app.callback(
    Output('multi-dropdown', 'options'),
    Input('selected-cluster', 'data')
)
def update_category_options(cluster_id):
    return generate_options_with_counts(categories, data, cluster_id)

@app.callback(
    Output('map', 'srcDoc'),
    Output('itineraries', 'data'),
    Output('rate-limit-alert', 'is_open'),
    Input('calculate-route', 'n_clicks'),
    Input('day-dropdown', 'value'),
    State('start-point-dropdown', 'value'),
    State('end-point-dropdown', 'value'),
    State('selected-cluster', 'data'),
    State('positions-dict', 'data'),
    State('itineraries', 'data'),
    prevent_initial_call=True
)
def update_map(n_clicks, selected_day, start_point, end_point, cluster_id, positions_dict, itineraries):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, False
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'calculate-route':
        if not positions_dict or not selected_day or not start_point or not end_point:
            return dash.no_update, itineraries, False

        try:
            chemin = tsp_voisin_le_plus_proche(G, start_point, end_point)
            folium_map = create_map_with_folium(chemin, positions_dict)

            # Stocker l'itinéraire calculé dans le store
            itineraries[str(selected_day)] = {
                'chemin': chemin,
                'positions_dict': positions_dict
            }

            return folium_map._repr_html_(), itineraries, False
        except openrouteservice.exceptions.ApiError as e:
            if "Rate limit exceeded" in str(e):
                time.sleep(60)
                return dash.no_update, itineraries, True
            else:
                raise
    elif trigger_id == 'day-dropdown':
        if selected_day is None or str(selected_day) not in itineraries:
            return dash.no_update, itineraries, False

        itineraire = itineraries[str(selected_day)]
        chemin = itineraire['chemin']
        positions_dict = itineraire['positions_dict']

        folium_map = create_map_with_folium(chemin, positions_dict)

        return folium_map._repr_html_(), itineraries, False

@app.callback(
    Output("download-map", "data"),
    Input("export-map", "n_clicks"),
    State("itineraries", "data"),
    State('day-dropdown', 'value'),
    prevent_initial_call=True
)
def export_map(n_clicks, itineraries, selected_day):
    if n_clicks and selected_day and str(selected_day) in itineraries:
        itineraire = itineraries[str(selected_day)]
        chemin = itineraire['chemin']
        positions_dict = itineraire['positions_dict']
        folium_map = create_map_with_folium(chemin, positions_dict)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
            folium_map.save(temp_file.name)
            temp_file_path = temp_file.name
        return dcc.send_file(temp_file_path)

if __name__ == '__main__':
    app.run_server(debug=True, port=80)
