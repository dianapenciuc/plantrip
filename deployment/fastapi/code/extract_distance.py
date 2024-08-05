import requests
from requests.auth import HTTPBasicAuth

API_KEY = '5b3ce3597851110001cf6248a4fd0430507245c588e586f8b4463c86'
base_url = 'https://api.openrouteservice.org/v2/matrix/'

headers = {'Accept': 'application/json','Authorization':'Bearer ' + API_KEY}
auth = HTTPBasicAuth('apikey', API_KEY)
"""
params: 
    locations - is a list of [long,lat] coordinates
    metrics - a list of metrics such as 'distance', 'duration'
profile:
    Options : 'foot-walking', 'foot-walking', 'cycling-regular'
locations: example
    [[2.33401570000001,48.8430613],[2.33978969999998,48.8454821],[2.3412439,48.8407953],[2.34158460000003,48.8436554],[2.33728500000007,48.8469529],\
    [2.33578190000003,48.8376631],[2.32759009999995,48.8400076],[2.32759009999995,48.8400076],[2.33140079999998,48.8474735],[2.32652280000002,48.8430927],\
    [2.33398839999995,48.8487542],[2.33058549999998,48.8482028],[2.33175057301639,48.8365491019355],[2.33263290000002,48.8491222],[2.34477249999998,48.844255],\
    [2.34482760000003,48.8442235],[2.33732599999996,48.8491491],[2.34500990000004,48.8439046],[2.33076329999994,48.8361188],[2.34600019999994,48.844219]],
"""

def get_ors_matrix(locations,metrics,profile='foot-walking'):
    url = base_url + profile
    data = {
        "locations": locations,
       "metrics":metrics
    }

    req = requests.post(url, headers=headers, json = data)
    json = req.json()
    if metrics == 'duration':
        return json['duration']
    elif metrics == 'distance':
        return json['distance']
    return json