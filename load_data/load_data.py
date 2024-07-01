import requests
import yaml

# Charger les configurations à partir du fichier YAML
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

numero_flux = config['numero_flux']
api_key = config['api_key']

url = f'https://diffuseur.datatourisme.fr/webservice/{numero_flux}/{api_key}'

headers = {
    'Accept-Encoding': 'gzip'  
}

response = requests.get(url, headers=headers, stream=True)

if response.status_code == 200:
    with open('datatourisme.zip', 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("Téléchargement réussi.")
else:
    print("Erreur lors du téléchargement : Code", response.status_code)

