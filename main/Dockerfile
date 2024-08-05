# Utiliser l'image Ubuntu comme base
FROM python:3.9

WORKDIR /usr/app/src/

# Copier les fichiers requirements.txt et installer les dépendances (si nécessaire)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les scripts Python et le script d'exécution dans le conteneur
COPY Extract_jsonld.py .
COPY Transform_csv.py .
COPY Load_csv_to_sql.py .
COPY run_scripts.sh .

# Rendre le script exécutable
RUN chmod +x run_scripts.sh

# Spécifier le script à exécuter lors du démarrage du conteneur
CMD ["./run_scripts.sh"]