#!/bin/bash

echo "Lauching Extract_jsonld.py in 5 seconds"
sleep 5s

echo "Lauching Extract_jsonld.py..."
# Exécuter le premier script Python
python Extract_jsonld.py
if [ $? -ne 0 ]; then
  echo "Extract_jsonld.py has failed, shutting down."
  exit 1
fi

echo "Lauching Transform_csv.py in 5 seconds"
sleep 5s
echo "Lauching Transform_csv.py..."

# Exécuter le deuxième script Python
python Transform_csv.py
if [ $? -ne 0 ]; then
  echo "Transform_csv.py has failed, shutting down."
  exit 1
fi

echo "Lauching Load_csv_to_sql.py in 5 seconds"
sleep 5s
echo "Lauching Load_csv_to_sql.py..."

# Exécuter le troisième script Python
python Load_csv_to_sql.py
if [ $? -ne 0 ]; then
  echo "Load_csv_to_sql.py has failed, shutting down."
  exit 1
fi

echo "All scripts terminated with success."
echo "Ending program"