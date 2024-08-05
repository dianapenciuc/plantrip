import time 

def temps_execution(function):
    def timer(*args, **kwargs):
        heure_debut = time.time()
        results = function(*args, **kwargs)
        heure_fin = time.time()
        temps = heure_fin - heure_debut
        print("Cette fonction s'est exécutée en {} s".format(temps))
        return results
    return timer