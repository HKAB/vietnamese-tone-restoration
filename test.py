import requests

url = 'http://localhost:5000/n_gram_predict'
r = requests.post(url,json={'sentence': 'tram nam trong coi nguoi ta'})

print(r.json())