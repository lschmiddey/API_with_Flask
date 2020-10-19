import requests

# https://your-heroku-app-name.herokuapp.com/predict
# http://localhost:5000/predict

data = {'Pclass': 1, 'Age': 22.0, 'Sex': 0, 'Parch': 1}
r = requests.get("http://localhost:5000/predict", params=data)


print(r.text)
