import requests

url = "http://localhost:8000/predict/"

data = {"externalStatus": "port out"} 


# Send POST request to the server
response = requests.post(url, json=data)

# Check the response
if response.status_code == 200:
    print("Prediction successful!")
    print("Predicted internal status:", response.json()["predicted_internal_status"])
else:
    print("Prediction failed:", response.text)
