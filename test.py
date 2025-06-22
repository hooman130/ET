import requests

# API URL
url = "https://api.hcdp.ikewai.org/raster/timeseries"

# Parameters
params = {
    "start": "2019-01-01",
    "end": "2019-12-31",
    "lat": 21.6832,
    "lng": -157.9604,
    "extent": "statewide",
    "datatype": "rainfall",
    "period": "day",
    "production": "new",
}

# Headers
headers = {
    "Authorization": "Bearer 1b8a6439c85b8e42e211b68ea68ac198"
}

# Making the GET request
response = requests.get(url, headers=headers, params=params)

# Checking if request was successful
if response.status_code == 200:
    data = response.json()  # Parse JSON response
    print(data)  # Print the data
else:
    print(f"Error {response.status_code}: {response.text}")


import requests

# API URL
url = "https://api.hcdp.ikewai.org/raster/timeseries"

# Parameters: temperature data using lat and lng, with aggregation "max"
params = {
    "start": "2019-01-01",
    "end": "2019-12-31",
    "lat": 21.6832,
    "lng": -157.9604,
    "extent": "statewide",
    "datatype": "temperature",  # Temperature instead of rainfall
    # "production": "new",
    "period": "day",
    "aggregation": "min"       # Aggregation parameter added
}

# Headers with your API token
headers = {
    "Authorization": "Bearer 1b8a6439c85b8e42e211b68ea68ac198"  # Replace with your actual token
}

# Making the GET request
response = requests.get(url, headers=headers, params=params)

# Checking if the request was successful
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error {response.status_code}: {response.text}")
