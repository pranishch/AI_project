import requests

try:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": open("./static/cat.jpg", "rb")}
    )
    
    # Debug output
    print("Status Code:", response.status_code)
    print("Headers:", response.headers)
    print("Raw Content:", response.content)
    
    # Only try to parse JSON if content exists
    if response.content:
        print("JSON Response:", response.json())
    else:
        print("Received empty response")

except requests.exceptions.RequestException as e:
    print("Request failed:", str(e))
except ValueError as e:
    print("JSON decode failed:", str(e))