import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

API_KEY = os.getenv('OPENWEATHER_API_KEY', 'fc37682183690a8f08872139556bc720')

city = "London"
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

try:
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print("✅ API Key works!")
        print(f"City: {data['name']}")
        print(f"Temperature: {data['main']['temp']}°C")
        print(f"Weather: {data['weather'][0]['description']}")
    else:
        print(f"❌ Error {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Connection error: {e}")