"""
Gets the current weather for a given city using the WeatherAPI service.
Code adapted from MCP-Bench's implementation of the same functionality.
Check documentation here: https://www.weatherapi.com/docs/
"""
import requests
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")
BASE_URL = "http://api.weatherapi.com/v1"
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "10"))
API_LANGUAGE = os.getenv("API_LANGUAGE", "en")

def get_current_weather(city: str) -> Dict[str, Any]:
    """Get current weather for a city using WeatherAPI"""
    url = f"{BASE_URL}/current.json?key={API_KEY}&q={city}&lang={API_LANGUAGE}"
    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        return {
            "status_code": 200,
            "data": { 
                "city": data.get("location", {}).get("name"),
                "country": data.get("location", {}).get("country"),
                "region": data.get("location", {}).get("region"),
                "weather": data.get("current", {}).get("condition", {}).get("text"),
                "temperature_c": data.get("current", {}).get("temp_c"),
                "temperature_f": data.get("current", {}).get("temp_f"),
                "feelslike_c": data.get("current", {}).get("feelslike_c"),
                "feelslike_f": data.get("current", {}).get("feelslike_f"),
                "humidity": data.get("current", {}).get("humidity"),
                "wind_kph": data.get("current", {}).get("wind_kph"),
                "wind_mph": data.get("current", {}).get("wind_mph"),
                "wind_dir": data.get("current", {}).get("wind_dir"),
                "pressure_mb": data.get("current", {}).get("pressure_mb"),
                "visibility_km": data.get("current", {}).get("vis_km"),
                "uv_index": data.get("current", {}).get("uv"),
                "last_updated": data.get("current", {}).get("last_updated")
            }
        }
    except requests.exceptions.RequestException as e:
        return {"status_code": 400, "error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"status_code": 500, "error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    city_name = "Ho Chi Minh"
    weather_info = get_current_weather(city_name)
    print(weather_info)