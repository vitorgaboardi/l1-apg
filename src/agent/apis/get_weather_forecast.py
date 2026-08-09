import requests
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")
BASE_URL = "http://api.weatherapi.com/v1"
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "10"))
API_LANGUAGE = os.getenv("API_LANGUAGE", "en")

def get_weather_forecast(city: str, days: int = 3) -> Dict[str, Any]:
    """Get weather forecast for a city using WeatherAPI"""
    url = f"{BASE_URL}/forecast.json?key={API_KEY}&q={city}&days={days}&lang={API_LANGUAGE}"
    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        forecast_days = []
        for day in data.get("forecast", {}).get("forecastday", []):
            forecast_days.append({
                "date": day.get("date"),
                "max_temp_c": day.get("day", {}).get("maxtemp_c"),
                "min_temp_c": day.get("day", {}).get("mintemp_c"),
                "max_temp_f": day.get("day", {}).get("maxtemp_f"),
                "min_temp_f": day.get("day", {}).get("mintemp_f"),
                "condition": day.get("day", {}).get("condition", {}).get("text"),
                "chance_of_rain": day.get("day", {}).get("daily_chance_of_rain"),
                "max_wind_kph": day.get("day", {}).get("maxwind_kph"),
                "avg_humidity": day.get("day", {}).get("avghumidity"),
                "uv_index": day.get("day", {}).get("uv")
            })
        
        return {
            "status_code": 200,
            "data": { 
                "city": data.get("location", {}).get("name"),
                "country": data.get("location", {}).get("country"),
                "region": data.get("location", {}).get("region"),
                "forecast": forecast_days
            }
        }
    except requests.exceptions.RequestException as e:
        return {"status_code": 400, "error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"status_code": 500, "error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    city_name = "Ho Chi Minh"
    weather_info = get_weather_forecast(city_name, days=3)
    print(weather_info)