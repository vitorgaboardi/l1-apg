import requests

BASE_URL = "https://geocoding-api.open-meteo.com/v1/search"

def get_geolocation(city_name: str) -> dict:
    """Gets the geolocation (latitude and longitude) of a given city name.
    Args:
       - city_name (str): The name of the city to get the geolocation for.

    Returns:
       dict: A dictionary containing:
           - 'city' (str): The name of the city.
           - 'region' (str): The region of the city.
           - 'country' (str): The country of the city.
           - 'latitude' (float): The latitude of the city.
           - 'longitude' (float): The longitude of the city.
           - 'timezone' (str): The timezone of the city.
    """
    try:
        resp = requests.get(f"{BASE_URL}?name={city_name}&count=3&language=en&format=json", timeout=3)
        resp.raise_for_status()
        data = resp.json()
        results = data.get('results', [])
        if not results:
            return {"status_code": 404, "error": f"City '{city_name}' not found."}
        place = results[0]
        return {
            "status_code": 200,
            "data": {
                "city": place.get("name"),
                "region": place.get("admin1"),
                "country": place.get("country"),
                "latitude": float(place.get("latitude")),
                "longitude": float(place.get("longitude")),
                "timezone": place.get("timezone"),
            }
        }
    except requests.exceptions.RequestException as e:
        return {"status_code": 400, "error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"status_code": 500, "error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    city_name = "Ho Chi Minh"
    geolocation_info = get_geolocation(city_name)
    print(geolocation_info)