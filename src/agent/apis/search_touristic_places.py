"""
Searches for restaurants given a city, radius, categories, conditions, and name.
"""

import requests
import os
from dotenv import load_dotenv
from requests.structures import CaseInsensitiveDict
from get_geolocation import get_geolocation

load_dotenv()
API_KEY = os.getenv("GEOPIFY_API_KEY")
BASE_URL = "https://api.geoapify.com/v2/places"


def search_touristic_places(city: str, radius: int = 10000, categories: list = [], conditions: list = [], name: str = "", limit: int = 5) -> dict:
    """Searches for touristic places given a city, radius, categories, conditions, and name.
    Args:
        - city (str): The name of the city to search for touristic places in.
        - radius (int): The radius (in meters) to search for touristic places around the city center. Default is 10000 meters.
        - categories (str): The categories of touristic places to search for. Default is "catering".
        - conditions (str): Additional conditions to filter the search results. Default is an empty string.
        - name (str): The name of the touristic place to search for. Default is an empty string.
        - limit (int): The maximum number of results to return. Default is 5 and maximum is 20.

    Returns:
        dict: A dictionary containing the search results, including touristic place names, addresses, and other relevant information.
    """
    geolocation = get_geolocation(city)
    if geolocation["status_code"] != 200:
        return {"status_code": geolocation["status_code"], "error": geolocation.get("error", "Failed to get geolocation")}
    
    # constructing the API request URL with filters
    search_string = ""

    # filtering by location
    latitude = geolocation["data"]["latitude"]
    longitude = geolocation["data"]["longitude"]
    filters=f"filter=circle:{longitude},{latitude},{radius}"
    search_string += filters

    # filtering by categories
    if categories == []:
        categories = "categories=tourism"
    else:
        categories = [f"tourism.{category_name}," for category_name in categories]
        categories = "categories=" + "".join(categories)[:-1]
    search_string += f"&{categories}"

    # filtering by condittions
    if conditions != []:
        conditions = "conditions=" + ",".join(conditions)
        search_string += f"&{conditions}"

    # filtering by name
    if name != "":
        search_string += f"&name={name}"

    # constructing the API request URL with filters
    limit = min(limit, 20)  # ensuring the limit does not exceed 20
    # print(f"Constructed search string: {search_string}")
    url = f"{BASE_URL}?{search_string}&limit={limit}&apiKey={API_KEY}"

    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            cleaned_data = []

            for feature in data.get("features", []):
                core_info = {
                    "name": feature.get("properties", {}).get("name"),
                    "address": feature.get("properties", {}).get("formatted"),
                    "categories": feature.get("properties", {}).get("categories", []),
                    "opening_hours": feature.get("properties", {}).get("opening_hours", "Not specified"),
                    "phone": feature.get("properties", {}).get("contact", {}).get("phone"),
                }
                cleaned_data.append(core_info)

            return {"status_code": 200, "data": cleaned_data}
        else:
            return {"status_code": resp.status_code, "error": "API request failed"}
    except requests.exceptions.RequestException as e:
        return {"status_code": 400, "error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"status_code": 500, "error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    city_name = "Paris"
    search_results = search_touristic_places(city_name, radius=10000, categories=["sights", "attraction"], conditions=[], name="")
    print(search_results)