import requests
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
FOOD_API_KEY = os.getenv("FOOD_API_KEY", "")

def get_nutrition(query: str, limit: int = 10) -> dict:
    """Gets nutrition information for a given food query.
    Args:
        - query (str): The food item to search for.
        - limit (int): The maximum number of results to return. Default is 10.  
    """
    url = f"{BASE_URL}?api_key={FOOD_API_KEY}&query={query}&pageSize={limit}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if not data.get("foods"):
                return {"status_code": 404, "error": "No results found for the food item."}

            nutrition_info = []
            for food in data["foods"]:
                name = food.get("description", "Unknown")
                ingredients = food.get("ingredients", "Unknown")
                calories = next((nutrient.get("value") for nutrient in food.get("foodNutrients", []) if nutrient.get("nutrientName") == "Energy"), "Unknown")
                protein = next((nutrient.get("value") for nutrient in food.get("foodNutrients", []) if nutrient.get("nutrientName") == "Protein"), "Unknown")
                carbohydrates = next((nutrient.get("value") for nutrient in food.get("foodNutrients", []) if nutrient.get("nutrientName") == "Carbohydrate, by difference"), "Unknown")
                fats = next((nutrient.get("value") for nutrient in food.get("foodNutrients", []) if nutrient.get("nutrientName") == "Total lipid (fat)"), "Unknown")

                nutrition_info.append({
                    "name": name,
                    "ingredients": ingredients,
                    "calories": calories,
                    "protein": protein,
                    "carbohydrates": carbohydrates,
                    "fats": fats
                })

            return {"status_code": 200, "data": nutrition_info}
        else:
            return {"status_code": response.status_code, "error": "Not Found"}
    except requests.exceptions.RequestException as e:
        return {"status_code": 400, "error": str(e)}
    except Exception as e:
        return {"status_code": 500, "error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    food_query = "big mac"
    nutrition = get_nutrition(food_query, limit=5)
    if nutrition["status_code"] == 200:
        for item in nutrition["data"]:
            print(f"Name: {item['name']}")
            print(f"Ingredients: {item['ingredients']}")
            print(f"Calories: {item['calories']} kcal")
            print(f"Protein: {item['protein']} g")
            print(f"Carbohydrates: {item['carbohydrates']} g")
            print(f"Fats: {item['fats']} g")
            print()
    else:
        print(f"Error: {nutrition.get('error', 'Failed to get nutrition information.')}")