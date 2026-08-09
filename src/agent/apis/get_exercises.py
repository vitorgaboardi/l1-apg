import requests

# BASE_URL = "https://exercisedb.dev/api/v1/exercises"
BASE_URL = "https://oss.exercisedb.dev/api/v1/exercises"

# Check website: https://exercisedb.dev/docs#tag/muscles/GET/api/v1/muscles

def get_exercises(muscle: str = "", equipment: str = "", body_parts: str = "", limit: int = 10) -> dict:
    """Gets exercises based on muscle, equipment, and target.
    Args:
       - muscle (str): The muscle group to filter exercises by. Default is an empty string, which means no filtering by muscle.
       - equipments (str): The equipment to filter exercises by. Default is an empty string, which means no filtering by equipment.
       - body_parts (str): The body part to filter exercises by. Default is an empty string, which means no filtering by body part.
       - limit (int): The maximum number of exercises to return. Default is 10.
       """
    url = ""
    if muscle:
        url += f"targetMuscles={muscle.replace(' ', '%20')}&"
    if equipment:
        url += f"equipments={equipment.replace(' ', '%20')}&"
    if body_parts:
        url += f"bodyParts={body_parts.replace(' ', '%20')}&"
    url = f"{BASE_URL}?{url}&limit={limit}"
    try: 
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        data = resp.json()
        
        exercises = []
        for exercise in data.get("data", []):
            exercises.append(
                {
                    "name": exercise.get("name"),
                    "muscles": exercise.get("targetMuscles", ""),
                    "equipments": exercise.get("equipments", ""),
                    "instructions": exercise.get("instructions", ""),
                }
            )

        return {
            "status_code": 200,
            "data": exercises
        }

    except requests.exceptions.RequestException as e:
        return {"status_code": 400, "error": str(e)}
    except Exception as e:
        return {"status_code": 500, "error": f"Unexpected error: {str(e)}"}    

if __name__ == "__main__":
    exercises = get_exercises(muscle="biceps", equipment="dumbbell", body_parts="upper arms", limit=3)
    print(f"This is the response: {exercises}")
    for exercise in exercises["data"]:
        print(f"Exercise name: {exercise['name']}")
        print(f"Muscles: {exercise['muscles']}")
        print(f"Equipments: {exercise['equipments']}")
        print(f"Instructions: {exercise['instructions']}")
        print()