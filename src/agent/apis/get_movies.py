import requests
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3/search/movie"
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "10"))
API_LANGUAGE = os.getenv("API_LANGUAGE", "en")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

def get_movies(name: str, year: int = None, max_results: int = 10) -> dict:
    """
    Get movie suggestions from TMDb API based on keyword and year.
    """
    url = f"{BASE_URL}?api_key={TMDB_API_KEY}&query={name}&language={API_LANGUAGE}"
    if year is not None:
        url += f"&year={year}"

    try:
        response = requests.get(url, timeout=API_TIMEOUT)
    except requests.exceptions.RequestException as e:
        return {"status_code": 400, "error": f"API request failed: {str(e)}"}

    if response.status_code == 200:
        data = response.json()
        if not data.get("results"):
            return {"status_code": 404, "error": "No results found for the movie."}

        movies = []
        for movie in data["results"][:max_results]:
            title = movie.get("title", "Unknown")
            overview = movie.get("overview", "No description available.")
            original_language = movie.get("original_language", "Unknown")
            popularity = movie.get("popularity", 0)
            release_date = movie.get("release_date", "Unknown")

            movies.append({
                "title": title,
                "overview": overview,
                "original_language": original_language,
                "popularity": popularity,
                "release_date": release_date
            })

        return {"status_code": 200, "data": movies}
    else:
        return {"status_code": response.status_code, "error": f"Request failed: {response.text[:200]}"}

if __name__ == "__main__":
    keyword = "Harry Potter"
    movies = get_movies(keyword)
    print(movies)
    for movie in movies["data"]:
        print(f"Title: {movie['title']}")
        print(f"Overview: {movie['overview']}")
        print(f"Original Language: {movie['original_language']}")
        print(f"Popularity: {movie['popularity']}")
        print(f"Release Date: {movie['release_date']}")
        print()