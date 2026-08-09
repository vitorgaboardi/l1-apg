import requests

BASE_URL = "https://api.frankfurter.dev/v1/latest"

# check website: https://frankfurter.dev/

def converter_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Converts an amount from one currency to another.
    Args:
        - amount (float): The amount of money to convert.
        - from_currency (str): The currency code to convert from (e.g., "USD").
        - to_currency (str): The currency code to convert to (e.g., "EUR").
    Returns:
        dict: A dictionary containing the converted amount and the exchange rate.
    """
    url = f"{BASE_URL}?base={from_currency}&symbols={to_currency}"
    try:
        resp = requests.get(url, timeout=3)
        data = resp.json()
        
        if resp.status_code == 200:
            rate = data.get("rates", {}).get(to_currency)
            if rate is not None:
                converted_amount = amount * rate
                return {
                    "status_code": 200,
                    "data": {
                        "converted_amount": converted_amount,
                        "exchange_rate": rate,
                        "date": data.get("date", "")
                    }
                }
            else:
                return {"status_code": 404, "error": f"Currency code '{to_currency}' not found."}
        else:
            return {"status_code": resp.status_code, "error": "Failed to fetch exchange rate."}

    except requests.exceptions.RequestException as e:
        return {"status_code": 400, "error": str(e)}
    except Exception as e:
        return {"status_code": 500, "error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    result = converter_currency(amount=100, from_currency="USD", to_currency="HKD")
    print(result)