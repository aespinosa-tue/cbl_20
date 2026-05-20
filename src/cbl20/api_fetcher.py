import time
import requests
import pandas as pd


BASE_URL = "https://data.police.uk/api"


def police_get(endpoint, params=None, max_retries=3):
    """
    Simple helper for calling the Police.uk API.
    endpoint example: "/crime-last-updated"
    """

    url = BASE_URL + endpoint

    for attempt in range(max_retries):
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 429:
            wait_time = 2 + attempt
            print(f"Rate limit reached. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            continue

        if response.status_code == 503:
            raise RuntimeError(
                "API returned 503. Your area may contain too many crimes. "
                "Try a smaller polygon, smaller area, or specific crime category."
            )

        response.raise_for_status()
        return response.json()

    raise RuntimeError("API request failed after retries.")

