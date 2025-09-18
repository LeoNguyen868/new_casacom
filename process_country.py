import pandas as pd
import pycountry
import reverse_geocoder as rg
from functools import lru_cache

# Use lru_cache to avoid repeated lookups for the same coordinates
@lru_cache(maxsize=1024)
def get_country_from_coordinates(lat, lon):
    try:
        # Use reverse_geocoder to get country information from coordinates
        result = rg.search((lat, lon), mode=1)
        if result and len(result) > 0:
            country_code = result[0]['cc']
            # Convert country code to full country name
            country = pycountry.countries.get(alpha_2=country_code)
            if country:
                return country.name
            else:
                return country_code
        return None
    except Exception:
        return None