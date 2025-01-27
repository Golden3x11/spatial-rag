PROMPT_NER_DISTANCE_WHERE = """
Please extract the following fields from the user's query:

1. `LOCATION_VECTOR_DB_PROMPT`: Provide a representation of the location mentioned in the user's query. This should correspond to a specific place or landmark (e.g., "Dworzec Wrocław Główny"). It should have been extracted in OSM format, e.g., 'amenity': 'restaurant', 'cuisine': 'polish'. if you can find address also pass it with prefix addr:city, addr:street, addr:housenumber, addr:postcode.
   
2. `PLACE_VECTOR_DB_PROMPT`: Provide a representation of the type of place or business mentioned in the query (e.g., "restaurant", "hotel", etc.). In the given example, it would be a "Polish restaurant". It should have been extracted in OSM format, e.g., 'amenity': 'restaurant', 'cuisine': 'polish'.

3. `DISTANCE`: If the query contains information about the proximity or distance, either explicitly or implicitly, include it. This could be a specific distance (e.g., "5 km away") or a general description (e.g., "near"). If available if not thy to change it to a number e.g., "near" -> "1 km"] Provide only number with km postfix.
Example Query:  "Find me a Polish restaurant near Dworzec Wrocław Główny.

You can add additional instructions to OSM format like steet, housenumber, city, postcode, etc. if you can find it.

Expected Extraction:
`LOCATION_VECTOR_DB_PROMPT`: 'name': 'Dworzec Wrocław Główny', 'description': 'train station'
`PLACE_VECTOR_DB_PROMPT`: 'amenity': 'restaurant', 'cuisine': 'polish'
`DISTANCE`: 1 km

Query: "{query}"
"""

PROMPT_FIND_LATITUDE_LONGITUDE = """
From provided Documents, find the latitude and longitude of the location mentioned in the user's query.
If it is not explicitly mentioned, try to extract it from the metadata of the documents. It can be near the location mentioned in the query.

QUERY: "{query}"

DOCUMENTS:

{documents}

Expected Extraction:
`EXPLANATION`: (short explanation of how the latitude and longitude were extracted)
`LATITUDE`: (latitude value)
`LONGITUDE`: (longitude value)
"""


PROMPT_END = """
Provide a short recommendation based on user query and the documents provided. 
Mention all documents. Write it short and concise.

QUERY: "{query}"

DOCUMENTS:

{documents}

RECOMMENDATION: 
"""