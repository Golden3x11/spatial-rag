import os
import torch
import numpy as np
import streamlit as st
from golemai.nlp.llm_resp_gen import LLMRespGen
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from prompts import PROMPT_NER_DISTANCE_WHERE, PROMPT_FIND_LATITUDE_LONGITUDE, PROMPT_END
import textwrap
import folium
from streamlit_folium import folium_static

load_dotenv()

# Initialize variables
EMBEDDER_ID = "intfloat/multilingual-e5-large-instruct"
api_key = os.getenv("CLARIN_KEY")

# Initialize LLMRespGen
llm_rg = LLMRespGen(
    model_type='api',
    system_msg='You are a helpful assistant.',
    prompt_template='',
    batch_size=1,
    api_url='https://services.clarin-pl.eu/api/v1/oapi',
    api_key=api_key,
    timeout=200,
).set_generation_config(
    model_id='mixtral-8x22B',
)

# Initialize embeddings and vector stores
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER_ID)
location_vector_store = FAISS.load_local(
    'all_wroclaw_new',
    embeddings,
    allow_dangerous_deserialization=True,
)

booksy_vector_store = FAISS.load_local(
    'booksy_wroclaw_new',
    embeddings,
    allow_dangerous_deserialization=True,
)

# Define functions
def generate_response(QUERY, prompt_template, verbose=False, **kwargs):
    llm_rg.set_prompt_template(prompt_template)
    prompts = llm_rg.prepare_prompt(
        query=QUERY,
        **kwargs,
    )
    result = None

    st.write("### Prompt:")
    st.write(prompts)
    result = llm_rg._generate_llm_response(
        inputs=prompts,
    )
    st.write("### Response:")
    st.write(result)
    return result

def extract_entities_from_documents(query, documents):
    result = generate_response(query, PROMPT_NER_DISTANCE_WHERE, verbose=True)
    location = result.split("`LOCATION_VECTOR_DB_PROMPT`:")[-1].split("`PLACE_VECTOR_DB_PROMPT`")[0].strip()
    place = result.split("`PLACE_VECTOR_DB_PROMPT`:")[-1].split("`DISTANCE`")[0].strip()
    distance = result.split("`DISTANCE`:")[-1].split("km")[0].strip()
    distance = float(distance) if distance.isdigit() else distance
    return location, place, distance

def extract_lat_lon_from_documents(query, documents):
    result = generate_response(query, PROMPT_FIND_LATITUDE_LONGITUDE, documents=documents, verbose=True)
    latitude = result.split("`LATITUDE`:")[-1].split("`LONGITUDE`")[0].strip()
    longitude = result.split("`LONGITUDE`:")[-1].strip()
    latitude, longitude = float(latitude), float(longitude)
    return latitude, longitude

def summarize_results(query, documents):
    result = generate_response(query, PROMPT_END, documents=documents, verbose=True)
    return result

def documents_to_string(results):
    return "\n".join([f"{i}. Metadata: {doc.metadata}\n   Rest: {doc.page_content}" for i, doc in enumerate(results)])

def generate_square_filter(center, radius_km=5, amenity=None):
    lat, lon = center
    lat_delta = radius_km / 110.574
    lon_delta = radius_km / (111.320 * np.cos(np.radians(lat)))
    filter_query = {
        "$and": [
            {"lattitude": {"$gte": lat - lat_delta}},
            {"lattitude": {"$lte": lat + lat_delta}},
            {"longitude": {"$gte": lon - lon_delta}},
            {"longitude": {"$lte": lon + lon_delta}},
        ]
    }
    if amenity:
        filter_query["$and"].append({"amenity": amenity})
    return filter_query

# Streamlit UI

st.set_page_config(
    page_title="Location Finder",
    page_icon="🗺️",
)

st.title("🗺️Location Finder")

option = st.radio(
    "Choose a setting:",
    ("Location Finder", "Booksy Finder")
)

default_query = "Find me a place where I can buy a good coffee near to plac Grunwaldzki, Wrocław"
booksy_query = "Find ma a hairdresser that costs less than 100 PLN for a men's haircut near to Galeria Dominikańska, Wrocław"

query = st.text_area("Enter your query:", booksy_query if option == "Booksy Finder" else default_query, height=100)

if st.button("Search"):
    with st.spinner("Processing Entity Extraction..."):
        with st.expander("Entity Extraction - Details"):
            st.subheader("Extracted Entities")
            location, place, distance = extract_entities_from_documents(query, None)
            st.write(f"**Location:** {location}")
            st.write(f"**Place:** {place}")
            st.write(f"**Distance:** {distance} km")

    with st.spinner("Processing Latitude and Longitude Extraction..."):
        with st.expander("Latitude and Longitude Extraction - Details"):
            st.subheader("Extracted Coordinates")
            results = location_vector_store.similarity_search(location, k=5)
            documents = documents_to_string(results)
            latitude, longitude = extract_lat_lon_from_documents(location, documents)
            st.write(f"**Latitude:** {latitude}")
            st.write(f"**Longitude:** {longitude}")

    results = None
    
    with st.spinner("Processing Summary..."):
        with st.expander("Summary - Details"):
            st.subheader("Summary of Results")
            vector_store = location_vector_store if option == "Location Finder" else booksy_vector_store
            results = vector_store.similarity_search(
                place,
                k=5,
                filter=generate_square_filter((latitude, longitude), radius_km=distance),
            )

            if not results:
                st.write("No results found.")
                answer = "No results found. Rewrite your prompt and try again."
            else:
                documents = documents_to_string(results)
                answer = summarize_results(query, documents)

            
    st.write("## Recommendation:")
    st.write(answer)

    with st.expander("Map - Details"):
        # Create a map centered at the center location
        center = (latitude, longitude)
        m = folium.Map(location=center, zoom_start=15)

        # Add markers for each result
        for res in results:
            folium.Marker(
                location=[res.metadata["lattitude"], res.metadata["longitude"]],
                popup=res.metadata["name"],
            ).add_to(m)

        # Add a marker for the center location
        folium.Marker(
            location=center,
            popup="Center Location",
            icon=folium.Icon(color="red"),
        ).add_to(m)

        # Display the map in the Streamlit app
        folium_static(m)
