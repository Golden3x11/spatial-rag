# Location Finder and Booksy Finder

This project provides a tool to find locations and services using natural language queries. It leverages embeddings and vector stores to perform similarity searches and generate responses. This is a spatial RAG (Retrieval-Augmented Generation) system.

## Features

- **Location Finder**: Find places based on user queries.
- **Booksy Finder**: Find services like hairdressers based on user queries.
- **Entity Extraction**: Extracts entities such as location, place, and distance from the query.
- **Latitude and Longitude Extraction**: Extracts coordinates from documents.
- **Summary Generation**: Summarizes results and provides recommendations.

## Usage

### Running the Streamlit App

1. Run the Streamlit app:
   ```sh
   streamlit run rag_streamlit.py
   ```

2. Open the provided URL in your browser to access the app.

## Project Structure

- `rag.ipynb`: Jupyter Notebook for interactive usage of the location and service finder.
- `create_vector_db.ipynb`: Jupyter Notebook for creating and managing the vector database.
- `rag_streamlit.py`: Streamlit app for a web interface.
- `prompts.py`: Contains prompt templates for entity extraction and summarization.
- `.env`: Environment file to store API keys.

## License

This project is licensed under the MIT License.
