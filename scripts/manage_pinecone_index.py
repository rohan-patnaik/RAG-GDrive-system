# scripts/manage_pinecone_index.py
import os
from pinecone import Pinecone, PodSpec # ServerlessSpec might be needed depending on your index type
# from pinecone import ServerlessSpec # Uncomment if using serverless indexes
from dotenv import load_dotenv
import traceback

def setup_pinecone_index():
    load_dotenv() # Load .env file

    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT") # e.g., "us-east-1" for pod, or a serverless region
    index_name = os.getenv("PINECONE_INDEX_NAME")
    embedding_dimension = 384 # For all-MiniLM-L6-v2

    if not all([api_key, environment, index_name]):
        print("Error: Pinecone API key, environment, or index name not found in .env")
        return

    try:
        print(f"Initializing Pinecone client...")
        pinecone_client = Pinecone(api_key=api_key)

        print(f"Checking if index '{index_name}' exists...")

        # Correct way to get index names for pinecone-client v3.0.0
        index_list_response = pinecone_client.list_indexes()
        existing_index_names = []
        if index_list_response and hasattr(index_list_response, 'indexes') and index_list_response.indexes:
            existing_index_names = [idx_model.name for idx_model in index_list_response.indexes if hasattr(idx_model, 'name')]
        
        print(f"Found existing indexes: {existing_index_names}")

        if index_name not in existing_index_names:
            print(f"Index '{index_name}' does not exist. Creating it...")

            # Determine if you are using Pod-based or Serverless indexes.
            # Adjust the spec accordingly.
            # For Pod-based (ensure 'environment' in PodSpec matches your Pinecone project's pod environment):
            spec_to_use = PodSpec(
                environment=environment
                # pod_type="starter" # Specify if required for your free tier, e.g., "s1.x1" or "p1.x1" for paid
            )

            # OR, for Serverless (ensure 'region' in ServerlessSpec matches your serverless region):
            # from pinecone import ServerlessSpec # Make sure to import
            # spec_to_use = ServerlessSpec(
            #     cloud="aws",  # or "gcp", "azure" - choose based on your 'environment'
            #     region=environment # This should be your serverless region like "us-east-1"
            # )

            pinecone_client.create_index(
                name=index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=spec_to_use
            )
            print(f"Index '{index_name}' creation initiated. Please wait for it to initialize in the Pinecone console.")
        else:
            print(f"Index '{index_name}' already exists.")

        # Verify connection to the index
        print(f"Attempting to connect to index '{index_name}'...")
        index = pinecone_client.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Successfully connected to index '{index_name}'. Stats: {stats}")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    setup_pinecone_index()
