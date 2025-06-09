from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone,ServerlessSpec
import os

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
pc.create_index(
    name = os.environ['PINECONE_INDEX_NAME'],
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(
        region='us-east-1',
        cloud='aws'
    )
)
wine_index = pc.Index(os.environ['PINECONE_INDEX_NAME'])
print(wine_index.describe_index_stats())