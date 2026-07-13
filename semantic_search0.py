''' 
https://machinelearningmastery.com/build-semantic-search-with-llm-embeddings/
Embedding models are used to convert text into numerical vector representations, capturing semantic meaning.
These embeddings allow for semantic search, where queries can retrieve relevant documents based on meaning rather than exact
keyword matches. The process involves encoding both the documents and the search queries into embeddings, and then using a nearest neighbors algorithm to find the most similar documents to a given query.
'''
import os
import pandas as pd
import json
from pydantic import BaseModel, Field
from openai import OpenAI
# from google.colab import userdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

HF_CACHE_BASE = "/ws/parkuma3-bgl/iosxr/test/.hf_cache"
os.makedirs(HF_CACHE_BASE, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_CACHE_BASE)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{HF_CACHE_BASE}/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{HF_CACHE_BASE}/transformers")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", f"{HF_CACHE_BASE}/sentence_transformers")

print("Loading dataset...")
target_month = "202605"
data_pattern = f"/ws/parkuma3-bgl/iosxr/test/ap/work/archive/erspan_ap_{target_month}*/all.log"
dataset = load_dataset(
	"text",
	data_files={"train": data_pattern},
	split="train[:1000]",
	cache_dir="/ws/parkuma3-bgl/iosxr/test/.hf_cache",
)

# Extract the text column into a Python list
documents = dataset["text"]

print(f"Loaded {len(documents)} documents.")
print(f"Sample: {documents[2][:101]}...")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text documents into numerical vector embeddings
print("Encoding documents (this may take a few seconds)...")
document_embeddings = model.encode(documents, show_progress_bar=True)

print(f"Created {document_embeddings.shape[0]} embeddings.")

search_engine = NearestNeighbors(n_neighbors=5, metric="cosine")

search_engine.fit(document_embeddings)
print("Search engine is ready!")
