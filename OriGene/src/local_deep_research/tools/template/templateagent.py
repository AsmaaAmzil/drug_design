import os
import pickle
import numpy as np
from openai import OpenAI
from ...config import template_embedding_api_key, template_embedding_api_base_url

current_dir = os.path.dirname(os.path.abspath(__file__))
pickle_dir = os.path.join(current_dir, "templates.pkl")
with open(pickle_dir, "rb") as f:
    templates = pickle.load(f)
    f.close()

client = OpenAI(
    api_key=template_embedding_api_key,
    base_url=template_embedding_api_base_url,
)


# def get_embedding(text):
#     resp = client.embeddings.create(
#         model="text-embedding-3-small", input=[text], encoding_format="float"
#     )
#     embedding = np.array(resp.data[0].embedding)
#     embedding = embedding / np.linalg.norm(embedding)
#     return embedding
# def get_embedding(text):
#     import numpy as np
#     try:
#         # your dummy fallback (no API needed)
#         embedding = np.zeros(4096, dtype=np.float64)
#         return embedding  # don't normalize zeros, just return as-is
#     except Exception as e:
#         print(f"Error embedding query: {e}")
#         return np.zeros(4096, dtype=np.float64)  # ← safe fallback here too
def get_embedding(text):
    import numpy as np
    import requests
    try:
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": "Bearer jina_1fafa66036f74791abaa02b6c0e5fb99VWt6TJwvVLiYWNbVmDM1b2knjakt", "Content-Type": "application/json"},
            json={"model": "jina-embeddings-v3", "input": [text], "dimensions": 1024}
        )
        embedding = response.json()["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float64)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return np.zeros(1024, dtype=np.float64)



def retrieve_small_template(query):
    query_embedding = get_embedding(query)
    score = np.dot(query_embedding, templates["small"]["embeddings"].T)
    index = np.argmax(score)
    return templates["small"]["value_list"][index]


def retrieve_large_template(query):
    query_embedding = get_embedding(query)
    score = np.dot(query_embedding, templates["large"]["embeddings"].T)
    index = np.argmax(score)
    return templates["large"]["value_list"][index]
