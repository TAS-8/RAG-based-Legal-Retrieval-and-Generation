import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from services.models import glove_embeddings, glove_data


embedding_dim = len(next(iter(glove_embeddings.values())))
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def get_document_embedding(text, embeddings, embedding_dim):
    tokens = tokenize(text)
    vectors = [embeddings[word] for word in tokens if word in embeddings]
    return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_dim)


def search_glove(query, top_k):
    query_vector = get_document_embedding(query, glove_embeddings, embedding_dim).reshape(1, -1)
    results = []

    for doc in glove_data:
        doc_vector = np.array(doc['glove_embedding']).reshape(1, -1)
        sim = cosine_similarity(query_vector, doc_vector)[0][0]
        results.append({
            "text": doc["text"],  
            "score": float(sim)
        })

    
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]
