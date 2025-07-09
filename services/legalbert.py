import torch
from models import legalbert_model as model, legalbert_tokenizer as tokenizer, device, faiss_index, metadata_list

def get_query_embedding(query: str):
    with torch.no_grad():
        encoded = tokenizer([query], return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        last_hidden = outputs.last_hidden_state
        mask = encoded['attention_mask'].unsqueeze(-1).expand(last_hidden.size())
        summed = torch.sum(last_hidden * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        return (summed / counts).cpu().numpy()

def search_faiss_index(query, top_k):
    query_embedding = get_query_embedding(query)
    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(metadata_list):
            result = metadata_list[idx].copy()
            result["distance"] = float(dist)
            result["id"] = idx
            results.append(result)

    return results


