# services/retrieval.py
import traceback
#from services.glove import search_glove
from legalbert import search_faiss_index

def safe_strip(value, default=''):
    
    if value and isinstance(value, str):
        return value.strip()
    return default

def format_result(result):
    return {
        'source': safe_strip(result.get('source'), 'Unknown'),
        'section': safe_strip(result.get('section_number')),
        'part': safe_strip(result.get('part_number')),
        'chapter': safe_strip(result.get('chapter_number')),
        'text': safe_strip(result.get('text')),
        'distance': round(float(result.get('distance') or 0), 2)
    }


def format_glove_result(result, query):
    return {
        'text': safe_strip(result.get('text')),
        'score': round(float(result.get('score', 0)), 3)
    }






def retrieve_documents(query, model_name,top_k=5):
    try:
        query_lower = query.lower()

        if model_name == 'legalbert':
            
                results= search_faiss_index(query, top_k)
                results = [format_result(r) for r in results]
                return results
  
      #  elif model_name == 'glove':
       #     top_matches =  search_glove(query_lower, top_k)
        #    results = [format_glove_result(r, query) for r in top_matches]
         #   return results
            
        else:
            return ["Invalid model selected."]
        
    except Exception as e:
            print(f"Embedding/retrieval error: {e}")
            traceback.print_exc()
            return ["Error retrieving results with embedding model."]



query="I want to learn about criminal procedure in Pakistan."
model_to_use = 'legalbert'  # Choose the model you want to use, e.g., 'glove', 'legalbert', etc.
answer = retrieve_documents(query, model_to_use, top_k=10)  # Use your desired k value
print("Query:", query)
print("Results:", answer)