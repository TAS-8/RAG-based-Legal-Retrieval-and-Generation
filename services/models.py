import os
import numpy as np
import json
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

def load_files():
     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

     with open(os.path.join(BASE_DIR, "data", "metadata_list.json"), "r") as f:
        metadata_list = json.load(f)
     faiss_index_path = os.path.join(BASE_DIR, "data", "faiss_index .index")
     faiss_index = faiss.read_index(faiss_index_path)


     #glove_path = os.path.join(BASE_DIR, "data", "glove.6B.100d.txt")
     #embedded_data_path = os.path.join(BASE_DIR, "data", "embedded_data.json")
     #with open(embedded_data_path, 'r', encoding='utf-8') as f:
      #  glove_data = json.load(f)

     return metadata_list, faiss_index
metadata_list, faiss_index=load_files()



# === Load LegalBERT ===
print("Loading LegalBERT...")

# Global device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model/tokenizer
model_name = "law-ai/InLegalBERT"
legalbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
legalbert_model = AutoModel.from_pretrained(model_name)

# Add [PAD] token if not present
if legalbert_tokenizer.pad_token is None:
    legalbert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Added [PAD] token to tokenizer.")

# Add legal domain tokens
custom_tokens = [
    "<SUPREME_COURT>", "<HIGH_COURT>", "<TRIBUNAL>", "<MAGISTRATE>", "<JUDGE>", "<ADVOCATE>",
    "<FILING>", "<APPEAL>", "<PETITION>", "<HEARING>", "<SUMMONS>", "<JUDGMENT>", "<ORDER>", "<DECREE>",
    "<FUNDAMENTAL_RIGHT>", "<DUE_PROCESS>", "<CONSTITUTIONAL_PROVISION>", "<AMENDMENT>", "<FEDERALISM>",
    "<CONTRACT>", "<BREACH>", "<NEGLIGENCE>", "<TORT>", "<PROPERTY_RIGHT>", "<INJUNCTION>",
    "<CRIMINAL_OFFENSE>", "<ARREST>", "<CHARGE>", "<EVIDENCE>", "<PUNISHMENT>", "<SENTENCE>",
    "<COMPANY_LAW>", "<DIRECTOR>", "<TAX_EVASION>", "<GST>", "<MERGER>", "<INSOLVENCY>",
    "<CIVIL_PROCEDURE>", "<CRIMINAL_PROCEDURE>", "<SECTION>", "<ARTICLE>", "<CLAUSE>",
    "<LEGAL_NOTICE>", "<AFFIDAVIT>", "<WARRANT>", "<BAIL>", "<STATUTE>", "<ACT>", "<RULE>"
]

num_added_tokens = legalbert_tokenizer.add_tokens(custom_tokens)
print(f"Added {num_added_tokens} custom tokens.")

# Resize model for new tokens
legalbert_model.resize_token_embeddings(len(legalbert_tokenizer))
legalbert_model.eval()
legalbert_model.to(device)


# === Load GloVe Embeddings ===
#print("Loading GloVe...")
#def load_glove_embeddings(glove_file_path):
 #   embeddings = {}
  #  with open(glove_file_path, 'r', encoding='utf-8') as f:
   #     for line in f:
    #        values = line.strip().split()
     #       word = values[0]
      #      vector = np.array(values[1:], dtype='float32')
       #     embeddings[word] = vector
   # return embeddings
#glove_embeddings = load_glove_embeddings(glove_path)



# === Exported items for use in other modules ===
__all__ = [
    "metadata_list",
    "faiss_index",
  #  "glove_embeddings",
   # "glove_data",
    "legalbert_model",
    "legalbert_tokenizer",
    "device"
]

