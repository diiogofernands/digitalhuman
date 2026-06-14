import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
from tqdm.auto import tqdm

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
device = "cuda"
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    device=device,
    model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16},
    tokenizer_kwargs={"padding_side": "left"},
)

print("Loading the dataset")
ds = load_dataset("asahi417/wikiart-all", split="test")
df = ds.to_pandas()

print("Dataset loaded, defining functions")
mask_na = df.isna()
mask_empty_str = df.apply(lambda col: col.map(lambda x: isinstance(x, str) and x in ("", "[]")))
mask_empty_list = df.apply(
    lambda col: col.map(
        lambda x: (isinstance(x, list) and len(x) == 0)
        or (isinstance(x, np.ndarray) and x.size == 0)
    )
)
mask_all_missing = mask_na | mask_empty_str | mask_empty_list

df_clean = df.copy()
df_clean = df_clean.astype(object)
for col in df_clean.columns:
    df_clean.loc[mask_all_missing[col], col] = "unknown"

def normalize_list_like(x):
    if isinstance(x, (list, np.ndarray)):
        return [str(v) for v in x]
    return [str(x)]

def make_text(r):
    t = str(r["title"])
    a = str(r["artistName"])
    g = normalize_list_like(r["genres"])
    s = normalize_list_like(r["styles"])
    tg = normalize_list_like(r["tags"])
    gp = str(r["group"])
    return (
        "title: " + t
        + " artist: " + a
        + " genres: " + ", ".join(g)
        + " styles: " + ", ".join(s)
        + " tags: " + ", ".join(tg)
        + " group: " + gp
    )

print("Loading FAISS")
dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dim)

metadata = []
batch_size = 16
batch_texts = []
batch_meta = []

for i in tqdm(range(len(df_clean)), desc="Building FAISS index"):
    row = df_clean.iloc[i]
    text = make_text(row)
    meta = {
        "id": row["id"],
        "title": row["title"],
        "artistName": row["artistName"],
    }
    batch_texts.append(text)
    batch_meta.append(meta)
    if len(batch_texts) == batch_size:
        emb = model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        index.add(emb)
        metadata.extend(batch_meta)
        batch_texts = []
        batch_meta = []

if len(batch_texts) > 0:
    emb = model.encode(
        batch_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    index.add(emb)
    metadata.extend(batch_meta)

faiss.write_index(index, "wikiart.index")
with open("wikiart_meta.pkl", "wb") as f:
    pickle.dump(metadata, f)
