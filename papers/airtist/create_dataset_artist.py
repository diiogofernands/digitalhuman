from datasets import load_dataset
from collections import defaultdict
import json
import math

# Load dataset
print("Loading dataset")
dataset_art = load_dataset("asahi417/wikiart-all")
df_art = dataset_art["test"]

MAX_PAINTINGS_PER_CHUNK = 1

# Group paintings by artist
artists = defaultdict(lambda: {"artistName": "", "artistUrl": "", "paintings": []})

for idx, row in enumerate(df_art):
    artist_id = row["artistUrl"]
    artists[artist_id]["artistName"] = row["artistName"]
    artists[artist_id]["artistUrl"] = row["artistUrl"]
    artists[artist_id]["paintings"].append({
        "id": idx,
        "url": row["url"],
        "title": row["title"]
    })

# Write to JSONL with chunking
total_lines = 0
with open("wikiart_artists_chunked.jsonl", "w") as f:
    line_id = 0
    for artist_url, data in artists.items():
        paintings = data["paintings"]
        num_paintings = len(paintings)
        
        # Calculate number of chunks needed
        num_chunks = math.ceil(num_paintings / MAX_PAINTINGS_PER_CHUNK)
        
        # Split evenly across chunks
        chunk_size = math.ceil(num_paintings / num_chunks)
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_paintings)
            chunk_paintings = paintings[start:end]
            
            record = {
                "id": line_id,
                "artistName": data["artistName"],
                "artistUrl": data["artistUrl"],
                "chunk": chunk_idx + 1,
                "total_chunks": num_chunks,
                "paintings": chunk_paintings
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            line_id += 1
            total_lines += 1

print(f"Total lines (prompts): {total_lines}")
print(f"Total artists: {len(artists)}")
print(f"Total paintings: {sum(len(a['paintings']) for a in artists.values())}")