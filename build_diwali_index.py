import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from tqdm import tqdm

print("Building Diwali Cultural Knowledge Index...")
print("=" * 80)

# Load Diwali dataset
print("\n1. Loading Diwali dataset...")
df = pd.read_csv("data/diwali.csv")
print(f"   Loaded {len(df)} entries")
print(f"   Facets: {df['facet'].unique().tolist()}")
print(f"   States: {df['state'].nunique()} states")

# Create searchable text for each entry
print("\n2. Creating searchable text representations...")
df['search_text'] = df.apply(
    lambda row: f"{row['concept']} from {row['state']}. {row['description']}",
    axis=1
)

# Extract cultural elements for quick lookup
print("\n3. Extracting cultural elements...")

# Extract names (persons mentioned in descriptions)
# For now, we'll create a curated list and extract from dataset later
# This can be enhanced with NER

# Extract items/concepts
concepts_by_facet = df.groupby('facet')['concept'].apply(list).to_dict()
print(f"   Organized {len(concepts_by_facet)} facets")

# Extract by state
concepts_by_state = df.groupby('state')['concept'].apply(list).to_dict()
print(f"   Organized {len(concepts_by_state)} states")

# Load sentence transformer model
print("\n4. Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
print("\n5. Creating embeddings for all entries...")
print("   This may take a few minutes...")

batch_size = 32
embeddings = []

for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df['search_text'].iloc[i:i+batch_size].tolist()
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings)
print(f"   Created embeddings with shape: {embeddings.shape}")

# Save index
print("\n6. Saving index...")
os.makedirs("data", exist_ok=True)

index_data = {
    'embeddings': embeddings,
    'dataframe': df,
    'concepts_by_facet': concepts_by_facet,
    'concepts_by_state': concepts_by_state,
    'model_name': 'all-MiniLM-L6-v2'
}

with open("data/diwali_index.pkl", "wb") as f:
    pickle.dump(index_data, f)

print(f"   Saved to: data/diwali_index.pkl")
print(f"   Index size: {os.path.getsize('data/diwali_index.pkl') / 1024 / 1024:.2f} MB")

# Create a quick reference guide from Diwali dataset
print("\n7. Extracting cultural elements from Diwali dataset...")

# Extract concepts from each facet
names = concepts_by_facet.get('names', [])
places = concepts_by_facet.get('places', [])
foods = concepts_by_facet.get('food', [])
festivals = concepts_by_facet.get('festivals', [])
drinks = concepts_by_facet.get('drinks', [])
clothing = concepts_by_facet.get('clothing', [])
textiles = concepts_by_facet.get('textiles', [])
dances = concepts_by_facet.get('dance', [])
arts = concepts_by_facet.get('arts', [])
languages = concepts_by_facet.get('languages', [])
games = concepts_by_facet.get('games', [])

print(f"   Extracted {len(names)} names")
print(f"   Extracted {len(places)} places")
print(f"   Extracted {len(foods)} foods")
print(f"   Extracted {len(festivals)} festivals")
print(f"   Extracted {len(drinks)} drinks")
print(f"   Extracted {len(clothing)} clothing items")
print(f"   Extracted {len(textiles)} textiles")

cultural_ref = {
    'names': names,
    'places': places,
    'foods': foods,
    'festivals': festivals,
    'drinks': drinks,
    'clothing': clothing,
    'textiles': textiles,
    'dances': dances,
    'arts': arts,
    'languages': languages,
    'games': games,
    'all_concepts': list(concepts_by_facet.keys()),
    'states': list(concepts_by_state.keys())
}

with open("data/cultural_reference.pkl", "wb") as f:
    pickle.dump(cultural_ref, f)

print(f"   Saved to: data/cultural_reference.pkl")

# Summary
print("\n" + "=" * 80)
print("INDEX BUILD COMPLETE!")
print("=" * 80)
print(f"""
Summary:
  - Total entries indexed: {len(df)}
  - Embedding dimensions: {embeddings.shape[1]}
  - Facets available: {len(concepts_by_facet)}
  - States covered: {len(concepts_by_state)}

Cultural elements extracted from Diwali dataset:
  - Names: {len(names)}
  - Places: {len(places)}
  - Foods: {len(foods)}
  - Drinks: {len(drinks)}
  - Festivals: {len(festivals)}
  - Clothing: {len(clothing)}
  - Textiles: {len(textiles)}
  - Dances: {len(dances)}
  - Arts: {len(arts)}
  - Languages: {len(languages)}
  - Games: {len(games)}
  - Total concepts: {sum(len(v) for v in cultural_ref.values() if isinstance(v, list))} items

Files created:
  - data/diwali_index.pkl (main index with embeddings)
  - data/cultural_reference.pkl (extracted cultural concepts)

Ready for RAG-based adaptation using authentic Diwali dataset concepts!
""")
print("=" * 80)
