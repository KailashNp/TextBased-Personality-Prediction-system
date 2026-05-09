import pandas as pd

# Load dataset
df = pd.read_csv("data/data-final.csv", sep="\t")
df = df.sample(n=5000, random_state=42)
print("Loaded dataset shape:", df.shape)

# -----------------------------
# Select only personality columns
# -----------------------------

personality_cols = [col for col in df.columns if any(prefix in col for prefix in ["EXT","EST","AGR","CSN","OPN"])]

df = df[personality_cols]

# Remove rows with missing values
df = df.dropna()

# -----------------------------
# Convert numerical answers into text tokens
# -----------------------------

def convert_to_text(row):
    tokens = []
    for col in personality_cols:
        tokens.append(f"{col}_{int(row[col])}")
    return " ".join(tokens)

df["text"] = df.apply(convert_to_text, axis=1)

# -----------------------------
# Create personality labels
# -----------------------------

traits = ["EXT","EST","AGR","CSN","OPN"]

for trait in traits:
    trait_cols = [c for c in personality_cols if c.startswith(trait)]
    score = df[trait_cols].sum(axis=1)
    median = score.median()
    df[trait] = (score > median).astype(int)

# Rename traits to match your project
df.rename(columns={
    "CSN":"CON",
    "EST":"NEU"
}, inplace=True)

# Keep only required columns
final_df = df[["text","OPN","CON","EXT","AGR","NEU"]]

# Save converted dataset
final_df.to_csv("data/mypersonality_sample.csv", index=False)

print("Dataset converted successfully.")