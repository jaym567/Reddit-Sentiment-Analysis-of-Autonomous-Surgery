import pandas as pd
import re

# --- 1. LOAD DATA ---
df = pd.read_csv("reddit_robotic_surgery.csv")

print(f"Loaded {len(df)} rows ✅")
print("\nSample rows:\n")
print(df.head(5))  # show a few examples

# --- 2. REMOVE BLANK / SHORT TEXTS ---
df = df[df["text"].notna()]              # remove NaN
df = df[df["text"].str.strip() != ""]    # remove empty
df = df[df["text"].str.len() > 20]       # remove super short (like "lol" or "ok")

# --- 3. CLEAN TEXT ---
def clean_text(text):
    # remove URLs
    text = re.sub(r"http\S+|www.\S+", "", text)
    # remove Reddit mentions and markdown
    text = re.sub(r"[\*\[\]\(\)>#]", "", text)
    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# --- 4. SAVE CLEANED DATA ---
df.to_csv("reddit_robotic_surgery_clean.csv", index=False)

print(f"\nSaved cleaned file with {len(df)} rows ✅")
