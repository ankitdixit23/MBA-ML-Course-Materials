# Some libraries we will use
import os
import numpy as np
import tiktoken
from openai import OpenAI
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load your OpenAI API key from an environment variable
# You should set this in your shell before running the script:
# export OPENAI_API_KEY="your_api_key_here"
# Alternatively, you can set is a permanent environment variable associated 
# with a conda environment or virtualenv (called myenv here):
# conda activate myenv
# conda env config vars set OPENAI_API_KEY="your_key_here"
# There are other ways as well; see https://platform.openai.com/docs/api-reference/authentication
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Flag to save figures
dpl = False  # set to False to save figures instead of showing them

# First sentence
text = (
    "Prof. Hansen teaches machine learning to MBA students."
)
print(text)

# For many modern OpenAI models, o200k_base is a common encoding.
# (Tokenization can differ by model/version; this is a just a demo.)
enc = tiktoken.get_encoding("o200k_base")

tokens = enc.encode(text)
print("Number of tokens:", len(tokens))
print("Token IDs:", tokens[:10])

# Show token strings (how the text is chunked)
token_strings = [enc.decode([t]) for t in tokens]
print("\nToken strings:\n", token_strings[:10])

# Second sentence
text = (
    "When Mr. Bilbo Baggins of Bag End announced that he would shortly be "
    "celebrating his eleventy-first birthday with a party of special "
    "magnificence, there was much talk and excitement in Hobbiton."
)
print(text)

# Tokenization
enc = tiktoken.get_encoding("o200k_base")

tokens = enc.encode(text)
print("Number of tokens:", len(tokens))
print("First 10 token IDs:", tokens[:10])

# Show token strings (how the text is chunked)
token_strings = [enc.decode([t]) for t in tokens]
print("\nFirst 10 token strings:\n", token_strings[:10])
print("\nLast 10 token strings:\n", token_strings[-10:])

# Display tokens with separators to see boundaries
view = " | ".join(token_strings)
print(view[:1200], "\n")

# Now let's look at an embedding vector (a numeric representation)
client = OpenAI(api_key=OPENAI_API_KEY)
emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

vec = np.array(emb.data[0].embedding, dtype=float)
print("Embedding dimension:", vec.shape[0])
print("Snippet (first 10 dims):", vec[:10])
print("Norm:", np.linalg.norm(vec))

# The usual measure of similarity between embeddings is "cosine similarity"
# Essentially correlation - would be identical to usual correlation if a and b
# were mean 0
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

###################################################################################
# Classic example
analogy_phrases = [
    "good",
    "bad",
    "happy",
    "sad"
]

emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=analogy_phrases
)

vecs = {p: np.array(d.embedding) for p, d in zip(analogy_phrases, emb.data)}

print("Embedding dimension:", vecs[analogy_phrases[1]].shape[0])
print("good (first 10 dims):", vecs[analogy_phrases[1]][:10])
print("bad (first 10 dims):", vecs[analogy_phrases[2]][:10])

for i in range(len(analogy_phrases)):
    for j in range(i+1, len(analogy_phrases)):
        print(f"{analogy_phrases[i]}  vs  {analogy_phrases[j]}: {cosine_sim(vecs[analogy_phrases[i]], vecs[analogy_phrases[j]]):.3f}")

result = vecs["good"] - vecs["bad"] + vecs["sad"]
for i in range(len(analogy_phrases)):
  print(f"{analogy_phrases[i]}  vs  good+(sad-bad): {cosine_sim(vecs[analogy_phrases[i]], result):.3f}")

####################################################################################
# Second example

phrases = [
    "bank",
    "river bank",
    "investment bank",
    "bank account",
    "piggy bank",
    "sat on the river bank"
]

emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=phrases
)

vecs = [np.array(d.embedding) for d in emb.data]

for i in range(len(phrases)):
    for j in range(i+1, len(phrases)):
        print(f"{phrases[i]}  vs  {phrases[j]}: {cosine_sim(vecs[i], vecs[j]):.3f}")

# "Bag of words embeddings"
def avg_word_embedding(phrase):
    words = phrase.split()  # simple whitespace tokenization (fine for demo)

    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=words
    )

    word_vecs = [np.array(d.embedding) for d in resp.data]
    return np.mean(word_vecs, axis=0)

vecs_bow = [avg_word_embedding(p) for p in phrases]

print("\n=== Averaged Word Embeddings (Bag-of-Words Style) ===")
for i in range(len(phrases)):
    for j in range(i+1, len(phrases)):
        print(f"{phrases[i]:25s} vs {phrases[j]:25s}: {cosine_sim(vecs_bow[i], vecs_bow[j]):.3f}")

########################################################################################
# More elaborate example

texts = [
    # Cluster A: Delivery / shipping / packaging
    "My order arrived late again and the tracking info was wrong.",
    "The package was damaged and the items were broken when delivered.",
    "Delivery was delayed and customer support couldn't locate the shipment.",
    "The box arrived crushed and the contents were missing.",
    "Tracking updated days later; the delivery window was not met.",

    # Cluster B: Billing / pricing / refunds
    "I was charged twice this month and need a refund.",
    "Canceling was confusing and I was billed after I canceled.",
    "The free trial ended and I was charged without a clear reminder.",
    "Your pricing changed and my monthly bill increased unexpectedly.",
    "I requested a refund but the charge is still pending on my card.",
]

# Get embeddings
resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
embeddings = np.array([d.embedding for d in resp.data])

# PCA to 2D
pca = PCA(n_components=2)
points = pca.fit_transform(embeddings)

# Plot (no manual colors)
plt.figure()
plt.scatter(points[:, 0], points[:, 1])

# Annotate
for i, txt in enumerate(texts):
    label = txt[:45] + ("..." if len(txt) > 45 else "")
    plt.annotate(label, (points[i, 0], points[i, 1]))

plt.title("Customer Feedback Embeddings (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, "toy_clustering_with_embeddings.png"))
    plt.close()

# Toy classifier from embeddings
shipping_idx = list(range(5))
billing_idx  = list(range(5, 10))

ship_centroid = embeddings[shipping_idx].mean(axis=0)
bill_centroid = embeddings[billing_idx].mean(axis=0)

for i, txt in enumerate(texts):
    s_ship = cosine_sim(embeddings[i], ship_centroid)
    s_bill = cosine_sim(embeddings[i], bill_centroid)
    label = "Shipping/Delivery" if s_ship > s_bill else "Billing/Pricing"
    print(f"{label:16s}  ship={s_ship:.3f}  bill={s_bill:.3f}  | {txt}")