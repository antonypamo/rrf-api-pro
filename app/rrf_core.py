import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math

def entropy(text):
    words = text.split()
    counts = Counter(words)
    total = len(words)
    return -sum((c/total) * math.log(c/total + 1e-9) for c in counts.values())

def coherence_score(embeddings):
    sims = cosine_similarity(embeddings)
    return float(np.mean(sims))

def build_graph(n):
    return {i: [(i+j)%n for j in range(1,6)] for i in range(n)}

def rrf_transform(embeddings):
    graph = build_graph(len(embeddings))
    new_embeddings = embeddings.copy()

    for i, neighbors in graph.items():
        avg = np.mean(embeddings[neighbors], axis=0)
        new_embeddings[i] = 0.7 * embeddings[i] + 0.3 * avg

    return new_embeddings

def rrf_score(text, embeddings):
    C = coherence_score(embeddings)
    S = 1 - np.std([np.linalg.norm(e) for e in embeddings])
    E = 1 - entropy(text)/5
    return float(0.4*C + 0.3*S + 0.3*E)
