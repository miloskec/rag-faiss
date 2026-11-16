import hashlib

import numpy as np


class FakeEmbedder:
    def __init__(self, vectors=None):
        self.vectors = vectors
        self.inputs = []

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        self.inputs.extend(texts)
        if self.vectors is not None:
            return np.asarray(self.vectors, dtype=np.float32)
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class HashEmbedder:
    def __init__(self, dimension=64):
        self.dimension = dimension

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in text.lower().replace(".", " ").replace(":", " ").split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                weight = 4.0 if token.startswith("aster-") else 1.0
                vector[int.from_bytes(digest[:4], "big") % self.dimension] += weight
            norm = np.linalg.norm(vector)
            if norm:
                vector /= norm
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores


class CapturingModelHandler:
    prompt_style = "plain"

    def __init__(self):
        self.prompt = None

    def generate_response(self, prompt):
        self.prompt = prompt
        return "deterministic test response"
