from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")

def get_embedding(text):
    return model.encode(text)
