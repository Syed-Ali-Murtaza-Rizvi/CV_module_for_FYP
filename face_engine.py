import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize model (CPU mode)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)  # -1 means CPU


def generate_embedding(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces = app.get(img)

    if len(faces) == 0:
        return None

    embedding = faces[0].embedding
    return embedding


def compare_embeddings(emb1, emb2):
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    similarity = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    return float(similarity)