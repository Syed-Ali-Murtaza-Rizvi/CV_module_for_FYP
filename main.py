from fastapi import FastAPI, UploadFile, File, Form
from face_engine import generate_embedding, compare_embeddings
import json

app = FastAPI()

@app.post("/register-face/")
async def register_face(file: UploadFile = File(...)):
    image_bytes = await file.read()

    embedding = generate_embedding(image_bytes)

    if embedding is None:
        return {"error": "No face detected"}

    return {
        "embedding": embedding.tolist()
    }

@app.post("/verify-face/")
async def verify_face(
    file: UploadFile = File(...),
    stored_embedding: str = Form(...)
):
    image_bytes = await file.read()

    new_embedding = generate_embedding(image_bytes)

    if new_embedding is None:
        return {"error": "No face detected"}

    # Convert string → list
    stored_embedding = json.loads(stored_embedding)

    similarity = compare_embeddings(new_embedding, stored_embedding)

    return {
        "similarity_score": similarity,
        "is_match": similarity > 0.6
    }