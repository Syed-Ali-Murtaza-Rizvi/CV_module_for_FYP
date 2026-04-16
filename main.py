from fastapi import FastAPI, UploadFile, File
from face_engine import generate_embedding, compare_embeddings

app = FastAPI()


@app.post("/generate-embedding/")
async def create_embedding(file: UploadFile = File(...)):
    image_bytes = await file.read()
    embedding = generate_embedding(image_bytes)

    if embedding is None:
        return {"error": "No face detected"}

    return {"embedding": embedding.tolist()}


@app.post("/compare/")
async def compare_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    img1 = await file1.read()
    img2 = await file2.read()

    emb1 = generate_embedding(img1)
    emb2 = generate_embedding(img2)

    if emb1 is None or emb2 is None:
        return {"error": "Face not detected in one or both images"}

    similarity = compare_embeddings(emb1, emb2)

    return {
        "similarity_score": similarity,
        "is_match": similarity > 0.6
    }