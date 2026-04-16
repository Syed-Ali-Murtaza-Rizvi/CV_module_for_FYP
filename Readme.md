 README.md
# 🧠 Face Recognition Service (CPU-Based)

An independent AI-powered Face Recognition module built using **FastAPI** and **InsightFace**, designed for integration into systems like Smart Attendance, Event Management, or Identity Verification platforms.

This service performs:
- Face Detection
- Face Embedding Generation
- Face Comparison using Cosine Similarity

---

## 🚀 Features

- 🔍 Detects faces from uploaded images
- 🧬 Generates 512-dimensional face embeddings
- 🔗 Compares two faces and returns similarity score
- ⚡ FastAPI-based lightweight REST API
- 💻 CPU-based (no GPU required)
- 🔌 Easily integratable with any backend (Django, Node.js, etc.)

---

## 🏗 Project Structure
face_service/
│
├── main.py # FastAPI app (API endpoints)
├── face_engine.py # Core AI logic (embedding + comparison)
├── requirements.txt # Dependencies
└── README.md


---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/face-recognition-service.git
cd face-recognition-service
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
Or manually:

pip install fastapi uvicorn insightface==0.7.3 onnxruntime opencv-python numpy python-multipart
▶️ Running the Service
uvicorn main:app --reload
Server will start at:

http://127.0.0.1:8000
📚 API Documentation
Interactive Swagger UI available at:

http://127.0.0.1:8000/docs
🔌 API Endpoints
🔹 1. Generate Face Embedding
POST /generate-embedding/

Request:
Form Data → file (image)

Response:
{
  "embedding": [0.123, 0.456, ...]
}
If no face detected:

{
  "error": "No face detected"
}
🔹 2. Compare Two Faces
POST /compare/

Request:
file1 → Image 1

file2 → Image 2

Response:
{
  "similarity_score": 0.82,
  "is_match": true
}
🧮 Face Matching Logic
This service uses Cosine Similarity:

similarity = (A · B) / (||A|| × ||B||)
Default Threshold:
similarity > 0.6 → Match
similarity ≤ 0.6 → Not Match
🧠 Technology Stack
FastAPI

InsightFace (ArcFace model)

ONNX Runtime (CPU)

OpenCV

NumPy

💡 Use Cases
🎓 Smart Attendance Systems

🎟 Event Check-in Systems

🔐 Identity Verification

🏫 University Projects (FYP)

🧑‍💼 Employee Attendance Systems

⚠️ Limitations
No liveness detection (can be added later)

CPU-based (slower than GPU for large-scale systems)

Assumes one face per image

🔮 Future Improvements
Add liveness detection (anti-spoofing)

GPU acceleration support

Batch face processing

Face alignment improvements

Logging & monitoring

🧪 Testing
Use Swagger UI or tools like Postman to test:

Same image → similarity ≈ 1.0

Same person → similarity > 0.6

Different person → similarity < 0.5

📜 License
This project is for educational purposes (FYP).

👨‍💻 Author
Murtaza Rizvi

⭐ Contribution
Feel free to fork and improve this module!

