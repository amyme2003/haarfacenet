import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace
import os

# Initialize Flask
app = Flask(__name__)  


# Initialize Firebase
cred = credentials.Certificate("face-recognition.json")  # Use your actual Firebase JSON key
firebase_admin.initialize_app(cred)
db = firestore.client()

# Ensure images directory exists
os.makedirs("images", exist_ok=True)

# Function to save embeddings to Firestore
def save_embedding(name, embedding):
    embedding = np.array(embedding)
    
    if embedding.shape[0] != 128:
        print(f"❌ Error: {name} embedding has incorrect shape {embedding.shape}, skipping save!")
        return  # Skip saving incorrect embeddings

    db.collection("faces").document(name).set({  # Use consistent collection name
        "name": name,
        "embedding": embedding.tolist()  # Convert to list for Firestore
    })
    print(f"✅ Saved {name} with embedding shape {embedding.shape}")

# Function to retrieve all embeddings from Firestore
def get_all_embeddings():
    docs = db.collection("faces").stream()  # Use the correct collection name
    embeddings = {}

    for doc in docs:
        data = doc.to_dict()
        if "embedding" in data:
            embedding = np.array(data["embedding"])
            if embedding.shape[0] == 128:  # Ensure valid shape
                embeddings[doc.id] = embedding
            else:
                print(f"⚠️ Warning: Skipping {doc.id} due to incorrect embedding size {embedding.shape}")
    
    return embeddings

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    image = request.files.get('image')

    if not name or not image:
        return jsonify({"error": "Missing name or image"}), 400

    image_path = f"images/{name}.jpg"
    image.save(image_path)

    try:
        # Generate face embedding
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet")[0]['embedding']
        save_embedding(name, embedding)
        return jsonify({"message": f"✅ Image added for {name}!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize():
    image = request.files.get('image')

    if not image:
        return jsonify({"error": "Missing image"}), 400

    image_path = "temp.jpg"
    image.save(image_path)

    try:
        # Generate embedding for input image
        embedding = np.array(DeepFace.represent(img_path=image_path, model_name="Facenet")[0]['embedding'])

        stored_embeddings = get_all_embeddings()

        best_match = {"name": "Unknown", "distance": float("inf")}

        for name, stored_embedding in stored_embeddings.items():
            if stored_embedding.shape[0] == 128:  # Ensure valid shape before comparing
                distance = np.linalg.norm(stored_embedding - embedding)

                if distance < best_match["distance"]:
                    best_match = {"name": name, "distance": distance}

        if best_match["distance"] < 10:  # Adjust threshold based on testing
            return jsonify({"name": best_match["name"]}), 200
        else:
            return jsonify({"name": "Unknown"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)