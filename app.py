import numpy as np
import os
import json
from flask import Flask, request, jsonify
from deepface import DeepFace

# Initialize Flask
app = Flask(__name__)

# Ensure images directory exists
os.makedirs("images", exist_ok=True)

# Embeddings file
EMBEDDINGS_FILE = "embeddings.json"

# Load existing embeddings from file
def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r") as f:
            return json.load(f)
    return {}

# Save embeddings to file
def save_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings, f)

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
        embeddings = load_embeddings()
        embeddings[name] = embedding  # Store as a list
        save_embeddings(embeddings)
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
        embeddings = load_embeddings()

        best_match = {"name": "Unknown", "distance": float("inf")}

        for name, stored_embedding in embeddings.items():
            stored_embedding = np.array(stored_embedding)  # Convert back to numpy array
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