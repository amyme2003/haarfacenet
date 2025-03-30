import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("face-recognition.json")  # Replace with your actual Firebase key
firebase_admin.initialize_app(cred)
db = firestore.client()

# Add test data
doc_ref = db.collection("faces").document("TestUser")
doc_ref.set({"name": "TestUser", "embedding": [0.1, 0.2, 0.3, 0.4]})

print("Test data added!")
