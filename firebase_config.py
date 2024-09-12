import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    # Path to your service account key file
    cred = credentials.Certificate('path/to/your-service-account.json')
    firebase_admin.initialize_app(cred)

    # Initialize Firestore (optional, in case you want to store OTPs in Firestore)
    return firestore.client()
