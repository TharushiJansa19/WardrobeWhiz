import os


class Config:
    # Basic application config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_default_secret_key'

    # MongoDB configuration
    MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME', 'wardrobe')

    MONGO_USERNAME = os.environ.get('MONGO_USERNAME')
    MONGO_PASSWORD = os.environ.get('MONGO_PASSWORD')
    MONGO_URI = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.avzbcs4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    FLASK_RUN_HOST = os.environ.get('FLASK_RUN_HOST', '127.0.0.1')
    FLASK_RUN_PORT = int(os.environ.get('FLASK_RUN_PORT', '5000'))

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
