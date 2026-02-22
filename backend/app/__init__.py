import sys
from flask import Flask
from flask_restx import Api
from app.config import Config
from app.utils.database import initialize_mongo_db
from app.api.auth_routes import auth_ns
from app.api.image_routes import image_ns


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    try:
        initialize_mongo_db(app)
    except Exception as e:
        app.logger.error(f'Fatal error during startup: {e}')
        sys.exit('Failed to initialize database, stopping server.')

    # Setup Flask-RESTx API
    api = Api(app, version='1.0', title='Wardrobe API', description='A description of Wardrobe API')

    # Register namespaces
    api.add_namespace(auth_ns, path='/api/auth')
    api.add_namespace(image_ns, path='/api/image')

    return app
