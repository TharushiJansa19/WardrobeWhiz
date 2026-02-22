from pymongo import MongoClient


def initialize_mongo_db(app):
    mongo_client = MongoClient(app.config['MONGO_URI'])
    app.mongo_db = mongo_client[app.config['MONGO_DB_NAME']]
    try:
        mongo_client.admin.command('ping')
        app.logger.info('MongoDB connected successfully.')
    except Exception as e:
        raise (e)
