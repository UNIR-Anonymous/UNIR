import json

import gridfs
from pymongo import MongoClient

# from sacred.observers import MongoObserver
from .observers import CustomMongoObserver

MONGO_CONF_PATH = '.../resources/mongo.json'


def _get_conf(mongo_path=MONGO_CONF_PATH):
    with open(mongo_path) as file:
        mongo_conf = json.load(file)
    return mongo_conf


def _get_mongo_connection_url(mongo_conf):
    db_user = '{}:{}'.format(mongo_conf['user'], mongo_conf['passwd'])
    db_host = '{}:{}'.format(mongo_conf['host'], mongo_conf['port'])
    return 'mongodb://{}@{}/{}'.format(db_user, db_host, mongo_conf['auth_db'])


def _get_db_name(mongo_conf):
    return mongo_conf['db']


def get_mongo_db(debug, mongo_conf=None, mongo_path=MONGO_CONF_PATH):
    if mongo_conf is None:
        mongo_conf = _get_conf(mongo_path)
    connection_url = _get_mongo_connection_url(mongo_conf)
    client = MongoClient(connection_url)
    db_name = _get_db_name(debug, mongo_conf)
    return client[db_name]


def get_mongo_collection(debug, mongo_conf=None, mongo_path=MONGO_CONF_PATH):
    if mongo_conf is None:
        mongo_conf = _get_conf(mongo_path)
    db = get_mongo_db(debug, mongo_conf)
    coll_name = mongo_conf['collection']
    return db[coll_name]


def get_gridfs(debug, mongo_conf=None):
    if mongo_conf is None:
        mongo_conf = _get_conf()
    return gridfs.GridFS(get_mongo_db(debug, mongo_conf))


def get_mongo_obs(mongo_conf=None):
    if mongo_conf is None:
        mongo_conf = _get_conf()
    db_url = _get_mongo_connection_url(mongo_conf)
    db_name = _get_db_name(mongo_conf)
    return CustomMongoObserver.create(url=db_url, db_name=db_name, collection=mongo_conf['collection'])
