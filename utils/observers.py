# import datetime
import os
from shutil import make_archive

from sacred.observers import RunObserver, MongoObserver

DEFAULT_MONGO_PRIORITY = 30


class CustomMongoObserver(MongoObserver):

    @staticmethod
    def create(url=None, db_name='sacred', collection='runs',
               overwrite=None, priority=DEFAULT_MONGO_PRIORITY,
               client=None, **kwargs):
        import pymongo
        import gridfs

        if client is not None:
            assert isinstance(client, pymongo.MongoClient)
            assert url is None, 'Cannot pass both a client and an url.'
        else:
            client = pymongo.MongoClient(url, **kwargs)
        database = client[db_name]
        if collection in MongoObserver.COLLECTION_NAME_BLACKLIST:
            raise KeyError('Collection name "{}" is reserved. '
                           'Please use a different one.'.format(collection))
        runs_collection = database[collection]
        metrics_collection = database["metrics"]
        fs = gridfs.GridFS(database)
        return CustomMongoObserver(runs_collection,
                                   fs, overwrite=overwrite,
                                   metrics_collection=metrics_collection,
                                   priority=priority)


class StoreOnExitObserver(RunObserver):

    @staticmethod
    def create(ex, base_name):
        return StoreOnExitObserver(ex)

    def __init__(self, ex, base_name):
        self.ex = ex
        self.base_name = base_name

    def _store(self, time):
        filename = make_archive(self.base_name, 'zip', self.base_name)
        self.ex.add_artifact(filename, os.path.basename(filename))

    def interrupted_event(self, interrupt_time, status):
        self._store(interrupt_time)

    def completed_event(self, stop_time, result):
        self._store(stop_time)
