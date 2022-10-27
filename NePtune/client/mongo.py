from mongoengine import connect


class MongoDB:
    host = '192.168.249.1'
    port = 30019

    user = 'neptune'
    dbname = 'NePtune'
    passwd = 'neptune2022@#$'
    authentication_source=''

    @classmethod
    def connect(cls):
        connect(db=cls.dbname, host=cls.host, port=cls.port, username=cls.user, password=cls.passwd,
                authentication_source="NePtune")
