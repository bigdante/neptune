from mongoengine import connect


class MongoDB:
    host = '127.0.0.1'
    # host = '192.168.'
    port = 30019
    user = 'xll'
    dbname = 'NePtune'
    passwd = 'xllKEG2022'

    authentication_source = ''
    alias_remote = 'NePtune'
    @classmethod
    def connect(cls):
        connect(db=cls.dbname, host=cls.host, port=cls.port, username=cls.user, password=cls.passwd,
                authentication_source="NePtune",alias=cls.alias_remote)