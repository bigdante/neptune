from mongoengine import connect


class MongoDB:
    host = '192.69.26.238'
    port = 30019
    user = 'neptune'
    dbname = 'NePtune'
    passwd = 'neptune2022'

    # host = '166.111.7.106'
    # port = 30020
    # user = 'xll'
    # dbname = 'NePtune'
    # passwd = 'xllKEG2022'

    authentication_source = ''
    alias_remote = 'NePtune'
    @classmethod
    def connect(cls):
        connect(db=cls.dbname, host=cls.host, port=cls.port, username=cls.user, password=cls.passwd,
                authentication_source="NePtune",alias=cls.alias_remote)