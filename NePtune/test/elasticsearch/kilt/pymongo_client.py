from pymongo import MongoClient
from bson.codec_options import DEFAULT_CODEC_OPTIONS
import socket

CUR_IP = socket.gethostbyname(socket.gethostname())
options = DEFAULT_CODEC_OPTIONS.with_options(unicode_decode_error_handler='ignore')


class Mongo106:
    ip = '127.0.0.1'
    port = 30020
    dbname = 'NePtune'
    username = 'neptune'
    password = 'neptune2022'
    mechanism = 'SCRAM-SHA-1'


servers = {106: Mongo106()}


class mongo:
    def __init__(self, server=106):
        """
            :param server: {173, 105, 106}
        """
        if server and CUR_IP != "166.111.7." + str(server):
            m = servers.get(server)
            assert m  # m is not None
            self.client = MongoClient(host=m.ip, port=m.port, unicode_decode_error_handler='ignore')
            self.db = self.client.get_database(m.dbname, codec_options=options)
            self.db.authenticate(name='kegger_bigsci', password='datiantian123!@#', mechanism=m.mechanism)
        else:
            server = int(CUR_IP.split('.')[-1])
            m = servers.get(server)
            assert m  # m is not None
            self.client = MongoClient("localhost:" + str(m.port),
                                      username=m.username,
                                      password=m.password,
                                      authSource=m.dbname,
                                      authMechanism=m.mechanism,
                                      unicode_decode_error_handler='ignore')
            self.db = self.client.get_database(m.dbname, codec_options=options)

        if server == 173:
            self.aminer_person_col = self.db['person_n']
            self.mag_person_col = self.db['MAG_Authors']
            self.person_map_col_l1 = self.db['OAG_person_map']
            self.person_map_col_l2 = self.db['OAG_person_map_L2']
            self.paper_map_col = self.db['OAG_map_L2']
            self.aminer_coauthor_col = self.db['person_coauthors']
            self.aminer_pub_col = self.db['publication_dupl']
            self.mag_person_col_old = self.db['mag_person']
            self.mag_id_translation = self.db['mag_id_translation']
            self.paper_map_col_l4_999 = self.db['OAG_L4_th999']
            self.venue_map_col = self.db['Venue_Merge_final']
            self.mag_venue = self.db['MAG_Venue']
            self.mag_paper_col = self.db['MAG_Papers']
            self.person_map_col_l3 = self.db['Author_match_merged_dupl']
            self.mag_coauthors = self.db['MAG_Coauthors']
            self.aminer_coauthors = self.db['person_coauthors']
            self.aminer_venue = self.db['venue_dupl']
            self.mag_venue = self.db['MAG_Venue']
            self.mag_pub_col = self.db['MAG_Papers']
        elif server == 106:
            self.CONCEPT = self.db.atlas_concept
            self.CONCEPT_DETAIL = self.db.atlas_concept_detail
            self.ENTITY = self.db.atlas_entity
            self.ENTITY_DETAIL = self.db.atlas_entity_detail
            self.ENTITY_INFO = self.db.atlas_entity_info
            self.THUMB_DETAIL = self.db.atlas_thumb_detail
            self.USER = self.db.atlas_user
            self.OP_LOG = self.db.atlas_op_log

            self.concept_unified = self.db.concept_unified
