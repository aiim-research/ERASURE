import json
from jsonc_parser.parser import JsoncParser

from erasure.utils.config.composer import compose

class Config:
    def __init__(self,file):
        self.__dict__ = file

    @classmethod
    def from_json(cls, json_file):
        return cls(compose(JsoncParser.parse_file(json_file)))
        #with open(json_file) as file:
        #    return cls(compose(json.load(file)))
        
    
        
    
    