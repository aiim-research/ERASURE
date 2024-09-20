import json

class Config:
    def __init__(self,file):
        self.__dict__ = file

    @classmethod
    def from_json(cls, json_file):
        with open(json_file) as file:
            return cls(json.load(file))
        
    
    