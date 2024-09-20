import os

from erasure.data.data_sources import DataSource


class FileDataSource(DataSource):

    def __init__(self, file_path):
        self.file_path = file_path

    def create_data(self):
        return super().create_data()
    
    ##TO BE IMPLEMENTED

