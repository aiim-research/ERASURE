from .datasource import DataSource
from erasure.data.datasets.Dataset import DatasetWrapper 
from torch.utils.data import ConcatDataset
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from datasets import load_dataset, DatasetDict, concatenate_datasets


class HFDatasetWrapper(DatasetWrapper):
    def __init__(self, data, preprocess,label,data_columns):
        super().__init__(data,preprocess)
        self.label = label
        self.data_columns = data_columns

    def __realgetitem__(self, index: int):
        sample = self.data[index]

        X = [value for key,value in sample.items() if key in self.data_columns]
        y = sample[self.label]

        return X,y

class HFDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.dataset = None
        self.path = self.local_config['parameters']['path']
        self.configuration = self.local_config.get("configuration","")
        self.label = self.local_config['parameters']['label']
        self.data_columns = self.local_config['parameters']['data_columns']
        self.to_encode = self.local_config['parameters']['to_encode']
        self.classes = self.local_config['parameters']['classes']

    def get_name(self):
        return self.path.split("/")[-1] 

    def create_data(self):
        ds = load_dataset(self.path,self.configuration)            

        self.label_mappings = {}
        for column_to_encode in self.to_encode:
            unique_labels = ds['train'].unique(column_to_encode)
            #unique_labels_sorted = sorted(unique_labels)
            self.label_mappings[column_to_encode] = {orig_label: new_label for new_label, orig_label in enumerate(unique_labels)}

            def encode_func(example, col=column_to_encode):
                example[col] = self.label_mappings[col][example[col]]
                return example
            
            for split in ds.keys():
                ds[split] = ds[split].map(encode_func)

        if isinstance(ds, dict) or hasattr(ds, "keys"):
            splits = [ds[split] for split in ds.keys()]
        else:
            splits = [ds]

        concat = ConcatDataset(splits)

        concat.classes = splits[0].unique(self.label) if self.classes == -1 else self.classes

        dataset = self.get_wrapper(concat)

        return dataset
    

    def encode_label(self,sample):
        sample["label"] = self.label_mapping[sample["label"]]
        return sample

    def get_simple_wrapper(self, data):
        return HFDatasetWrapper(data, self.preprocess, self.label, self.data_columns)
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['path'] = self.local_config['parameters']['path']
        self.local_config['parameters']['configuration'] = self.local_config.get("configuration","")
        self.local_config['parameters']['label'] = self.local_config['parameters'].get('label',"")
        self.local_config['parameters']['data_columns'] = self.local_config['parameters']['data_columns']
        self.local_config['parameters']['to_encode'] = self.local_config['parameters'].get("to_encode",[])
        self.local_config['parameters']['classes'] = self.local_config['parameters'].get("classes",-1)


class SpotifyHFDataSource(HFDataSource):

    def create_data(self):
        ds = load_dataset(self.path,self.configuration)    

        df = ds['train'].to_pandas()

        unique_artists = set()
        for artist_list in df['artists']:
            if artist_list is not None:
                artists = artist_list.split(';')  
                unique_artists.update(artists)       

        artist_to_id = {artist: idx for idx, artist in enumerate(unique_artists)}

        def map_artists_to_ids(artist_list):
            if not artist_list: 
                return []
            return [artist_to_id[artist] for artist in artist_list.split(';') if artist in artist_to_id]


        df['artist_ids'] = df['artists'].apply(map_artists_to_ids)

        ds['train'] = ds['train'].remove_columns("artists")  
        ds['train'] = ds['train'].add_column("artists", df['artists'].tolist())


        self.label_mappings = {}
        for column_to_encode in self.to_encode:
            unique_labels = ds['train'].unique(column_to_encode)
            #unique_labels_sorted = sorted(unique_labels)
            self.label_mappings[column_to_encode] = {orig_label: new_label for new_label, orig_label in enumerate(unique_labels)}

            def encode_func(example, col=column_to_encode):
                example[col] = self.label_mappings[col][example[col]]
                return example
            
            for split in ds.keys():
                ds[split] = ds[split].map(encode_func)

        if isinstance(ds, dict) or hasattr(ds, "keys"):
            splits = [ds[split] for split in ds.keys()]
        else:
            splits = [ds]

        concat = ConcatDataset(splits)

        concat.classes = splits[0].unique(self.label) if self.classes == -1 else self.classes

        dataset = self.get_wrapper(concat)

        return dataset