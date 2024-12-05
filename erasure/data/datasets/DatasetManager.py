from erasure.core.base import Configurable
from erasure.core.factory_base import *
from fractions import Fraction
import numpy as np
from .Dataset import Dataset
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate


class DatasetManager(Configurable):

    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx, local_ctx)
        self.partitions = {}
        self.info(self.params['DataSource'])
        datasource = get_instance_config(self.params['DataSource'])
        self.partitions['all'] = datasource.validate_and_create_data()
        self.parts_cfgs = self.params['partitions']
        self.info(self.partitions['all'].data)
        self.batch_size=self.params['batch_size']
        self.name = datasource.get_name()

        #count number of classes in the dataset
        self.n_classes = self.partitions['all'].get_n_classes()

        self.__prepare_partitions()

    def __prepare_partitions(self):
        
        for split in self.parts_cfgs: 
           splitted_data = get_instance_config(split)
           
           self.partitions = splitted_data.split_data(self.partitions)

        self.info(list(self.partitions.keys()))

    def get_loader_for_ids(self, list_ids):

        dataset = self.partitions['all']

        main_loader = DataLoader(Subset(dataset.data, list_ids), batch_size=self.batch_size, collate_fn = skip_nones_collate, shuffle=False, worker_init_fn = torch.initial_seed())

        return main_loader
           
           
    def get_loader_for(self, split_id, fold_fraction = None):

        fold_fraction = None

        dataset = self.partitions['all'].data if split_id == 'all' else Dataset(Subset(self.partitions['all'].data, self.partitions[split_id])).data

        num_samples = len(dataset)

        if split_id == 'train':
            self.info(f"TRAINING WITH {num_samples} samples")

        if fold_fraction is not None:
            number_of_folds = fold_fraction.denominator

            fold_id = fold_fraction.numerator

            fold_size = num_samples // number_of_folds

            indices = np.arange(num_samples)

            folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(number_of_folds)]

            fold_indices = folds[fold_id]

            main_indices = np.concatenate([folds[i] for i in range(number_of_folds) if i != fold_id])
    
            main_loader = DataLoader(Subset(dataset, main_indices), batch_size=self.batch_size, collate_fn = skip_nones_collate,shuffle=False, worker_init_fn = torch.initial_seed())
            fold_loader = DataLoader(Subset(dataset, fold_indices), batch_size=self.batch_size, collate_fn = skip_nones_collate,shuffle=False, worker_init_fn = torch.initial_seed())

        else:
            main_loader = DataLoader(dataset, batch_size=self.batch_size,  collate_fn = skip_nones_collate, shuffle=False, worker_init_fn = torch.initial_seed())
            fold_loader = None

        return main_loader, fold_loader

    def revise_split(self, split_id, ids_list, additive=False):
        print(f"REVISING SPLIT {split_id} with {len(ids_list)} samples")
        print(f"ids_list: {ids_list}")
        if not additive:
            print(self.partitions[split_id])
            print(ids_list)
            self.partitions[split_id] = [sample for sample in self.partitions[split_id] if sample not in ids_list]
        else:
            self.partitions[split_id] = list(set(self.partitions[split_id] + ids_list))
        
    def get_dataset_from_partition(self, split_id):
        """
        Returns the dataset corresponding to the given partition ID.

        Args:
            partition_id (str): The ID of the partition.

        Returns:
            Dataset: The dataset corresponding to the partition ID.
        """
        if split_id in self.partitions:
            return self.partitions['all'].data if split_id == 'all' else Dataset(Subset(self.partitions['all'].data, self.partitions[split_id])).data
        else:
            raise ValueError(f"Partition ID '{split_id}' not found in partitions.")
   

def skip_nones_collate(batch):
    batch = [item for item in batch if item is not None]
    return default_collate(batch)

