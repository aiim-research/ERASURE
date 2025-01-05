from abc import ABC, abstractmethod
import random
from torch.utils.data import Subset
from erasure.core.base import Configurable
from .Dataset import DatasetWrapper
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

class DataSplitter(ABC):
    def __init__(self, ref_data,parts_names):
        self.ref_data = ref_data
        self.parts_names = parts_names
    
    @abstractmethod
    def split_data(self, data):
        pass

    def set_source(self, datasource):
        self.source = datasource

    
class DataSplitterPercentage(DataSplitter):
    def __init__(self, percentage, parts_names, ref_data = 'all', shuffle=True):
        super().__init__(ref_data,parts_names) 
        self.percentage = percentage
        self.shuffle = shuffle

    def split_data(self,partitions):
        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else self.source.get_extended_wrapper(Subset(partitions['all'].data, partitions[self.ref_data]))

        total_size = len(ref_data.data)
        split_point = int(total_size * self.percentage)

        indices = list(range(total_size)) if not self.shuffle else torch.randperm(total_size)

        split_indices_1 = indices[:split_point]
        split_indices_2 = indices[split_point:]

        partitions[self.parts_names[0]] = split_indices_1
        partitions[self.parts_names[1]] = split_indices_2

        return partitions
    
    
class DataSplitterConcat(DataSplitter):
    def __init__(self, concat_splits, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.concat_splits = concat_splits

    def split_data(self, partitions):
        unified_indices = []

        for split in self.concat_splits:
            if split in partitions:
                unified_indices.extend(partitions[split])
            else:
                raise KeyError(f"Partition '{split}' not found in partitions.")

        partitions[self.parts_names[0]] = unified_indices


        return partitions


class DataSplitterClass(DataSplitter):
    def __init__(self, label, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.label = label


    def split_data(self,partitions):

        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else self.source.get_extended_wrapper(Subset(partitions['all'].data, partitions[self.ref_data]))
        
        filtered_indices = [
            idx for idx in range(len(ref_data))  
            if ref_data[idx][1] == self.label  
        ]
        

        other_indices = [
            idx for idx in range(len(ref_data)) 
            if idx not in filtered_indices
        ]

        partitions[self.parts_names[0]] = filtered_indices 
        partitions[self.parts_names[1]] = other_indices

        return partitions


class DataSplitterNSamples(DataSplitter):
    def __init__(self, n_samples, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.n_samples = n_samples

    def split_data(self,partitions):
        
        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else self.source.get_extended_wrapper(Subset(partitions['all'].data, partitions[self.ref_data]))
        
        split_point = self.n_samples if self.n_samples is not None else 0

        indices = [ idx for idx in range(len(ref_data)) ]
        
        split_indices_1 = indices[:split_point]
        split_indices_2 = indices[split_point:]

        partitions[self.parts_names[0]] = split_indices_1
        partitions[self.parts_names[1]] = split_indices_2

        return partitions
    
class DataSplitterList(DataSplitter):
    def __init__(self, samples_ids, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.samples_ids = samples_ids

    def split_data(self,partitions):
        
        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else self.source.get_extended_wrapper(Subset(partitions['all'].data, partitions[self.ref_data]))

                
        indices = ref_data.data.indices

        if self.samples_ids:
            split_indices_1 = self.samples_ids
            split_indices_2 = [id for id in indices if id not in self.samples_ids]


        partitions[self.parts_names[0]] = split_indices_1
        partitions[self.parts_names[1]] = split_indices_2

        return partitions
    
class DataSplitterByZ(DataSplitter):
    def __init__(self, z_label, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.z_label = z_label


    def split_data(self,partitions):

        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else self.source.get_extended_wrapper(Subset(partitions['all'].data, partitions[self.ref_data]))
        
        dataloader = DataLoader(ref_data, batch_size=5000)

        filtered_indices = []
        current_index = 0  # To track the global index of each batch
        import torch
        for batch in tqdm(dataloader, desc="Filtering Data"):
            _, _, Z = batch  # Assuming Z is the third element in the batch
            
            # Apply a boolean mask to find matching Z values
            mask = (Z == self.z_label)
            matching_indices = torch.nonzero(mask, as_tuple=True)[0]  # Indices within the batch
            
            # Convert batch indices to global indices and store them
            filtered_indices.extend((current_index + matching_indices).tolist())
            
            # Update the global index for the next batch
            current_index += len(Z)

        other_indices = [
            idx for idx in range(len(ref_data)) 
            if idx not in filtered_indices
        ]

        print(len(filtered_indices))

        partitions[self.parts_names[0]] = filtered_indices 
        partitions[self.parts_names[1]] = other_indices

        return partitions