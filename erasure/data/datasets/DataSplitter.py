from abc import ABC, abstractmethod
import random
from torch.utils.data import Subset
from erasure.core.base import Configurable
from .Dataset import DatasetWrapper
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import hashlib
import numpy as np

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
        indices = partitions[self.ref_data] if self.ref_data != 'all' else list(range(len(partitions[self.ref_data].data)))

        self.total_size = len(indices)
        split_point = int(self.total_size * self.percentage)

        indices = self.get_indices(indices) if self.shuffle else indices

        split_indices_1 = indices[:split_point]
        split_indices_2 = indices[split_point:]

        partitions[self.parts_names[0]] = split_indices_1
        partitions[self.parts_names[1]] = split_indices_2

        return partitions
    
    def get_indices(self, indices):
        seedlist = self.create_seed_list()

        seed = self.get_seed_from_name(self.parts_names[0], seedlist)

        indices = self.shuffle_with_seed(indices, seed)

        return indices
    
    def shuffle_with_seed(self, indices, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        permuted_order = torch.randperm(len(indices), generator=generator).tolist()
        
        shuffled_indices = [indices[i] for i in permuted_order]

        return shuffled_indices
    
    def create_seed_list(self):
        generator = torch.Generator()
        
        # Generate a list of seeds
        seeds = [torch.randint(0, 2**32 - 1, (1,), generator=generator).item() for _ in range(10000)]
        return seeds
    
    def get_seed_from_name(self,name, seed_list):
        hashed_value = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        
        position = hashed_value % len(seed_list)
        
        return seed_list[position]
    
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
            ref_data.data.indices[idx] for idx in range(len(ref_data))  
            if ref_data[idx][1] == self.label  
        ]

        other_indices = [
            ref_data.data.indices[idx] for idx in range(len(ref_data)) 
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
        
        indices = partitions[self.ref_data] if self.ref_data != 'all' else list(range(len(partitions[self.ref_data].data)))
        
        split_point = self.n_samples if self.n_samples is not None else 0
        
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
    def __init__(self, z_labels, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.z_labels = z_labels


class DataSplitterByZ(DataSplitter):
    def __init__(self, z_labels, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.z_labels = z_labels


    def split_data(self,partitions):
        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else self.source.get_extended_wrapper(Subset(partitions['all'].data, partitions[self.ref_data]))        
        
        dataloader = DataLoader(ref_data, batch_size=10000)

        filtered_indices = []
        all_possible_z = []
        current_index = 0  

        for batch in tqdm(dataloader, desc="Filtering Data"):
            _, _, Z = batch
            all_possible_z.extend(Z)
            
            mask = torch.Tensor( np.isin(Z, self.z_labels) )
            matching_indices = torch.nonzero(mask, as_tuple=True)[0]  
            
            filtered_indices.extend((current_index + matching_indices).tolist())
            
            current_index += len(Z)

        all_indices = set(range(len(ref_data)))
        other_indices = list(all_indices - set(filtered_indices))

        filtered_indices = [partitions[self.ref_data][i] for i in filtered_indices]
        other_indices = [partitions[self.ref_data][i] for i in other_indices]

        partitions[self.parts_names[0]] = filtered_indices 
        partitions[self.parts_names[1]] = other_indices


        '''
        all_possible_z = torch.tensor(all_possible_z)
        all_possible_z = torch.unique(all_possible_z)
        all_possible_z = torch.sort(all_possible_z).values
        print("all possible z_labels in the data: ", all_possible_z)

        print("ratio of z_labels in the data: ", len(filtered_indices)/len(other_indices))
        
        '''
        
        return partitions
    
class DataSplitterByZList(DataSplitter):
    def __init__(self, z_labels, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.z_labels = z_labels


    def split_data(self,partitions):
        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else self.source.get_extended_wrapper(Subset(partitions['all'].data, partitions[self.ref_data]))
        
        
        dataloader = DataLoader(ref_data, batch_size=10000)

        filtered_indices = []
        all_possible_z = []
        current_index = 0  

        #### TODO: CAN BE OPTIMIZED
        for batch in tqdm(dataloader, desc="Filtering Data"):
            _, _, Z = batch  # Z is a list of tensors
            Z = torch.stack(Z, dim=1)
            all_possible_z.extend(Z)
            matched_indices = []

            for i, z_tensor in enumerate(Z):
                results = [i for i in self.z_labels if i in z_tensor]

                if len(results)>0:
                    matched_indices.append(current_index + i)  

            filtered_indices.extend(matched_indices)
            current_index += len(Z)  

        all_indices = set(range(len(ref_data)))
        other_indices = list(all_indices - set(filtered_indices))

        filtered_indices = [partitions[self.ref_data][i] for i in filtered_indices]
        other_indices = [partitions[self.ref_data][i] for i in other_indices]

        partitions[self.parts_names[0]] = filtered_indices 
        partitions[self.parts_names[1]] = other_indices

        '''
        all_possible_z = torch.tensor(all_possible_z)
        all_possible_z = torch.unique(all_possible_z)
        all_possible_z = torch.sort(all_possible_z).values
        print("all possible z_labels in the data: ", all_possible_z)

        print("ratio of z_labels in the data: ", len(filtered_indices)/len(other_indices))
        '''
        return partitions
    

class DataSplitterAnyZisIn(DataSplitter):
    def __init__(self, z_labels, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.z_labels = z_labels


    def split_data(self, partitions):
        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else self.source.get_extended_wrapper(Subset(partitions['all'].data, partitions[self.ref_data]))

        dataloader = DataLoader(ref_data, batch_size=10000)

        filtered_indices = []
        all_possible_z = []
        current_index = 0  

        for batch in tqdm(dataloader, desc="Filtering Data"):
            _, _, Z = batch  
            if not isinstance(Z, list):
                Z = [Z]
            Z = torch.stack(Z, dim=1)  
            all_possible_z.extend(Z)

            z_labels_tensor = torch.tensor(self.z_labels)

            matched_indices = []

            for i, z_tensor in enumerate(Z):
                if torch.isin(z_tensor, z_labels_tensor).any():
                    matched_indices.append(current_index + i)  

            filtered_indices.extend(matched_indices)
            current_index += len(Z)  
            
            all_indices = set(range(len(ref_data)))
            other_indices = list(all_indices - set(filtered_indices))

            filtered_indices = [partitions[self.ref_data][i] for i in filtered_indices]
            other_indices = [partitions[self.ref_data][i] for i in other_indices]

            partitions[self.parts_names[0]] = filtered_indices 
            partitions[self.parts_names[1]] = other_indices

            # concat dataset to tensor and get unique values
            all_possible_z = torch.tensor(all_possible_z)
            all_possible_z = torch.unique(all_possible_z)
            all_possible_z = torch.sort(all_possible_z).values
            print("all possible z_labels in the data: ", all_possible_z)

            print("ratio of z_labels in the data: ", len(filtered_indices)/len(other_indices))

            return partitions