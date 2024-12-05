from abc import ABC, abstractmethod
import random
from torch.utils.data import Subset
from erasure.core.base import Configurable
from .Dataset import Dataset


class DataSplitter(ABC):
    def __init__(self, ref_data,parts_names):
        self.ref_data = ref_data
        self.parts_names = parts_names
    
    @abstractmethod
    def split_data(self, data):
        pass

    
class DataSplitterPercentage(DataSplitter):
    def __init__(self, percentage, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.percentage = percentage

    def split_data(self,partitions):
        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else Dataset(Subset(partitions['all'].data, partitions[self.ref_data]))
            
        total_size = len(ref_data.data)
        split_point = int(total_size * self.percentage)

        indices = list(range(total_size))

        split_indices_1 = indices[:split_point]
        split_indices_2 = indices[split_point:]

        partitions[self.parts_names[0]] = split_indices_1
        partitions[self.parts_names[1]] = split_indices_2

        return partitions

class DataSplitterClass(DataSplitter):
    def __init__(self, label, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.label = label


    def split_data(self,partitions):
        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else Dataset(Subset(partitions['all'].data, partitions[self.ref_data]))

        filtered_indices = [idx for idx, (_, label) in enumerate(ref_data.data) if label == self.label]

        other_indices = [idx for idx, (_,label) in enumerate(ref_data.data) if idx not in filtered_indices]

        partitions[self.parts_names[0]] = filtered_indices 
        partitions[self.parts_names[1]] = other_indices

        return partitions


    '''
    def split_data(self,partitions):
        
        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else Dataset(Subset(partitions['all'], partitions[self.ref_data]))
        print(ref_data)
        print(len(ref_data.data))

        label_indices = [ i, (x,label) in enumerate(ref_data.data) if label == self.label]
        
        label_indices = [i for i, item in enumerate(ref_data.data) if item is not None and isinstance(item, tuple) and len(item) == 2 and item[1] == self.label]
        print("LABEL_INDICES", len(label_indices))
        all_indices = set(ref_data.data.indices)
        split_indices_2 = list(all_indices - set(label_indices))


        partitions[self.parts_names[0]] = label_indices
        partitions[self.parts_names[1]] = split_indices_2
    
        return partitions
    ''' 

class DataSplitterNSamples(DataSplitter):
    def __init__(self, n_samples, parts_names, ref_data = 'all'):
        super().__init__(ref_data,parts_names) 
        self.n_samples = n_samples

    def split_data(self,partitions):
        
        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else Dataset(Subset(partitions['all'].data, partitions[self.ref_data]))

        
        total_size = len(ref_data.data)

        
        split_point = self.n_samples if self.n_samples is not None else 0
        

        indices = ref_data.data.indices

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
        
        ref_data = partitions[self.ref_data] if self.ref_data == 'all' else Dataset(Subset(partitions['all'].data, partitions[self.ref_data]))

                
        indices = ref_data.data.indices

        if self.samples_ids:
            split_indices_1 = self.samples_ids
            split_indices_2 = [id for id in indices if id not in self.samples_ids]


        partitions[self.parts_names[0]] = split_indices_1
        partitions[self.parts_names[1]] = split_indices_2

        return partitions
    