from .datasource import DataSource
from erasure.data.datasets.Dataset import Dataset 
from ucimlrepo import fetch_ucirepo 
from torch.utils.data import TensorDataset
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class UCIRepositoryDataSource(DataSource):
    def __init__(self, id):
        self.id = id
        self.dataset = None  
    
    def get_name(self):
        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)
        return self.dataset.name

    def create_data(self):
        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)
        
        X = self.dataset.data.features
        y = self.dataset.data.targets

        data = pd.concat([X, y], axis=1)

        ###TODO: Move this code to a preprocess function to organize it in a nicer way.
        data.replace('?', pd.NA, inplace=True)

        data.dropna(inplace=True)

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        if pd.api.types.is_string_dtype(y):
            y = y.str.strip()  
            y = y.str.rstrip('.')

        if pd.api.types.is_integer_dtype(y):
            if y.min() > 0:
                y = y - y.min() 

        categorical_columns = X.select_dtypes(include=['object', 'category']).columns

        if len(categorical_columns) > 0:
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

        X = X.apply(pd.to_numeric)

        X_array = X.to_numpy(dtype='float32')

        if y.dtypes == 'object' or y.dtypes.name == 'category':
            label_encoder = LabelEncoder()
            y_array = label_encoder.fit_transform(y)
        else:
            y_array = y.to_numpy()

        y_array = y_array.astype('int64') 

        X_tensor = torch.tensor(X_array, dtype=torch.float32)
        y_tensor = torch.tensor(y_array, dtype=torch.long)

        tensor_dataset = TensorDataset(X_tensor, y_tensor)

        dataset = Dataset(tensor_dataset)

        return dataset
