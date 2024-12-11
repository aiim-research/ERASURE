

class DatasetWrapper:
    def __init__(self, data, preprocess = []):
        self.data = data 
        self.preprocess = preprocess

    def get_n_classes(self):
        ##access data from Dataset
        ##access datasets[0] because Dataset contains a ConcatDataset and the first one is the Training set
        ##so we get the number of classes from the training set
        n_classes = len(self.data.classes)

        return n_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        X,y = sample
        sample = self.apply_preprocessing(X,y)
        return sample


    def apply_preprocessing(self, X, y):
        """
        Apply each preprocessing step to the data (X, y).
        """
        for preprocess in self.preprocess:
            X, y = preprocess.process(X, y)
        return X, y