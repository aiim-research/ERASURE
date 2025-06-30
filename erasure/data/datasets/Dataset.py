

class DatasetWrapper:
    def __init__(self, data, preprocess = []):
        self.data = data 
        self.preprocess = preprocess

    def get_n_classes(self):
        n_classes = len(self.data.classes)

        return n_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        X,y = self.__realgetitem__(index)

        X,y,Z = self.apply_preprocessing(X, y, None)

        return X,y

    def __realgetitem__(self, index: int):
        sample = self.data[index]
        X,y = sample
        return X,y


    def apply_preprocessing(self, X, y,Z):
        """
        Apply each preprocessing step to the data (X, y).
        """
        for preprocess in self.preprocess:
            X,y,Z = preprocess.process(X,y,Z)
        return X, y, Z

class DatasetExtendedWrapper:
    def __init__(self, inst):
        self.inst = inst
        self.data = self.inst.data #TODO Migliorare il forwarding
        self.preprocess = self.inst.preprocess

    def get_n_classes(self):        
        return self.inst.get_n_classes()

    def __len__(self):
        return self.inst.__len__()

    def __getitem__(self, index: int):
        X,y = self.inst.__realgetitem__(index)
        X,y,Z = self.apply_preprocessing(X, y, None)
        return X,y,Z

    def apply_preprocessing(self, X,y,Z):
        return self.inst.apply_preprocessing(X,y,Z)


        