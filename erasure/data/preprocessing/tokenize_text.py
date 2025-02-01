from erasure.data.preprocessing.preprocess import Preprocess
import torch
from erasure.core.factory_base import get_instance_kvargs, get_instance_config
from transformers import AutoTokenizer


class TokenizerWrapper:
    def __init__(self, tokenizer, **kwargs):  
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        self.kwargs = kwargs

    def __call__(self, text):
        return self.tokenizer(text, **self.kwargs)

class TokenizeX(Preprocess):
    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx, local_ctx)
        self.tokenizer = get_instance_config(self.local_config["parameters"]["tokenizer"])

    def process(self, X, y, z):    
        if isinstance(X, str): 
            tokenized = self.tokenizer(X)
            X = tokenized["input_ids"].squeeze(0)  
        else:
            raise ValueError("Expected text input in X, but got non-string data.")
    
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        X_tensor = torch.stack((input_ids, attention_mask), dim=0) 

        X_tensor = X_tensor.squeeze(1)

        return X_tensor, int(y), z

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['tokenizer'] = self.local_config['parameters'].get('tokenizer', 'bert-base-uncased')
        self.local_config['parameters']['max_length'] = self.local_config['parameters'].get('max_length', 128)
