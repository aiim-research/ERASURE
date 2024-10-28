import os
import argparse
import torch
import numpy as np
import random
from erasure.utils.config.global_ctx import set_seed
from torch.utils.data import DataLoader, TensorDataset

set_seed(1)


from erasure.utils.config.local_ctx import Local
from erasure.utils.config.global_ctx import Global 
from erasure.core.factory_base import ConfigurableFactory
from erasure.data.datasets.DatasetManager import DatasetManager

arg_parser = argparse.ArgumentParser(description="Erasure Framework.")

arg_parser.add_argument("config_file", type=str, help="This is the path of the configuration file.")

args = arg_parser.parse_args()

config_file = args.config_file

if __name__ == "__main__":
    global_ctx = Global(config_file)
    global_ctx.factory = ConfigurableFactory(global_ctx)

    print(f"Current PyTorch seed: {torch.initial_seed()}")

    #Create Dataset
    data_manager = global_ctx.factory.get_object( Local( global_ctx.config.data ))

    #Create Predictor
    current = Local(global_ctx.config.predictor)
    current.dataset = data_manager
    # predictor = global_ctx.factory.get_object(current)




    # original dataset
    original_dataset = data_manager

    # generic Shadow Model, same configuration as the original model
    current = Local(global_ctx.config.predictor)
    current.dataset = original_dataset
    shadow_model = global_ctx.factory.get_object(current)   # create and train the shadow model

    train_loader, val_loader = shadow_model.dataset.get_loader_for('train')

    # attack_dataset = torch.utils.data.TensorDataset(torch.empty(1, 11), torch.empty(1))
    attack_samples = []
    attack_labels = []


    for batch, (X, labels) in enumerate(train_loader):
        result = shadow_model.model(X)    # shadow model prediction

        original_labels = labels.view(len(X), -1)
        predictions = result[1]

        attack_samples.append(
            torch.cat([labels.view(len(labels), -1), predictions], dim=1)
        )
        attack_labels.append(
            torch.zeros(len(X))
        )


    attack_dataset = torch.utils.data.TensorDataset(torch.cat(attack_samples), torch.cat(attack_labels)) 



    # test molto vari
    attack_loader = torch.utils.data.DataLoader(attack_dataset, batch_size=128)
    X, label = next(iter(attack_loader))











