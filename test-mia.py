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

    train_loader, _ = shadow_model.dataset.get_loader_for('train')
    test_loader, _ = shadow_model.dataset.get_loader_for('test')

    attack_samples = []
    attack_labels = []

    with torch.no_grad():
        # TRAINING set
        for batch, (X, labels) in enumerate(train_loader):
            original_labels = labels.view(len(X), -1)
            _, predictions = shadow_model.model(X)    # shadow model prediction

            attack_samples.append(
                torch.cat([labels.view(len(labels), -1), predictions], dim=1)
            )
            attack_labels.append(
                torch.ones((len(X), 1), dtype=torch.float)   # 1: training samples
                #torch.ones(len(X), dtype=torch.long)   # 1: training samples
            )
        # TESTING set
        for batch, (X, labels) in enumerate(test_loader):
            original_labels = labels.view(len(X), -1)
            _, predictions = shadow_model.model(X)  # shadow model prediction

            attack_samples.append(
                torch.cat([labels.view(len(labels), -1), predictions], dim=1)
            )
            attack_labels.append(
                torch.zeros((len(X), 1), dtype=torch.float)   # 0: testing samples
                #torch.zeros(len(X), dtype=torch.long)   # 0: testing samples
            )

    attack_dataset = torch.utils.data.TensorDataset(torch.cat(attack_samples), torch.cat(attack_labels))



    local_config = global_ctx.config.evaluator['parameters']['measures'][0]

    # build a datamanager for the attack dataset
    torch.save(attack_dataset, "tmp/mia.pt")
    attack_datamanager = global_ctx.factory.get_object( Local( local_config["parameters"]["data"] ))

    current = Local(local_config['parameters']['predictor'])
    current.dataset = attack_datamanager
    attack_model = global_ctx.factory.get_object(current)


    # test molto vari
    attack_loader = torch.utils.data.DataLoader(attack_dataset, batch_size=128)
    # X, label = next(iter(attack_loader))


    with torch.no_grad():
        for batch, (X, labels) in enumerate(attack_loader):
            original_labels = labels.view(len(X), -1)
            _, predictions = attack_model.model(X)    # attack model prediction

            a = torch.cat([labels.view(len(labels), -1), predictions], dim=1)
            print(a)

    



