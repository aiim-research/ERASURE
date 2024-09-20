import os
import argparse
import torch
import numpy as np
import random
from erasure.utils.config.global_ctx import set_seed 

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
    predictor = global_ctx.factory.get_object(current)


    #Create unlearners 
    unlearners = []
    unlearners_cfg = global_ctx.config.unlearners
    for identifier_cfg in unlearners_cfg:
        current = Local(identifier_cfg)
        current.dataset = data_manager
        current.model = predictor
        unlearners.append( global_ctx.factory.get_object(current) )
        global_ctx.factory.get_object(current).unlearn()

    
    


