import copy
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

    global_ctx.logger.info(f"Current PyTorch seed: {torch.initial_seed()}")

    #Create Dataset
    print("CONFIG DATA ", global_ctx.config.data)
    data_manager = global_ctx.factory.get_object( Local( global_ctx.config.data ))
    print("DATA MANAGER ", data_manager)
    #Create Predictor
    current = Local(global_ctx.config.predictor)
    current.dataset = data_manager
    predictor = global_ctx.factory.get_object(current)
    global_ctx.logger.info('Global Predictor: ' + str(predictor))

    #Create unlearners 
    unlearners = []
    unlearners_cfg = global_ctx.config.unlearners
    for un in unlearners_cfg:
        current = Local(un)
        current.dataset = data_manager
        current.predictor = copy.deepcopy(predictor)
        unlearners.append( global_ctx.factory.get_object(current) )

    
    #Evaluator
    current = Local(global_ctx.config.evaluator)
    current.unlearners = unlearners
    evaluator = global_ctx.factory.get_object(current)

    # Evaluations
    for unlearner in unlearners:
        global_ctx.logger.info('####\t\t Evaluating: '+unlearner.__class__.__name__ +'\t\t####')
        evaluator.evaluate(unlearner,predictor)

 


