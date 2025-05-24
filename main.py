import copy
import os
import argparse
from erasure.utils.logger import GLogger
import torch
import numpy as np
import random

import tracemalloc
#from erasure.utils.config.global_ctx import set_seed 

#set_seed(1)


from erasure.utils.config.local_ctx import Local
from erasure.utils.config.global_ctx import Global, bcolors 
from erasure.core.factory_base import ConfigurableFactory
from erasure.data.datasets.DatasetManager import DatasetManager

tracemalloc.start()
arg_parser = argparse.ArgumentParser(description="Erasure Framework.")

arg_parser.add_argument("config_file", type=str, help="This is the path of the configuration file.")

args = arg_parser.parse_args()

config_file = args.config_file

if __name__ == "__main__":
    global_ctx = Global(config_file)
    global_ctx.factory = ConfigurableFactory(global_ctx)

    #Create Dataset
    data_manager = global_ctx.factory.get_object( Local( global_ctx.config.data ))
    global_ctx.dataset = data_manager

    #Create Predictor
    current = Local(global_ctx.config.predictor)
    current.dataset = data_manager
    predictor = global_ctx.factory.get_object(current)
    global_ctx.predictor = predictor
    global_ctx.logger.info('Global Predictor: ' + str(predictor))

    #Create unlearners 
    unlearners = []
    unlearners_cfg = global_ctx.config.unlearners
    for un in unlearners_cfg:
        current = Local(un)
        current.dataset = data_manager
        #current.predictor = copy.deepcopy(predictor)

        new_current = Local(global_ctx.config.predictor)
        new_current.dataset = data_manager
        new_current.skip_training = True
        current.predictor = global_ctx.factory.get_object(new_current)
        
        current.predictor.model.load_state_dict(predictor.model.state_dict())

        unlearners.append( global_ctx.factory.get_object(current) )

    #Evaluator
    current = Local(global_ctx.config.evaluator)
    current.unlearners = unlearners
    evaluator = global_ctx.factory.get_object(current)

    # Evaluations
    for unlearner in unlearners:
        global_ctx.logger.info(f'''{bcolors.OKGREEN}####\t\t Evaluating Unlearner {unlearner.__class__.__name__} \t\t####{bcolors.ENDC}''')
        evaluator.evaluate(unlearner,predictor)

 


