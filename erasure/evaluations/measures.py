import json
from collections import defaultdict
import torch
import numpy as np

from erasure.core.factory_base import get_function
from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.evaluations.utils import compute_accuracy, compute_relearn_time
from erasure.utils.cfg_utils import init_dflts_to_of
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
import pandas as pd 
import os
import yaml

class TorchSKLearn(Measure):
    def __init__(self, global_ctx: Global, local_ctx):
        super().__init__(global_ctx, local_ctx)

        self.partition_name = self.local.config['parameters']['partition']
        self.target = self.local.config['parameters']['target']
        self.metric_name = self.local.config['parameters']['name']
        self.metric_params = self.local.config['parameters']['function']['parameters']
        self.metric_func = get_function(self.local.config['parameters']['function']['class'])

    def check_configuration(self):
        super().check_configuration()
        init_dflts_to_of(self.local.config, 'function', 'sklearn.metrics.accuracy_score') # Default empty node for: sklearn.metrics.accuracy_score
        self.local.config['parameters']['partition'] = self.local.config['parameters'].get('partition', 'test')  # Default partition: test
        self.local.config['parameters']['name'] = self.local.config['parameters'].get('name', self.local.config['parameters']['function']['class'])  # Default name as metric name
        self.local.config['parameters']['target'] = self.local.config['parameters'].get('target', 'unlearned')  # Default partition: test

    def process(self, e: Evaluation):
        erasure_model = e.predictor

        if self.target == 'unlearned':
            erasure_model = e.unlearned_model
        
        loader, _ = e.unlearner.dataset.get_loader_for(self.partition_name)

        var_labels, var_preds = [], []

        with torch.no_grad():
            for batch, (X, labels) in enumerate(loader):
                _, pred = erasure_model.model(X.to(erasure_model.model.device))

                var_labels += list(labels.squeeze().to('cpu').numpy())
                var_preds += list(pred.squeeze().to('cpu').numpy())

            # preprocessing predictions TODO: made a preprocessing class?
            var_preds = np.argmax(var_preds, axis=1)            
            
            value = self.metric_func(var_labels, var_preds,**self.metric_params)
            self.info(f"{self.metric_name} of \"{self.partition_name}\" on {self.target}: {value} of {erasure_model}")

            e.add_value(self.metric_name+'.'+self.partition_name+'.'+self.target,value)

        return e
    
class PartitionInfo(Measure):
    def __init__(self, global_ctx: Global, local_ctx):
        super().__init__(global_ctx, local_ctx)

        self.partition_name = self.local.config['parameters']['partition']

    def check_configuration(self):
        super().check_configuration()
        self.local.config['parameters']['partition'] = self.local.config['parameters'].get('partition', 'forget')  # Default partition: test
        

    def process(self, e:Evaluation):
        info={}
        info['name']=self.partition_name

        partition = e.unlearner.dataset.partitions[self.partition_name]
        part_len=len(partition)
        
        info['size']=part_len
        
        loader, _ = e.unlearner.dataset.get_loader_for(self.partition_name)

        distribution = defaultdict(int)

        for _,labels in loader:
            for l in labels:
                distribution[l.item()] += 1

        distribution = {key:(value/part_len) for key,value in distribution.items()}
        info['classes_dist'] = distribution
        e.add_value('part_info.'+self.partition_name, info)

        return e

class AUS(Measure):
    """ Adaptive Unlearning Score """

    def process(self, e: Evaluation):
        or_model = e.predictor
        ul_model = e.unlearned_model

        test_loader, _ = e.unlearner.dataset.get_loader_for('test')
        forget_loader, _ = e.unlearner.dataset.get_loader_for('forget')

        or_test_accuracy = compute_accuracy(test_loader, or_model.model)
        ul_test_accuracy = compute_accuracy(test_loader, ul_model.model)
        ul_forget_accuracy = compute_accuracy(forget_loader, ul_model.model)

        aus = (1 - (or_test_accuracy - ul_test_accuracy)) / (1 + abs(ul_test_accuracy - ul_forget_accuracy))

        self.info(f"Adaptive Unlearning Score: {aus}")
        e.add_value("AUS", aus)

        return e


class SaveValues(Measure):
    # TODO: add configuration nodes
    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx, local_ctx)
        self.path = self.params['path']
        self.output_format = self.local_config['parameters'].get('output_format', self.path.split(".")[-1])

        valid_extensions = {'json': '.json', 'csv': '.csv', 'yaml':'.yaml', 'xlsx':'.xlsx'}
        if self.output_format not in valid_extensions:
            self.global_ctx.logger.info(f"Unsupported output format: {self.output_format}, defaulting to JSON")
            self.output_format = 'json'
        if not self.path.endswith(valid_extensions[self.output_format]):
            self.global_ctx.logger.info(f"File extension in path '{self.path}' does not match the specified output format '{self.output_format}'. "
                f"Expected extension: '{valid_extensions[self.output_format]}'."
                f"Defaulting to json.")
            self.output_format = 'json'
            self.path = "".join(self.path.split(".")[:-1]) + ".json"


    def process(self, e:Evaluation):

        if self.output_format == 'json':
            self.process_json(e)
   
        elif self.output_format == 'csv':
            self.process_csv(e)

        elif self.output_format == 'yaml':
            self.process_yaml(e)

        elif self.output_format == 'xlsx':
            self.process_excel(e)

        return e
    
    def process_json(self, e):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'a') as json_file:
            json.dump(e.data_info, json_file, indent=4)
            json_file.write(',')

    def process_csv(self, e):
        df = pd.DataFrame.from_dict([self.flatten_dict(e.data_info)])
        if not pd.io.common.file_exists(self.path):
            df.to_csv(self.path, mode='w', index=False)  
        else:
            df.to_csv(self.path, mode='a', index=False, header=False) 

    def process_excel(self, e):
        df = pd.DataFrame.from_dict([self.flatten_dict(e.data_info)])
        if not os.path.exists(self.path):  
            df.to_excel(self.path, index=False, engine='openpyxl')  
        else:
            with pd.ExcelWriter(self.path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                sheet_name = "Sheet1"
                startrow = writer.sheets[sheet_name].max_row
                df.to_excel(writer, index=False, header=False, startrow=startrow)

    def process_yaml(self, e):
        flat_data = self.flatten_dict(e.data_info)
        with open(self.path, 'a') as yaml_file:  
            yaml.dump(flat_data, yaml_file, default_flow_style=False)

    
    def flatten_dict(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class RelearnTime(Measure):

    def process(self, e: Evaluation):

        relearn_time = compute_relearn_time(e.unlearned_model, "forget", max_accuracy=self.params['req_acc'])

        self.info(f'Relearning Time: {relearn_time} epochs')
        e.add_value('Relearning Time (epochs):', relearn_time)

        return e


class AIN(Measure):
    """ Anamnesis Index (AIN) """

    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx, local_ctx)

        pass

    def process(self, e: Evaluation):

        # Gold Model creation
        current = Local(self.params["gold_model"])
        current.dataset = e.predictor.dataset
        current.predictor = e.predictor
        gold_model_unlearner = self.global_ctx.factory.get_object(current)
        self.gold_model = gold_model_unlearner.unlearn()

        # orginal accuracy on forget
        forget_loader, _ = e.unlearner.dataset.get_loader_for('forget')
        original_forget_accuracy = compute_accuracy(forget_loader, e.predictor.model)

        max_accuracy = (1-self.params["alpha"]) * original_forget_accuracy

        # relearn time of Unleaned model on forget
        rt_unlearned = compute_relearn_time(e.unlearned_model, "forget", max_accuracy=max_accuracy)

        # relearn time of Gold model on forget
        rt_gold = compute_relearn_time(e.unlearned_model, "forget", max_accuracy=max_accuracy)

        ain = rt_unlearned / rt_gold
        self.info(f'AIN: {ain}')
        e.add_value('AIN:', ain)

        return e



class MisuraGold(Measure):
    def process(self, e:Evaluation):
        return e