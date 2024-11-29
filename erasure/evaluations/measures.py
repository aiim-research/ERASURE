import json
from collections import defaultdict

from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.evaluations.utils import compute_accuracy, compute_relearn_time
from erasure.utils.config.local_ctx import Local
import pandas as pd 
import os
import yaml


class Accuracy(Measure):
    def process(self, e: Evaluation): 
        
        model1 = e.predictor
        model2 = e.unlearned_model

        test_loader, _ = e.unlearner.dataset.get_loader_for('test')

        og_accuracy = compute_accuracy(test_loader, model1.model)
        new_accuracy = compute_accuracy(test_loader,model2.model)

        print("ORIGINAL ACCURACY WAS ", og_accuracy, model1)
        print("NEW ACCURACY IS ", new_accuracy, model2)

        e.add_value('Accuracies', {'Original_accuracy:':og_accuracy, 'New_accuracy:':new_accuracy})

        return e


class Accuracies(Measure):
    def process(self, e: Evaluation):
        original = e.unlearner.predictor
        unlearned = e.unlearned_model


        # Test set
        test_loader, _ = e.unlearner.dataset.get_loader_for('test')
        or_test_accuracy = compute_accuracy(test_loader, original.model)
        un_test_accuracy = compute_accuracy(test_loader, unlearned.model)
        self.info(f"original test accuracy: {or_test_accuracy}")
        self.info(f"original test accuracy: {un_test_accuracy}")

        # Forget set
        forget_loader, _ = e.unlearner.dataset.get_loader_for('forget set')
        or_forget_accuracy = compute_accuracy(forget_loader, original.model)
        un_forget_accuracy = compute_accuracy(forget_loader, unlearned.model)
        self.info(f"original forget accuracy: {or_forget_accuracy}")
        self.info(f"original forget accuracy: {un_forget_accuracy}")

        # Retain set
        retain_loader, _ = e.unlearner.dataset.get_loader_for('other_classes')
        or_retain_accuracy = compute_accuracy(retain_loader, original.model)
        un_retain_accuracy = compute_accuracy(retain_loader, unlearned.model)
        self.info(f"original retain accuracy: {or_retain_accuracy}")
        self.info(f"original retain accuracy: {un_retain_accuracy}")

        return e


class AUS(Measure):
    """ Adaptive Unlearning Score """

    def process(self, e: Evaluation):
        or_model = e.predictor
        ul_model = e.unlearned_model

        test_loader, _ = e.unlearner.dataset.get_loader_for('test')
        forget_loader, _ = e.unlearner.dataset.get_loader_for('forget set')

        or_test_accuracy = compute_accuracy(test_loader, or_model.model)
        ul_test_accuracy = compute_accuracy(test_loader, ul_model.model)
        ul_forget_accuracy = compute_accuracy(forget_loader, ul_model.model)

        aus = (1 - (or_test_accuracy - ul_test_accuracy)) / (1 + abs(ul_test_accuracy - ul_forget_accuracy))

        print("Adaptive Unlearning Score:", aus)
        e.add_value("AUS", aus)

        return e


class ForgetSetInfo(Accuracy):
    def process(self, e:Evaluation):
        e.add_value('Size of identified forget set', len(e.forget_set))
        
        forget_set_loader = e.unlearner.dataset.get_loader_for_ids(e.forget_set)

        distributions = defaultdict(int)

        for _,labels in forget_set_loader:
            for l in labels:
                distributions[l.item()] += 1

        distributions = {key:(value/len(e.forget_set)) for key,value in distributions.items()}

        e.add_value('Distribution of classes in the forget set', distributions)

        return e


class SaveValues(Accuracy):
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

        relearn_time = compute_relearn_time(e.unlearned_model, "forget set", max_accuracy=self.params['req_acc'])

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

        # orginal accuracy on Forget Set
        forget_loader, _ = e.unlearner.dataset.get_loader_for('forget set')
        original_forget_accuracy = compute_accuracy(forget_loader, e.predictor.model)

        max_accuracy = (1-self.params["alpha"]) * original_forget_accuracy

        # relearn time of Unleaned model on Forget set
        rt_unlearned = compute_relearn_time(e.unlearned_model, "forget set", max_accuracy=max_accuracy)

        # relearn time of Gold model on Forget set
        rt_gold = compute_relearn_time(e.unlearned_model, "forget set", max_accuracy=max_accuracy)

        ain = rt_unlearned / rt_gold
        self.info(f'AIN: {ain}')
        e.add_value('AIN:', ain)

        return e



class MisuraGold(Measure):
    def process(self, e:Evaluation):
        return e