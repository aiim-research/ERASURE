from copy import deepcopy
import copy
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from erasure.core.factory_base import get_function
from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.utils.cfg_utils import init_dflts_to_of
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local


class MembershipInference(Measure):
    """ Membership Inference Attack (MIA)
        as presented in https://doi.org/10.1109/SP.2017.41
    """
    def __init__(self, global_ctx: Global, local_ctx):
        super().__init__(global_ctx, local_ctx)

        self.n_shadows = self.local.config['parameters']['shadows']['n_shadows']
        self.data_out_path = self.local.config['parameters']['shadows']['data_out_path']
        self.train_part_plh = self.local.config['parameters']['shadows']['train_part_plh']
        self.test_part_plh = self.local.config['parameters']['shadows']['test_part_plh']
        self.base_model_cfg = self.params["shadows"]["base_model"]
        self.attack_in_data_cfg = self.local_config["parameters"]["attack_in_data"]

        self.forget_part = 'forget'
        #self.test_part = 'test'

        self.dataset = global_ctx.factory.get_object( Local( self.local.config['parameters']['shadows']['shadow_in_data'] ))
        self.dataset.add_partitions(self.local.config['parameters']['shadows']['dataset_preproc'])
         
         # Shadow Models
        shadow_models = []
        for k in range(self.n_shadows):
            self.info(f"Creating shadow model {k}")
            self.dataset.add_partitions(copy.deepcopy([self.local.config['parameters']['shadows']['per_shadows_partition']]), "_"+str(k))
            shadow_models.append(self.__create_shadow_model(k))

        # Attack DataManagers
        attack_datasets = self.__create_attack_datasets(shadow_models)

        # Attack Models
        self.attack_models = {}
        for c, dataset in attack_datasets.items():
            self.info(f"Creating attack model {c}")
            current = Local(self.local_config['parameters']['attack_model'])
            current.dataset = dataset
            self.attack_models[c] = self.global_ctx.factory.get_object(current)
        
    def check_configuration(self):
        super().check_configuration()
        #init_dflts_to_of(self.local.config, 'function', 'sklearn.metrics.accuracy_score') # Default empty node for: sklearn.metrics.accuracy_score
        #self.local.config['parameters']['partition'] = self.local.config['parameters'].get('partition', 'test')  # Default partition: test
        #self.local.config['parameters']['name'] = self.local.config['parameters'].get('name', self.local.config['parameters']['function']['class'])  # Default name as metric name
        #self.local.config['parameters']['target'] = self.local.config['parameters'].get('target', 'unlearned')  # Default partition: test

    def process(self, e: Evaluation):
        self.info("Membership Inference Attack")

        # Target Model (unlearned model)
        original = e.predictor
        unlearned = e.unlearned_model      

        forget_dataloader = original.dataset.get_loader_for(self.forget_part)

        original_forget = self.__test_dataset(self.attack_models, original, forget_dataloader )
        target_forget = self.__test_dataset(self.attack_models, unlearned, forget_dataloader )
        #target_test = self.__test_dataset(self.attack_models, unlearned, "test")

        self.info(f"Original Forget: {original_forget/original_forget.sum()}")
        self.info(f"Target Forget: {target_forget/target_forget.sum()}")
        #self.info(f"Target Test: {target_test/target_test.sum()}")

        # Forgetting Rate (doi: 10.1109/TDSC.2022.3194884)
        fr = (target_forget[0] - original_forget[0]) / original_forget[1]
        self.info(f"Forgetting Rate: {fr}")
        e.add_value("Forgetting Rate", fr)

        return e

    def __create_shadow_model(self, k):
        """ create generic Shadow Model """
        # create shadow model
        shadow_base_model = copy.deepcopy(self.base_model_cfg)
        shadow_base_model['parameters']['training_set'] = self.train_part_plh +"_"+str(k)
        current = Local(shadow_base_model)

        current.dataset = self.dataset
        shadow_model = self.global_ctx.factory.get_object(current)

        return shadow_model

    def __create_attack_datasets(self, shadow_models):
        """ Create n_classes attack datasets from the given shadow models """
        attack_samples = []
        attack_labels = []

        # Attack Dataset creation
        for k in range(self.n_shadows):
            samples, labels = self.__get_attack_samples(shadow_models[k],k)
            attack_samples.append(samples)
            attack_labels.append(labels)

        # concat all batches in single array -- all samples are in the first dimension
        attack_samples = torch.cat(attack_samples)
        attack_labels = torch.cat(attack_labels)

        # shuffle samples
        perm_idxs = torch.randperm(len(attack_samples))
        attack_samples = attack_samples[perm_idxs]
        attack_labels = attack_labels[perm_idxs]

        # create Datasets based on true original label
        attack_datasets = {}
        for c in range(self.dataset.n_classes):
            c_idxs = (attack_samples[:,0] == c).nonzero(as_tuple=True)[0]
            attack_datasets[c] = torch.utils.data.TensorDataset(attack_samples[c_idxs,1:], attack_labels[c_idxs])
            attack_datasets[c].n_classes = self.dataset.n_classes

        # create DataManagers for the Attack model
        attack_datamanagers = {}
        
        os.makedirs(self.data_out_path, exist_ok=True) # TODO Random temp path
        for c in range(self.dataset.n_classes):
            file_path = self.data_out_path+str(c)
            torch.save(attack_datasets[c], file_path)
            # Create DataMangers and reload data
            attack_data = deepcopy(self.attack_in_data_cfg)
            attack_data["parameters"]["DataSource"]["parameters"]["path"] = file_path
            attack_datamanagers[c] = self.global_ctx.factory.get_object(Local(attack_data))

        return attack_datamanagers

    def __get_attack_samples(self, shadow_model,k):
        """ From the shadow model, generate the attack samples """

        train_loader, _ = self.dataset.get_loader_for(self.train_part_plh +"_"+str(k))
        test_loader, _ = self.dataset.get_loader_for(self.test_part_plh +"_"+str(k))


        attack_samples = []
        attack_labels = []

        samples, labels = self.__generate_samples(shadow_model, train_loader, 1)
        attack_samples.append(samples)
        attack_labels.append(labels)

        samples, labels = self.__generate_samples(shadow_model, test_loader, 0)
        attack_samples.append(samples)
        attack_labels.append(labels)

        return torch.cat(attack_samples), torch.cat(attack_labels)

    def __generate_samples(self, model, loader, label_value):
        attack_samples = []
        attack_labels = []

        with torch.no_grad():
            for X, labels in loader:
                original_labels = labels.view(len(labels), -1)
                X = X.to(model.device)
                _, predictions = model.model(X) # shadow model prediction #TODO check model to decide if applying the Softmax or not torch.nn.functional.softmax(model.model(X))
                predictions = predictions.to('cpu')

                attack_samples.append(
                    torch.cat([original_labels, predictions], dim=1)
                )
                attack_labels.append(
                    torch.full([len(X)], label_value, dtype=torch.long)   # 1: training samples, 0: testing samples
                )

        return torch.cat(attack_samples), torch.cat(attack_labels)

    def __test_dataset(self, attack_models, target_model, dataloader):
        """ tests samples from the original dataset """

        #loader, _ = target_model.dataset.get_loader_for(split_name)
        attack_predictions = []
        with torch.no_grad():
            for X, labels in dataloader:
                _, target_predictions = target_model.model(X.to(target_model.device))
                for i in range(len(target_predictions)):
                    _, prediction = attack_models[labels[i].item()].model(target_predictions[i])
                    softmax = nn.Softmax(dim=0)
                    prediction = softmax(prediction)
                    attack_predictions.append(prediction)

        attack_predictions = torch.stack(attack_predictions)    # convert into a Tensor
        predicted_labels = torch.argmax(attack_predictions, dim=1)    # get the predicted label

        return torch.bincount(predicted_labels)

