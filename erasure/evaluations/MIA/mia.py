import copy
from copy import deepcopy
import os

from numpy import argmax
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from erasure.evaluations.MIA.MembershipInference import MembershipInference
from erasure.evaluations.evaluation import Evaluation
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local


class Attack(MembershipInference):
    """ Membership Inference Attack (MIA)
        as presented in https://doi.org/10.1109/SP.2017.41
    """

    def process(self, e: Evaluation):
        self.info("Membership Inference Attack")

        # Target Model (unlearned model)
        original = e.predictor
        unlearned = e.unlearned_model

        forget_ids = self.dataset.partitions[self.forget_part]
        forget_dataloader = self.dataset.get_loader_for_ids(forget_ids)

        original_forget = self.test_dataset(self.attack_models, original, forget_dataloader)
        '''original_forget[0] = 1e-14
        original_forget[1] = 1.0-original_forget[0]''' #train
        target_forget = self.test_dataset(self.attack_models, unlearned, forget_dataloader)
        #target_test = self.__test_dataset(self.attack_models, unlearned, "test")

        self.info(f"Original Forget: {original_forget / original_forget.sum()}")
        self.info(f"Target Forget: {target_forget / target_forget.sum()}")
        #self.info(f"Target Test: {target_test/target_test.sum()}")

        # Forgetting Rate (doi: 10.1109/TDSC.2022.3194884)
        fr = (target_forget[0] - original_forget[0]) / (original_forget[1] +0.01)
        fr = fr.item()

        self.info(f"Forgetting Rate: {fr}")
        e.add_value("Forgetting Rate", fr)

        return e

    def create_attack_datasets(self, shadow_models):
        """ Create n_classes attack datasets from the given shadow models """
        attack_samples = []
        attack_labels = []

        # Attack Dataset creation
        for k in range(self.n_shadows):
            samples, labels = self.get_attack_samples(shadow_models[k], k)
            attack_samples.append(samples)
            attack_labels.append(labels)

        # concat all batches in single array -- all samples are in the first dimension
        attack_samples = torch.cat(attack_samples)
        attack_labels = torch.cat(attack_labels)

        # shuffle samples
        #perm_idxs = torch.randperm(len(attack_samples))
        #attack_samples = attack_samples[perm_idxs]
        #attack_labels = attack_labels[perm_idxs]

        # create Datasets based on true original label
        attack_datasets = {}
        for c in range(self.dataset.n_classes):
            c_idxs = (attack_samples[:,0] == c).nonzero(as_tuple=True)[0]
            attack_datasets[c] = torch.utils.data.TensorDataset(attack_samples[c_idxs,1:], attack_labels[c_idxs])
            attack_datasets[c].n_classes = self.dataset.n_classes

        # create DataManagers for the Attack model
        attack_datamanagers = {}

        os.makedirs(os.path.dirname(self.data_out_path), exist_ok=True) # TODO Random temp path
        for c in range(self.dataset.n_classes):
            file_path = self.data_out_path+str(c)
            torch.save(attack_datasets[c], file_path)
            # Create DataMangers and reload data
            attack_data = deepcopy(self.attack_in_data_cfg)
            attack_data["parameters"]["DataSource"]["parameters"]["path"] = file_path
            attack_datamanagers[c] = self.global_ctx.factory.get_object(Local(attack_data))

        return attack_datamanagers

    def get_attack_samples(self, shadow_model, k):
        """ From the shadow model, generate the attack samples """

        #train_loader, _ = self.dataset.get_loader_for(self.train_part_plh +"_"+str(k))
        #test_loader, _ = self.dataset.get_loader_for(self.test_part_plh +"_"+str(k))

        split_point = min(len(self.dataset.partitions[self.train_part_plh +"_"+str(k)]),\
                          len(self.dataset.partitions[self.test_part_plh +"_"+str(k)]))
        #                  len(self.dataset.partitions[self.attack_test_part]))
        
        train_indices = self.dataset.partitions[self.train_part_plh +"_"+str(k)][:split_point]
        test_indices = self.dataset.partitions[self.test_part_plh +"_"+str(k)][:split_point]
        #test_indices = self.dataset.partitions[self.attack_test_part][:split_point]


        train_loader = self.dataset.get_loader_for_ids(train_indices)
        test_loader = self.dataset.get_loader_for_ids(test_indices)


        attack_samples = []
        attack_labels = []

        samples, labels = self.generate_samples(shadow_model, train_loader, 1)
        attack_samples.append(samples)
        attack_labels.append(labels)

        samples, labels = self.generate_samples(shadow_model, test_loader, 0)
        attack_samples.append(samples)
        attack_labels.append(labels)

        return torch.cat(attack_samples), torch.cat(attack_labels)

    def generate_samples(self, model, loader, label_value):
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

    def test_dataset(self, attack_models, target_model, dataloader):
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

        return torch.bincount(predicted_labels, minlength=2)

