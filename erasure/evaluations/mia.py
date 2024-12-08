from copy import deepcopy
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.utils.config.local_ctx import Local


class MembershipInference(Measure):
    """ Membership Inference Attack (MIA)
        as presented in https://doi.org/10.1109/SP.2017.41
    """

    def process(self, e: Evaluation):
        self.info("Membership Inference Attack")
        # Target Model (unlearned model)
        target_model = e.unlearned_model
        original_model = e.predictor

        # original dataset
        original_dataset = target_model.dataset

        # Shadow Models
        shadow_models = []
        for k in range(self.params["shadow"]["n_models"]):
            self.info(f"Creating shadow model {k}")
            shadow_models.append(self.__create_shadow_model(original_dataset, k))

        # Attack Datasets and DataManagers
        attack_datasets = self.__create_attack_datasets(shadow_models)

        # Attack Models
        attack_models = {}
        for c, dataset in attack_datasets.items():
            self.info(f"Creating attack model {c}")
            current = Local(self.local_config['parameters']['predictor'])
            current.dataset = dataset
            attack_models[c] = self.global_ctx.factory.get_object(current)

        # Test data
        train_loader, _ = target_model.dataset.get_loader_for("train")

        original_foget = self.__test_dataset(attack_models, original_model, "forget")
        target_forget = self.__test_dataset(attack_models, target_model, "forget")
        target_test = self.__test_dataset(attack_models, target_model, "test")
        self.info(f"Original Forget: {original_foget/original_foget.sum()}")
        self.info(f"Target Forget: {target_forget/target_forget.sum()}")
        self.info(f"Target Test: {target_test/target_test.sum()}")

        # Forgetting Rate (doi: 10.1109/TDSC.2022.3194884)
        fr = (target_forget[0] - original_foget[0]) / original_foget[1]
        self.info(f"Forgetting Rate: {fr}")
        e.add_value("Forgetting Rate", fr)

        return e

    def __create_shadow_model(self, original_dataset, k):
        """ create generic Shadow Model """

        # create DataSet from random samples
        ref_data = original_dataset.partitions[self.params["shadow"]["ref_data"]]
        sample_idxs = torch.randperm(len(ref_data))[:self.params["shadow"]["n_samples"]]   # n random samples indices
        shadow_dataset = torch.utils.data.Subset(original_dataset.partitions["all"].data, sample_idxs)
        shadow_dataset.n_classes = original_dataset.partitions["all"].get_n_classes()

        # Create DatasetManager for shadow dataset
        data_path = self.params["shadow"]["data"]["parameters"]["DataSource"]["parameters"]["path"]
        os.makedirs(data_path, exist_ok=True)
        shadow_data = deepcopy(self.params["shadow"]["data"])
        shadow_data["parameters"]["DataSource"]["parameters"]["path"] = data_path+str(k)
        torch.save(shadow_dataset, data_path+str(k))
        shadow_datamanager = self.global_ctx.factory.get_object(Local(shadow_data))

        # create Model
        current = Local(self.params["shadow"]["predictor"])
        current.dataset = shadow_datamanager
        shadow_model = self.global_ctx.factory.get_object(current)

        return shadow_model

    def __create_attack_datasets(self, shadow_models):
        """ Create n_classes attack datasets from the given shadow models """
        attack_samples = []
        attack_labels = []

        # Attack Dataset creation
        for shadow_model in shadow_models:
            samples, labels = self.__get_attack_samples(shadow_model)
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
        n_classes = shadow_models[0].dataset.n_classes
        for c in range(n_classes):
            c_idxs = (attack_samples[:,0] == c).nonzero(as_tuple=True)[0]
            attack_datasets[c] = torch.utils.data.TensorDataset(attack_samples[c_idxs,1:], attack_labels[c_idxs])
            attack_datasets[c].n_classes = n_classes

        # create DataManagers for the Attack model
        attack_datamanagers = {}
        data_path = self.params['data']['parameters']['DataSource']['parameters']['path']
        os.makedirs(data_path, exist_ok=True)
        for c in range(n_classes):
            torch.save(attack_datasets[c], data_path+str(c))
            attack_data = deepcopy(self.local_config["parameters"]["data"])
            attack_data["parameters"]["DataSource"]["parameters"]["path"] = data_path + str(c)
            attack_datamanagers[c] = self.global_ctx.factory.get_object(Local(attack_data))

        return attack_datamanagers

    def __get_attack_samples(self, shadow_model):
        """ From the shadow model, generate the attack samples """

        train_loader, _ = shadow_model.dataset.get_loader_for('train')
        test_loader, _ = shadow_model.dataset.get_loader_for('test')

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
                _, predictions = model.model(X)  # shadow model prediction
                predictions = predictions.to('cpu')

                attack_samples.append(
                    torch.cat([original_labels, predictions], dim=1)
                )
                attack_labels.append(
                    torch.full([len(X)], label_value, dtype=torch.long)   # 1: training samples, 0: testing samples
                )

        return torch.cat(attack_samples), torch.cat(attack_labels)

    def __test_dataset(self, attack_models, target_model, split_name):
        """ tests samples from the original dataset """

        loader, _ = target_model.dataset.get_loader_for(split_name)
        attack_predictions = []
        with torch.no_grad():
            for X, labels in loader:
                _, target_predictions = target_model.model(X.to(target_model.device))
                for i in range(len(target_predictions)):
                    _, prediction = attack_models[labels[i].item()].model(target_predictions[i])
                    softmax = nn.Softmax(dim=0)
                    prediction = softmax(prediction)
                    attack_predictions.append(prediction)

        attack_predictions = torch.stack(attack_predictions)    # convert into a Tensor
        predicted_labels = torch.argmax(attack_predictions, dim=1)    # get the predicted label

        return torch.bincount(predicted_labels)

