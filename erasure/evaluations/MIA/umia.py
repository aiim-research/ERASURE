import copy
import os

import sklearn
import sklearn.linear_model
import sklearn.metrics
import torch

from erasure.core.factory_base import get_instance_config
from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.evaluations.utils import compute_accuracy
from erasure.utils.config.local_ctx import Local


class Attack(Measure):
    """ Unlearning (Population) Membership Inference Attack (MIA)
        taken from https://doi.org/10.48550/arXiv.2403.01218
        (Kurmanji version)
    """

    def init(self):
        self.attack_in_data_cfg = self.params["attack_in_data"]
        self.attack_model_cfg = self.params["attack_model"]

        self.local_config["parameters"]["attack_in_data"]["parameters"]['DataSource']["parameters"]['path'] += '_'+str(self.global_ctx.config.globals['seed'])
        self.data_out_path = self.local_config["parameters"]["attack_in_data"]["parameters"]['DataSource']["parameters"]['path']

        self.forget_part = 'forget'
        self.test_part = 'test'

        self.params["loss_fn"]["parameters"]["reduction"] = "none"
        self.loss_fn = get_instance_config(self.params['loss_fn'])

    def check_configuration(self):
        super().check_configuration()

        if "attack_model" not in self.params:
            self.params["attack_model"] = None

        if "loss_fn" not in self.params:
            self.params["loss_fn"] = copy.deepcopy(self.global_ctx.config.predictor["parameters"]["loss_fn"])


    def process(self, e: Evaluation):
        # Target Model (unlearned model)
        target_model = e.unlearned_model

        # generate dataset from sampling the target model
        self.info("Creating attack dataset")
        attack_dataset = self.create_attack_dataset(target_model)

        # build a binary classifier
        self.info("Creating attack model")

        if self.attack_model_cfg:
            current = Local(self.attack_model_cfg)
            current.dataset = attack_dataset
            attack_model = self.global_ctx.factory.get_object(current)  # ToDo: attenzione alla cache!

            # Compute accuracy
            test_loader, _ = attack_dataset.get_loader_for("test")
            umia_accuracy = compute_accuracy(test_loader, attack_model.model)

        else:
            # sklearn Logistic Regression
            attack_loader, _ = attack_dataset.get_loader_for("all")
            X, y = attack_loader.dataset[:]
            attack_model = sklearn.linear_model.LogisticRegression()
            cv = sklearn.model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.8)
            accuracies = sklearn.model_selection.cross_val_score(
                attack_model, X, y, cv=cv, scoring="accuracy")

            umia_accuracy = accuracies.mean()

        self.info(f"UMIA accuracy: {umia_accuracy}")
        e.add_value("UMIA", umia_accuracy)

        return e

    def create_attack_dataset(self, target_model):
        """ Create the attack dataset from the target model"""
        attack_samples = []
        attack_labels = []

        # Attack Dataset creation
        samples, labels = self.get_attack_samples(target_model)
        attack_samples.append(samples)
        attack_labels.append(labels)

        # concat all batches in single array -- all samples are in the first dimension
        attack_samples = torch.cat(attack_samples)
        attack_labels = torch.cat(attack_labels)

        # shuffle samples
        perm_idxs = torch.randperm(len(attack_samples))
        attack_samples = attack_samples[perm_idxs]
        attack_labels = attack_labels[perm_idxs]

        # create Datasets
        n_classes = 2
        attack_dataset = torch.utils.data.TensorDataset(attack_samples, attack_labels)
        attack_dataset.n_classes = n_classes

        # create DataManagers for the Attack model
        os.makedirs(os.path.dirname(self.data_out_path), exist_ok=True)
        torch.save(attack_dataset, self.data_out_path)
        attack_datamanager = self.global_ctx.factory.get_object(Local(self.attack_in_data_cfg))

        return attack_datamanager

    def get_attack_samples(self, model):
        """ From the unlearned model, generate the attack samples """

        forget_loader, _ = model.dataset.get_loader_for(self.forget_part)
        test_loader, _ = model.dataset.get_loader_for(self.test_part)

        forget_samples, forget_labels = self.generate_samples(model, forget_loader, 1)
        test_samples, test_labels = self.generate_samples(model, test_loader, 0)

        # we need the same number of samples from each partition
        samples_size = min(len(forget_samples), len(test_samples))

        forget_samples = forget_samples[:samples_size]
        forget_labels = forget_labels[:samples_size]
        test_samples = test_samples[:samples_size]
        test_labels = test_labels[:samples_size]

        return torch.cat([forget_samples, test_samples]), torch.cat([forget_labels, test_labels])

    def generate_samples(self, model, loader, label_value):
        attack_samples = []
        attack_labels = []

        with torch.no_grad():
            for X, labels in loader:
                X = X.to(model.device)
                _, predictions = model.model(X)  # model prediction

                losses = self.loss_fn(predictions, labels)
                # predictions = torch.nn.functional.softmax(predictions, dim=0)

                losses = losses.to('cpu')
                predictions = predictions.to('cpu')

                attack_samples.append(
                    losses.unsqueeze(1)
                    # predictions
                )
                attack_labels.append(
                    torch.full([len(X)], label_value, dtype=torch.float32)   # 1: forgetting samples, 0: testing samples
                    # torch.full([len(X)], label_value, dtype=torch.long)
                )

        return torch.cat(attack_samples), torch.cat(attack_labels)