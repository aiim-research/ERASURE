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
        attack_dataset = self.__create_attack_dataset(target_model)

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
            # Hardcoded Linear Regression
            train_loader, _ = attack_dataset.get_loader_for("train")
            X_train, y_train = train_loader.dataset[:]

            attack_model = sklearn.linear_model.LogisticRegression()
            attack_model.fit(X_train, y_train)

            # Compute accuracy
            test_loader, _ = attack_dataset.get_loader_for("test")
            X_test, y_test = test_loader.dataset[:]
            umia_accuracy = sklearn.metrics.accuracy_score(y_test, attack_model.predict(X_test))

        self.info(f"UMIA accuracy: {umia_accuracy}")
        e.add_value("UMIA", umia_accuracy)

        return e



    def __create_attack_dataset(self, target_model):
        """ Create the attack dataset from the target model"""
        attack_samples = []
        attack_labels = []

        # Attack Dataset creation
        samples, labels = self.__get_attack_samples(target_model)
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
        n_classes = 1
        attack_dataset = torch.utils.data.TensorDataset(attack_samples, attack_labels)
        attack_dataset.n_classes = n_classes

        # create DataManagers for the Attack model
        os.makedirs(os.path.dirname(self.data_out_path), exist_ok=True)
        torch.save(attack_dataset, self.data_out_path)
        attack_datamanager = self.global_ctx.factory.get_object(Local(self.attack_in_data_cfg))

        return attack_datamanager


    def __get_attack_samples(self, model):
        """ From the unlearned model, generate the attack samples """

        forget_loader, _ = model.dataset.get_loader_for(self.forget_part)
        test_loader, _ = model.dataset.get_loader_for(self.test_part)

        # we need the same number of samples from each partition
        samples_size = min(len(forget_loader.dataset), len(test_loader.dataset))

        attack_samples = []
        attack_labels = []

        forget_samples, forget_labels = self.__generate_samples(model, forget_loader, 1)
        attack_samples.append(forget_samples[:samples_size])
        attack_labels.append(forget_labels[:samples_size])

        test_samples, test_labels = self.__generate_samples(model, test_loader, 0)
        attack_samples.append(test_samples[:samples_size])
        attack_labels.append(test_labels[:samples_size])

        return torch.cat(attack_samples), torch.cat(attack_labels)

    def __generate_samples(self, model, loader, label_value):
        attack_samples = []
        attack_labels = []

        with torch.no_grad():
            for X, labels in loader:
                original_labels = labels.view(len(labels), -1)
                X = X.to(model.device)
                _, predictions = model.model(X)  # model prediction

                losses = torch.tensor([self.loss_fn(predictions[i], labels[i]) for i in range(len(predictions))])

                predictions = predictions.to('cpu')
                losses = losses.to('cpu')

                attack_samples.append(
                    losses.unsqueeze(1)
                )
                attack_labels.append(
                    torch.full([len(X)], label_value, dtype=torch.float32)   # 1: forgetting samples, 0: testing samples
                )

        return torch.cat(attack_samples), torch.cat(attack_labels)