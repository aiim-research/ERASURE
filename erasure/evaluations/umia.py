import os
from copy import deepcopy

import torch

from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.evaluations.utils import compute_accuracy
from erasure.utils.config.local_ctx import Local


class UnlearningMembershipInference(Measure):
    """ Unlearning (Population) Membership Inference Attack (MIA)
        taken from https://doi.org/10.48550/arXiv.2403.01218
        (Kurmanji version)
    """

    def process(self, e: Evaluation):
        # Target Model (unlearned model)
        target_model = e.unlearned_model

        # generate dataset from sampling the target model
        self.info("Creating attack dataset")
        attack_dataset = self.__create_attack_dataset(target_model)

        # build a binary classifier
        self.info("Creating attack model")
        current = Local(self.local_config['parameters']['predictor'])
        current.dataset = attack_dataset
        attack_model = self.global_ctx.factory.get_object(current)

        # measure accuracy on remaining halves
        test_loader, _ = attack_model.dataset.get_loader_for('test')
        umia_accuracy = compute_accuracy(test_loader, attack_model.model)

        self.info(f"UMIA accuracy: {umia_accuracy}")
        e.add_value("UMIA", umia_accuracy)

        return e



    def __create_attack_dataset(self, target_model):
        """ Create the attack dataset """
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

        # create Datasets based on true original label
        n_classes = target_model.dataset.n_classes
        attack_dataset = torch.utils.data.TensorDataset(attack_samples, attack_labels)
        attack_dataset.n_classes = n_classes

        # create DataManagers for the Attack model
        attack_data = self.params["data"]
        data_path = attack_data['parameters']['DataSource']['parameters']['path']
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save(attack_dataset, data_path)
        attack_datamanager = self.global_ctx.factory.get_object(Local(attack_data))

        return attack_datamanager


    def __get_attack_samples(self, model):
        """ From the unlearned model, generate the attack samples """

        forget_loader, _ = model.dataset.get_loader_for('forget')
        test_loader, _ = model.dataset.get_loader_for('test')

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
                _, predictions = model.model(X)  # shadow model prediction
                predictions = predictions.to('cpu')

                attack_samples.append(
                    # torch.cat([original_labels, predictions], dim=1)
                    predictions
                )
                attack_labels.append(
                    torch.full([len(X)], label_value, dtype=torch.long)   # 1: forgetting samples, 0: testing samples
                )

        return torch.cat(attack_samples), torch.cat(attack_labels)