from copy import deepcopy
import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from erasure.evaluations.MIA.MembershipInference import MembershipInference
from erasure.evaluations.evaluation import Evaluation
from erasure.utils.config.local_ctx import Local


class Attack(MembershipInference):
    """ Likelihood Ratio membership inference Attack (LiRA)
        (https://doi.org/10.48550/arXiv.2403.01218)
        the Carlini way (https://doi.org/10.1109/SP46214.2022.9833649)
    """

    def process(self, e: Evaluation):
        self.info("Likelihood Ratio")

        # Target Model (unlearned model)
        original = e.predictor
        unlearned = e.unlearned_model

        forget_dataloader, _ = original.dataset.get_loader_for(self.forget_part)
        forget_ids = original.dataset.partitions[self.forget_part]

        original_forget = self.test_dataset(self.attack_models, original, forget_dataloader, forget_ids)
        target_forget = self.test_dataset(self.attack_models, unlearned, forget_dataloader, forget_ids)

        self.info(f"Original Forget: {original_forget}")
        self.info(f"Target Forget: {target_forget}")
        #self.info(f"Target Test: {target_test/target_test.sum()}")

        self.info(f"LiRA: {target_forget}")
        e.add_value("LiRA", target_forget)

        return e

    def create_attack_datasets(self, shadow_models):
        """ Create |forget_set| attack datasets from the given shadow models.
        Each dataset contains samples for the same index """
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
        perm_idxs = torch.randperm(len(attack_samples))
        attack_samples = attack_samples[perm_idxs]
        attack_labels = attack_labels[perm_idxs]

        # create Datasets based on sample index
        attack_datasets = {}
        for f_id in self.dataset.partitions[self.forget_part]:
            f_idxs = (attack_samples[:, 0] == f_id).nonzero(as_tuple=True)[0]
            if len(f_idxs) > 0:
                attack_datasets[f_id] = torch.utils.data.TensorDataset(attack_samples[f_idxs, 1:], attack_labels[f_idxs])
                attack_datasets[f_id].n_classes = 10

        # create DataManagers for the Attack model
        attack_datamanagers = {}

        os.makedirs(os.path.dirname(self.data_out_path), exist_ok=True)  # TODO Random temp path
        for f_id in attack_datasets:
            file_path = self.data_out_path + str(f_id)
            torch.save(attack_datasets[f_id], file_path)
            # Create DataMangers and reload data
            attack_data = deepcopy(self.attack_in_data_cfg)
            attack_data["parameters"]["DataSource"]["parameters"]["path"] = file_path
            attack_datamanagers[f_id] = self.global_ctx.factory.get_object(Local(attack_data))

        return attack_datamanagers

    def get_attack_samples(self, shadow_model, k):
        """ From the shadow model, generate the attack samples """

        forget_loader, _ = self.dataset.get_loader_for(self.forget_part)
        forget_ids = self.dataset.partitions[self.forget_part]
        shadow_train_ids = self.dataset.partitions[self.train_part_plh + "_" + str(k)]

        label_values = [int(f_id in shadow_train_ids) for f_id in forget_ids]

        attack_samples = []
        attack_labels = []

        samples, labels = self.generate_samples(shadow_model, forget_loader, label_values)
        attack_samples.append(samples)
        attack_labels.append(labels)

        return torch.cat(attack_samples), torch.cat(attack_labels)

    def generate_samples(self, model, loader, label_values):

        attack_samples = []
        attack_labels = []

        with torch.no_grad():
            for X, labels in loader:
                X = X.to(model.device)
                labels = labels.to(model.device)

                _, predictions = model.model(X)  # shadow model prediction

                losses = self.loss_fn(predictions, labels)
                # predictions = torch.nn.functional.softmax(predictions, dim=0)

                losses = losses.to('cpu')
                predictions = predictions.to('cpu')

                attack_samples.append(
                    # losses.unsqueeze(1)
                    predictions
                )

        attack_samples = torch.cat(attack_samples)

        forget_ids = torch.tensor(self.dataset.partitions[self.forget_part])[:len(attack_samples),None]
        attack_samples = torch.cat([forget_ids, attack_samples], dim=1)
        attack_labels = torch.tensor(label_values)[:len(attack_samples)]

        return attack_samples, attack_labels

    def test_dataset(self, attack_models, target_model, dataloader, data_ids):
        """ tests samples from the original dataset """

        attack_predictions = []
        with torch.no_grad():
            for batch, (X, labels) in enumerate(dataloader):
                X = X.to(target_model.device)
                labels = labels.to(target_model.device)
                _, target_predictions = target_model.model(X)
                for i in range(len(target_predictions)):
                    curr_id = batch*dataloader.batch_size + i
                    curr_f_id = data_ids[curr_id]
                    if curr_f_id in attack_models:
                        loss = self.loss_fn(target_predictions[i], labels[i])

                        # evaluation = attack_models[curr_f_id].evaluate(loss)

                        _, prediction = attack_models[curr_f_id].model(target_predictions[i])
                        evaluation = torch.nn.functional.softmax(prediction, dim=0)

                        evaluation = evaluation.to('cpu')
                        if evaluation is not None:
                            evaluation += 0.0001
                            attack_predictions.append(evaluation[1]/evaluation[0])

        return np.mean(attack_predictions)

