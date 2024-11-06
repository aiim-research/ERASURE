import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.utils.config.local_ctx import Local


class MembershipInference(Measure):
    def process(self, e: Evaluation):

        # Target Model (unlearned model)
        target_model = e.unlearned_model

        # original dataset
        original_dataset = target_model.dataset

        # generic Shadow Model i, same configuration as the original model
        current = Local(self.global_ctx.config.predictor)
        current.dataset = original_dataset
        shadow_model = self.global_ctx.factory.get_object(current)

        # Attack model Dataset building
        train_loader, _ = shadow_model.dataset.get_loader_for('train')
        test_loader, _ = shadow_model.dataset.get_loader_for('test')

        attack_samples = []
        attack_labels = []

        with torch.no_grad():
            # TRAINING set
            for batch, (X, labels) in enumerate(train_loader):
                original_labels = labels.view(len(labels), -1)
                _, predictions = shadow_model.model(X)  # shadow model prediction

                attack_samples.append(
                    torch.cat([original_labels, predictions], dim=1)
                )
                attack_labels.append(
                    torch.ones((len(X), 1), dtype=torch.float)  # 1: training samples
                    # torch.ones(len(X), dtype=torch.long)   # 1: training samples
                )
            # TESTING set
            for batch, (X, labels) in enumerate(test_loader):
                original_labels = labels.view(len(labels), -1)
                _, predictions = shadow_model.model(X)  # shadow model prediction

                attack_samples.append(
                    torch.cat([original_labels, predictions], dim=1)
                )
                attack_labels.append(
                    torch.zeros((len(X), 1), dtype=torch.float)  # 0: testing samples
                    # torch.zeros(len(X), dtype=torch.long)   # 0: testing samples
                )

        attack_dataset = TensorDataset(torch.cat(attack_samples), torch.cat(attack_labels))

        # build a datamanager for the attack dataset
        torch.save(attack_dataset, "tmp/mia.pt")
        attack_datamanager = self.global_ctx.factory.get_object(Local(self.local_config["parameters"]["data"]))

        current = Local(self.local_config['parameters']['predictor'])
        current.dataset = attack_datamanager
        attack_model = self.global_ctx.factory.get_object(current)

        return e