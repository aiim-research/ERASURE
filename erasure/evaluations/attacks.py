import torch
from torch.utils.data import DataLoader

from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.utils.config.local_ctx import Local


class MembershipInference(Measure):
    """ Membership Inference Attack (MIA) """

    def process(self, e: Evaluation):

        # Target Model (unlearned model)
        target_model = e.unlearned_model

        # original dataset
        original_dataset = target_model.dataset

        # Shadow Model creation
        shadow_models = []
        for k in range(self.params["shadow"]["n_models"]):
            shadow_models.append(self.__create_shadow_model(original_dataset))

        attack_samples = []
        attack_labels = []

        # Attack Dataset creation
        for shadow_model in shadow_models:
            samples, labels = self.__get_attack_samples(shadow_model)
            attack_samples.append(samples)
            attack_labels.append(labels)


        # concat all batches in single array -- all samples are in the first dimesion
        attack_samples = torch.cat(attack_samples)
        attack_labels = torch.cat(attack_labels)
        # shuffle samples
        perm_idxs = torch.randperm(len(attack_samples))
        attack_samples = attack_samples[perm_idxs]
        attack_labels = attack_labels[perm_idxs]

        # create the Dataset
        attack_dataset = torch.utils.data.TensorDataset(attack_samples, attack_labels)
        attack_dataset.n_classes = original_dataset.partitions['all'].get_n_classes()

        # build a datamanager for the attack dataset
        data_path = self.params['data']['parameters']['DataSource']['parameters']['path']
        torch.save(attack_dataset, data_path)
        attack_datamanager = self.global_ctx.factory.get_object(Local(self.local_config["parameters"]["data"]))

        current = Local(self.local_config['parameters']['predictor'])
        current.dataset = attack_datamanager
        attack_model = self.global_ctx.factory.get_object(current)

        return e

    def __create_shadow_model(self, original_dataset):
        """ create generic Shadow Model """

        # create DataSet from random samples
        ref_data = original_dataset.partitions[self.params["shadow"]["ref_data"]]
        sample_idxs = torch.randperm(len(ref_data))[:self.params["shadow"]["n_samples"]]   # n random samples indices
        shadow_dataset = torch.utils.data.Subset(original_dataset.partitions["all"].data, sample_idxs)
        shadow_dataset.n_classes = original_dataset.partitions["all"].get_n_classes()

        # Create DatasetManager for shadow dataset
        data_path = self.params["shadow"]["data"]["parameters"]["DataSource"]["parameters"]["path"]
        torch.save(shadow_dataset, data_path)
        shadow_datamanager = self.global_ctx.factory.get_object(Local(self.params["shadow"]["data"]))

        # create Model
        current = Local(self.params["shadow"]["predictor"])
        current.dataset = shadow_datamanager
        shadow_model = self.global_ctx.factory.get_object(current)

        return shadow_model

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
            for batch, (X, labels) in enumerate(loader):
                original_labels = labels.view(len(labels), -1)
                _, predictions = model.model(X)  # shadow model prediction

                attack_samples.append(
                    torch.cat([original_labels, predictions], dim=1)
                )
                attack_labels.append(
                    # torch.ones((len(X), 1), dtype=torch.float)  # 1: training samples
                    # torch.ones(len(X), dtype=torch.long)   # 1: training samples
                    torch.full([len(X)], label_value, dtype=torch.long)   # 1: training samples, 0: testing samples
                )

        return torch.cat(attack_samples), torch.cat(attack_labels)