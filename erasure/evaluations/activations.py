import torch

from erasure.core.factory_base import get_function
from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.utils.cfg_utils import init_dflts_to_of


class ActivationDistance(Measure):

    def init(self):
        super().init()

        self.distance_name = self.params['name']
        self.distance_params = self.params['function']['parameters']
        self.distance_func = get_function(self.params['function']['class'])

        self.forget_part = self.params["forget_part"]

    def check_configuration(self):
        super().check_configuration()
        init_dflts_to_of(self.local.config, 'function', 'erasure.evaluations.activations.l2norm')  # Default distance is L2 norm
        self.params['name'] = self.params.get('name', self.params['function']['class'])  # Default name as distance name
        self.params['forget_part'] = self.params.get('forget_part', "forget")


    def process(self, e:Evaluation):

        original_model = e.predictor
        unlearned_model = e.unlearned_model

        forget_ids = original_model.dataset.partitions[self.forget_part]

        sample_distances = []  # activation distance for each sample

        for fid in forget_ids:
            dataloader = original_model.dataset.get_loader_for_ids([fid])
            X, labels  = next(iter(dataloader))
            X = X.to(original_model.device)

            # compute activations for the sample
            original_activations = get_activations(original_model.model, X)
            unlearned_activations = get_activations(unlearned_model.model, X)

            # compute distance between activations
            distance = self.distance_func(original_activations, unlearned_activations)
            sample_distances.append(distance)

        distance = torch.mean(torch.tensor(sample_distances)).item()

        self.info(f"{self.distance_name}: {distance}")
        e.add_value(self.distance_name, distance)

        return e


def get_activations(model, x):

    activations = []

    # define hook function
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[1]
        activations.append(output.detach())

    hooks = []

    # Register hooks dynamically for all layers
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(hook))

    # Forward pass to collect activations
    with torch.no_grad():
        model(x)

    # Remove hooks after use
    for h in hooks:
        h.remove()

    return activations


def l2norm(list1, list2):
    distances = []

    for elem1, elem2 in zip(list1, list2):
        distances.append(
            torch.norm(elem1 - elem2)
        )

    return torch.mean(torch.tensor(distances)).item()

