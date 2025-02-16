import torch
import scipy

from erasure.core.factory_base import get_function
from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.utils.cfg_utils import init_dflts_to_of
from erasure.utils.config.global_ctx import strtobool

##########
# Models
##########

class ModelDistance(Measure):
    """ Compute the distance between the original and the unlearned model.
    The distance function is given as a parameter.
    """

    def init(self):
        # super().init()

        self.distance_name = self.params['name']
        self.distance_params = self.params['function']['parameters']
        self.distance_func = get_function(self.params['function']['class'])

        self.block_diag = strtobool(self.params['block_diag'])

        self.activations = strtobool(self.params['activations'])
        self.ref_data = self.params["ref_data"]


    def check_configuration(self):
        super().check_configuration()
        init_dflts_to_of(self.local.config, 'function', 'erasure.evaluations.distances.l2norm') # Default distance is L2 norm
        self.params['name'] = self.params.get('name', self.params['function']['class'])  # Default name as distance name

        self.params['block_diag'] = self.params.get('block_diag', "false")

        self.params['activations'] = self.params.get('activations', "false")
        self.params['ref_data'] = self.params.get('ref_data', "forget")


    def process(self, e:Evaluation):

        if self.activations:
            distance = self.activation_distance(e)
        else:
            distance = self.weight_distance(e)

        self.info(f"{self.distance_name}: {distance:.20f}")
        e.add_value(self.distance_name, distance)

        return e


    def weight_distance(self, e):

        original = e.predictor
        unlearned = e.unlearned_model

        original_params = list(original.model.parameters())
        unlearned_params = list(unlearned.model.parameters())

        if self.block_diag:
            original_params = [create_block_diagonal(original_params)]
            unlearned_params = [create_block_diagonal(unlearned_params)]

        # compute the distance of all layers
        return self.distance_func(unlearned_params, original_params, **self.distance_params)


    def activation_distance(self, e):

        original_model = e.predictor
        unlearned_model = e.unlearned_model

        data_ids = original_model.dataset.partitions[self.ref_data]

        sample_distances = []  # activation distance for each sample

        for id in data_ids:
            dataloader = original_model.dataset.get_loader_for_ids([id])
            X, labels  = next(iter(dataloader))
            X = X.to(original_model.device)

            # compute activations for the sample
            original_activations = get_activations(original_model.model, X)
            unlearned_activations = get_activations(unlearned_model.model, X)

            if self.block_diag:
                original_activations = [create_block_diagonal(original_activations)]
                unlearned_activations = [create_block_diagonal(unlearned_activations)]

            # compute distance between activations
            distance = self.distance_func(original_activations, unlearned_activations)
            sample_distances.append(distance)

        return torch.mean(torch.tensor(sample_distances)).item()


##########
# Distance functions
##########

def l2norm(list1, list2):

    distances = []

    # compute the L2 norm distance for each pair
    for mat1, mat2 in zip(list1, list2):
        distances.append(
            torch.norm(mat1 - mat2)
        )

    # return the mean of all norms
    return torch.mean(torch.tensor(distances)).item()


def hausdorff(list1, list2):

    distances = []

    # compute the hausdorff distance for each pair
    for mat1, mat2 in zip(list1, list2):
        mat1 = mat1.detach().reshape(len(mat1), -1).cpu()
        mat2 = mat2.detach().reshape(len(mat2), -1).cpu()

        distances.append(
            scipy.spatial.distance.directed_hausdorff(mat1, mat2)[0]
        )

    # return the mean of all distances
    return torch.mean(torch.tensor(distances)).item()

def kldivergence(list1, list2):
    distances = []

    # compute KL-divergence for each layer
    for mat1, mat2 in zip(list1, list2):
        mat1 = mat1.detach().flatten().cpu()
        mat1 = torch.nn.functional.softmax(mat1, dim=0)
        mat1 = torch.clamp(mat1, 1e-6)
        mat2 = mat2.detach().flatten().cpu()
        mat2 = torch.nn.functional.softmax(mat2, dim=0)
        mat2 = torch.clamp(mat2, 1e-6)

        distances.append(
            scipy.special.kl_div(mat1, mat2).sum()
        )

    # aggregate the results with mean
    return torch.mean(torch.tensor(distances)).item()

def jsdistance(list1, list2):
    distances = []

    # compute JS-divergence for each layer
    for mat1, mat2 in zip(list1, list2):
        mat1 = mat1.detach().flatten().cpu()
        mat1 = torch.nn.functional.softmax(mat1, dim=0)
        mat1 = torch.clamp(mat1, 1e-6)
        mat2 = mat2.detach().flatten().cpu()
        mat2 = torch.nn.functional.softmax(mat2, dim=0)
        mat2 = torch.clamp(mat2, 1e-6)

        distances.append(
            scipy.spatial.distance.jensenshannon(mat1, mat2)
        )

    # aggregate the results with mean
    return torch.mean(torch.tensor(distances)).item()


##########
# Utils
##########

def create_block_diagonal(list):
    new_list = []
    for elem in list:
        new_list.append(
                elem.detach().reshape(len(elem), -1)
        )

    return torch.block_diag(*new_list)


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
