import torch
import scipy
import sklearn

from erasure.core.factory_base import get_function
from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.utils.cfg_utils import init_dflts_to_of
from erasure.utils.config.global_ctx import strtobool

class ModelDistance(Measure):
    """ Compute the distance between the original and the unlearned model.
    The distance is given as a parameter.
    """

    def init(self):
        # super().init()

        self.distance_name = self.params['name']
        self.distance_params = self.params['function']['parameters']
        self.distance_func = get_function(self.params['function']['class'])
        self.block_diag = strtobool(self.params['block_diag'])


    def check_configuration(self):
        super().check_configuration()
        init_dflts_to_of(self.local.config, 'function', 'erasure.evaluations.distances.weights.l2norm') # Default distance is L2 norm
        self.params['name'] = self.params.get('name', self.params['function']['class'])  # Default name as distance name
        self.params['block_diag'] = self.params.get('block_diag', "false")

    def process(self, e:Evaluation):
        unlearned = e.unlearned_model
        original = e.predictor

        unlearned_params = list(unlearned.model.parameters())
        original_params = list(original.model.parameters())

        if self.block_diag:
            unlearned_params = [create_block_diagonal(unlearned_params)]
            original_params = [create_block_diagonal(original_params)]

        # compute the distance of all layers
        distance = self.distance_func(unlearned_params, original_params, **self.distance_params)

        self.info(f"{self.distance_name}: {distance}")
        e.add_value(self.distance_name, distance)

        return e


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
        mat1 = mat1.detach().reshape(len(mat1), -1)
        mat2 = mat2.detach().reshape(len(mat2), -1)

        distances.append(
            scipy.spatial.distance.directed_hausdorff(mat1, mat2)[0]
        )

    # return the mean of all distances
    return torch.mean(torch.tensor(distances)).item()

def kldivergence(list1, list2):
    distances = []

    for mat1, mat2 in zip(list1, list2):
        mat1 = mat1.detach().flatten()
        mat1 = torch.nn.functional.softmax(mat1, dim=0)
        mat2 = mat2.detach().flatten()
        mat2 = torch.nn.functional.softmax(mat2, dim=0)

        distances.append(
            scipy.special.kl_div(mat1, mat2).mean()
            # sklearn.metrics.mutual_info_score(mat1, mat2)
        )

    return torch.mean(torch.tensor(distances)).item()

def create_block_diagonal(list):
    new_list = []
    for elem in list:
        new_list.append(
                elem.detach().reshape(len(elem), -1)
        )

    return torch.block_diag(*new_list)
