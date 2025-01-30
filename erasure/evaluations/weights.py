import torch
import scipy

from erasure.core.factory_base import get_function
from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from erasure.utils.cfg_utils import init_dflts_to_of


class ModelDistance(Measure):
    """ Compute the distance between the original and the unlearned model.
    The distance is given as a parameter.
    """

    def init(self):
        super().init()

        self.distance_name = self.params['name']
        self.distance_params = self.params['function']['parameters']
        self.distance_func = get_function(self.params['function']['class'])

    def check_configuration(self):
        super().check_configuration()
        init_dflts_to_of(self.local.config, 'function', 'erasure.evaluations.weights.l2norm') # Default distance is L2 norm
        self.params['name'] = self.params.get('name', self.params['function']['class'])  # Default name as distance name

    def process(self, e:Evaluation):
        unlearned = e.unlearned_model
        original = e.predictor

        distance = self.distance_func(unlearned.model, original.model, **self.distance_params)

        self.info(f"{self.distance_name}: {distance}")
        e.add_value(self.distance_name, distance)

        return e


def l2norm(model1, model2):
    model1_params = list(model1.parameters())
    model2_params = list(model2.parameters())

    # ensure the models have the same structure
    if len(model1_params) != len(model2_params):
        raise ValueError("The models do not have the same number of parameters.")

    # compute the L2 norm layer wise
    distances = []
    for param1, param2 in zip(model1_params, model2_params):
        if param1.shape != param2.shape:
            raise ValueError("Mismatch in parameter shapes between models.")

        distances.append(
            torch.norm(param1 - param2)
        )

    # return the mean of all norms
    return torch.mean(torch.tensor(distances)).item()


def hausdorff(model1, model2):
    model1_params = list(model1.parameters())
    model2_params = list(model2.parameters())

    # ensure the models have the same structure
    if len(model1_params) != len(model2_params):
        raise ValueError("The models do not have the same number of parameters.")

    # compute the L2 norm layer wise
    distances = []
    for param1, param2 in zip(model1_params, model2_params):
        if param1.shape != param2.shape:
            raise ValueError("Mismatch in parameter shapes between models.")

        param1 = param1.detach().reshape(len(param1), -1)
        param2 = param2.detach().reshape(len(param2), -1)

        distances.append(
            scipy.spatial.distance.directed_hausdorff(param1, param2)[0]
        )

    # return the mean of all distances
    return torch.mean(torch.tensor(distances)).item()

