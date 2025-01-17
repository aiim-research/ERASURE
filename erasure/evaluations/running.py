import time
import torch.profiler

from erasure.core.measure import Measure
from erasure.evaluations.manager import Evaluation


class UnlearnRunner(Measure):
    """ Generic measure class that calls the unlearn() method """

    def process(self, e: Evaluation):
        e.unlearned_model = e.unlearner.unlearn()

        return e


class RunTime(UnlearnRunner):
    """ Wallclock running time to execute the unlearn """
    def process(self, e: Evaluation):
        if not e.unlearned_model:
            start_time = time.time()

            e.unlearned_model = e.unlearner.unlearn()
            metric_value = time.time() - start_time

        e.add_value('RunTime', metric_value)

        return e

class TorchFlops(UnlearnRunner):
    """ FLOPS to execute the unlearn (iw works only with PyTorch models) """
    def process(self, e: Evaluation):
        if not e.unlearned_model:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True
            ) as prof:
                
                e.unlearned_model = e.unlearner.unlearn()

        metric_value = sum(event.flops for event in prof.key_averages())

        e.add_value('TorchFlops', metric_value)

        return e

