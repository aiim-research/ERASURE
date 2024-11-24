import time

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

