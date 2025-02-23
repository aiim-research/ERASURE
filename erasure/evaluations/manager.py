import sys
import traceback

from erasure.core.base import Configurable
from erasure.core.factory_base import get_instance_kvargs
from erasure.evaluations.evaluation import Evaluation
from erasure.evaluations.running import UnlearnRunner
from erasure.utils.config.global_ctx import Global
from erasure.core.unlearner import Unlearner
from erasure.utils.config.local_ctx import Local


class Evaluator(Configurable):

    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.__init_measures__()

    def evaluate(self, unlearner: Unlearner, predictor):
        e = Evaluation(unlearner,predictor)
        for measure in self.measures:
            try:
                e = measure.process(e)
            except Exception as err:
                self.global_ctx.logger.warning(f"Error occurred during execution of evaluation {measure}")
                self.global_ctx.logger.warning(repr(err))
                if isinstance(measure, UnlearnRunner):
                    traceback.print_exc()

        return e

    def __init_measures__(self):
        self.measures = []
        for measure in self.params['measures']:
            current = Local(measure)
            self.measures.append( self.global_ctx.factory.get_object(current) )

        # the first metric has to be one that calls the unlearn() method of the unlearner
        if not isinstance(self.measures[0], UnlearnRunner):
            config = {"class": "erasure.evaluations.running.UnlearnRunner", "parameters":{}}
            current = Local(config)
            self.measures.insert(0, self.global_ctx.factory.get_object(current))

        assert isinstance(self.measures[0], UnlearnRunner)

