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

        # Attack Model
        pass


        return e