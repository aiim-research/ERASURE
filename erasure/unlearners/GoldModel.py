from erasure.core.unlearner import Unlearner
from erasure.utils.config.global_ctx import Global

class GoldModel(Unlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        """
        Initializes the GoldModel class with global and local contexts.

        Args:
            global_ctx (Global): The global context containing configurations and shared resources.
            local_ctx (Local): The local context containing specific configurations for this instance.
        """

        super().__init__(global_ctx, local_ctx)
        self.ref_data = local_ctx.config['parameters'].get("ref_data", 'forget set')  # Default reference data is forget
    
    def __unlearn__(self):
        # retrain the model by removing the reference data (forget set by default)

        predictor = self.get_retrained(self.ref_data)
        return predictor