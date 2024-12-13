from erasure.core.unlearner import Unlearner
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local

class GoldModel(Unlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        """
        Initializes the GoldModel class with global and local contexts.

        Args:
            global_ctx (Global): The global context containing configurations and shared resources.
            local_ctx (Local): The local context containing specific configurations for this instance.
        """

        super().__init__(global_ctx, local_ctx)

        self.train_data = self.local.config['parameters']['train_data']  
        self.ref_data = self.local.config['parameters']['ref_data']  
        self.forget_set = self.dataset.partitions[self.ref_data]
    
    def __unlearn__(self):
        """
        Retrain the model from scratch with a specific (sub)set of the full dataset (usually retain set to evaluate the performance of the model after unlearning)
        """

        cfg_dataset = self.dataset.local_config 
        cfg_model = self.predictor.local_config

        #Create Dataset
        data_manager = self.global_ctx.factory.get_object(Local(cfg_dataset))
        data_manager.revise_split(self.train_data, self.forget_set)
    
        #Create Predictor
        current = Local(cfg_model)
        current.dataset = data_manager
        predictor = self.global_ctx.factory.get_object(current)
            
        return predictor
    
    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['train_data'] = self.local.config['parameters'].get("train_data", 'train')  # Default train data is train
        self.local.config['parameters']['ref_data'] = self.local.config['parameters'].get("ref_data", 'forget')  # Default reference data is forget