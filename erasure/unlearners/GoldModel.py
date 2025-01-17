import copy
from erasure.core.unlearner import Unlearner
from erasure.utils.config.local_ctx import Local

class GoldModel(Unlearner):
    def init(self):
        """
        Initializes the GoldModel class with global and local contexts.
        """
        super().init()

        #Create Dataset
        if self.local.config['parameters']['data'] == 'global':
            data_manager = self.global_ctx.dataset
        else:
            data_manager = self.global_ctx.factory.get_object(Local(self.local.config['parameters']['data']))
    
        #Create Predictor
        self.current = Local(self.local.config['parameters']['predictor'])
        self.current.dataset = self.global_ctx.dataset
    
    def __unlearn__(self):
        """
        Retrain the model from scratch with a specific (sub)set of the full dataset (usually retain set to evaluate the performance of the model after unlearning)
        """

        predictor = self.global_ctx.factory.get_object(self.current)
            
        return predictor
    
    def check_configuration(self):
        super().check_configuration()

        if 'data' not in self.local.config['parameters']:
            self.local.config['parameters']['data'] = 'global'#copy.deepcopy(self.global_ctx.dataset.local_config)

        self.local.config['parameters']['training_set'] = self.local.config['parameters'].get("training_set", 'retain')  # Default train data is retain
        self.local.config['parameters']['cached'] = self.local.config['parameters'].get("cached", False)  # Default cached to False

        if 'predictor' not in self.local.config['parameters']:
            self.local.config['parameters']['predictor'] = copy.deepcopy(self.global_ctx.predictor.local_config)
            self.local.config['parameters']['predictor']['parameters']['cached'] = self.local.config['parameters']['cached']
            self.local.config['parameters']['predictor']['parameters']['training_set'] = self.local.config['parameters']['training_set']
        else:
            self.local.config['parameters']['predictor']['parameters']['training_set'] = self.local.config['parameters']['predictor']['parameters'].get('training_set',self.local.config['parameters']['training_set'])

        self.local.config['parameters']['predictor']['parameters']['cached'] = self.local.config['parameters']['predictor']['parameters'].get('cached',self.local.config['parameters']['cached'])

        
    