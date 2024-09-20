
from erasure.utils.config.global_ctx import Global
from abc import abstractmethod


class Base:
    def __init__(self, global_ctx: Global):
        self.global_ctx = global_ctx
        self.info = self.global_ctx.logger.info
        
class Configurable(Base):
    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx)
        self.local = local_ctx
        self.local_config = self.local.config
        self.params = self.local.config['parameters']
        self.check_configuration()
        self.init()

    def check_configuration(self):
        self.local.config['parameters'] = self.local.config.get('parameters',{})


    @abstractmethod
    def init(self):
        pass
