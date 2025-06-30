from erasure.data.preprocessing.preprocess import Preprocess
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local

class PatternMatch(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.patterns = self.local_config['parameters']['patterns']

    def process(self, X, y, z):
        z_match = 0
        for pattern in self.patterns:
            if z == pattern:
                z_match = 1
                break
        return X, y, z_match

    def check_configuration(self):
        super().check_configuration()
        self.patterns = self.local_config['parameters'].get('patterns',None)

        for pattern in self.patterns:
            assert isinstance(pattern, list), f"Each pattern must be a list, got {type(pattern)}"

