from erasure.core.unlearner import Unlearner
from erasure.utils.config.global_ctx import Global
import torch
from collections import Counter
import heapq
import random


class GoldModel(Unlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        super().__init__(global_ctx, local_ctx)
        self.ref_data = local_ctx.config['parameters'].get("ref_data", 'retain set')  # Default reference data is retain
    
    def __unlearn__(self):

        # TODO: I think that instead of passing what I want to remove, just pass the entire training set 
        predictor = self.get_retrained(self.dataset.partitions['forget set'])
        return predictor