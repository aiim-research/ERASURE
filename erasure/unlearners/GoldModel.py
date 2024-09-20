from erasure.core.unlearner import Unlearner
from erasure.utils.config.global_ctx import Global
import torch
from collections import Counter
import heapq
import random


class GoldModel(Unlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        super().__init__(global_ctx, local_ctx)
    
    def unlearn(self):

        return self.get_retrained(self.dataset.partitions['forget set'])