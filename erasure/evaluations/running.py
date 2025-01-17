import time
import torch.profiler
import platform

if platform.system() != 'Darwin':
    from pypapi import papi_low as papi
    from pypapi import events as papi_events

from erasure.core.measure import Measure
from erasure.evaluations.manager import Evaluation
from erasure.utils.config.local_ctx import Local


class UnlearnRunner(Measure):
    """ Generic measure class that calls the unlearn() method """
    def init(self):
        super().init()
        if 'inner' in self.params:            
            current = Local(self.params['inner'])
            self.inner = self.global_ctx.factory.get_object(current)

    def process(self, e: Evaluation):
        if not hasattr(self,'inner'):
            e.unlearned_model = e.unlearner.unlearn()
        else:
            self.inner.process(e)
        return e
    
class ChainOfRunners(UnlearnRunner):
    """ Utility Class for building a nested chain of Runners """

    def init(self):
        prev_cfg = {}    
        for cls in reversed(self.params['runners']):
            curr_cfg = {'class':cls}
            if bool(prev_cfg):
                curr_cfg['parameters'] = {}
                curr_cfg['parameters']['inner'] = prev_cfg
            prev_cfg = curr_cfg
        
        current = Local(prev_cfg)
        self.head = self.global_ctx.factory.get_object(current)

    def process(self, e: Evaluation):
        self.head.process(e)

        return e

class RunTime(UnlearnRunner):
    """ Wallclock running time to execute the unlearn """
    def process(self, e: Evaluation):
        if not e.unlearned_model:
            start_time = time.time()

            super().process(e)
            metric_value = time.time() - start_time

            e.add_value('RunTime', metric_value)

        return e

class PAPI(UnlearnRunner):
    """ PAPI Events to execute the unlearn """
    def init(self):
        super().init()
        papi.library_init()
        self.evs = papi.create_eventset()
        papi.add_event(self.evs, papi_events.PAPI_TOT_INS)
        papi.add_event(self.evs, papi_events.PAPI_TOT_CYC)
        papi.add_event(self.evs, papi_events.PAPI_LST_INS)

    def process(self, e: Evaluation):
        if not e.unlearned_model:
            papi.start(self.evs)

            super().process(e)

            result = papi.stop(self.evs)
            #e.add_value('PAPI', result)
            e.add_value('PAPI_TOT_INS', result[0])
            e.add_value('PAPI_TOT_CYC', result[1])
            e.add_value('PAPI_LST_INS', result[2])

            #papi.cleanup_eventset(self.evs)
            #papi.destroy_eventset(evs)
        
        return e

class TorchFlops(UnlearnRunner):
    """ FLOPS to execute the unlearn (iw works only with PyTorch models) """

    def process(self, e: Evaluation):
        if not e.unlearned_model:
            activities=[torch.profiler.ProfilerActivity.CPU]
                        
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            if torch.xpu.is_available():
                activities.append(torch.profiler.ProfilerActivity.XPU)

            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True
            ) as prof:
                
                super().process(e)

            metric_value = sum(event.flops for event in prof.key_averages())

            e.add_value('TorchFlops', metric_value)

        return e

