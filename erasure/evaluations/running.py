import time
import torch.profiler
import platform

#if platform.system() != 'Darwin':
#    from pypapi import papi_low as papi
#    from pypapi import events as papi_events

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

        self.events = self.params["events"]

        papi.library_init()
        self.evs = papi.create_eventset()

        for event_name in self.events:
            event = getattr(papi_events, event_name)
            print(event, papi_events.PAPI_LST_INS)
            papi.add_event(self.evs, event)
    
    def check_configuration(self):
        self.params["events"] = self.params.get("events", ['PAPI_TOT_INS', 'PAPI_TOT_CYC', 'PAPI_LST_INS'])

    def process(self, e: Evaluation):
        if not e.unlearned_model:
            papi.start(self.evs)

            super().process(e)

            result = papi.stop(self.evs)

            for i, event_name in enumerate(self.events):
                e.add_value(event_name, result[i])
        
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
                with_flops=True
            ) as prof:
                
                super().process(e)

            filtered_events = [event for event in prof.events() if event.flops not in (None, 0)]
            metric_value = sum(event.flops for event in filtered_events)

            e.add_value('TorchFlops', metric_value)

        return e

