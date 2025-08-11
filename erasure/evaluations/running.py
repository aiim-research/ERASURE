import time
import torch.profiler
import platform
import sys
from contextlib import contextmanager,nullcontext

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


def get_peak_rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF)
        peak = r.ru_maxrss
        if sys.platform == "darwin":
            return peak / (1024 * 1024)  
        else:
            return peak / 1024           
    except Exception:
        # Fallback/cross-platform
        try:
            import psutil, os
            p = psutil.Process(os.getpid())
            mi = p.memory_info()
            peak = getattr(mi, "peak_wset", None) or getattr(mi, "peak_rss", None) or mi.rss
            return peak / (1024 * 1024)
        except Exception:
            return None

@contextmanager
def python_alloc_tracker():
    import tracemalloc
    tracemalloc.start()
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        python_alloc_tracker.current_mb = current / (1024 * 1024)
        python_alloc_tracker.peak_mb = peak / (1024 * 1024)
        tracemalloc.stop()

@contextmanager
def torch_gpu_peak_tracker():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        yield
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch_gpu_peak_tracker.peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                torch_gpu_peak_tracker.peak_mb = None
        except Exception:
            torch_gpu_peak_tracker.peak_mb = None


class RunTime(UnlearnRunner):
    """ Wallclock running time + memory usage """
    def process(self, e: Evaluation):
        if not e.unlearned_model:

            device = e.predictor.device

            dict_memory = {'cuda': torch_gpu_peak_tracker, 'cpu': python_alloc_tracker}

            factory = dict_memory.get(device)
            tracker = factory() if callable(factory) else nullcontext()

            start_time = time.time()

            with tracker:    
                super().process(e)

            runtime = time.time() - start_time
            e.add_value('RunTime', runtime)


            peak_rss_mb = get_peak_rss_mb()
            if peak_rss_mb is not None:
                e.add_value('PeakRSS_MB', peak_rss_mb)

            peak_mb = getattr(tracker, 'peak_mb', None)
            current_mb = getattr(tracker, 'current_mb', None)

            if peak_mb is not None:
                if self.memory_metric == 'CUDA':
                    e.add_value('CudaPeak_MB', peak_mb)
                elif self.memory_metric == 'python':
                    e.add_value('PyHeapPeak_MB', peak_mb)

            if current_mb is not None and self.memory_metric == 'python':
                e.add_value('PyHeapCurrent_MB', current_mb)

            return e
    
'''
class RunTime(UnlearnRunner):
    """ Wallclock running time to execute the unlearn """
    def process(self, e: Evaluation):
        if not e.unlearned_model:
            start_time = time.time()

            super().process(e)
            metric_value = time.time() - start_time

            e.add_value('RunTime', metric_value)

        return e
'''

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

