"""
We use Performance Application Programming Interface (PAPI) to evaluate the performance of the models in terms of instructions executed.
The following error may occur when running the code. pypapi.exceptions.PapiPermissionError: Permission level does not permit operation. (PAPI_EPERM)
Fix:
    Run the code as superuser (root)
    Adjust the kernel parameters to allow non-root access to performance counters. You can temporarily set these parameters:
sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
    For a more permanent solution, add the following lines to /etc/sysctl.conf:
kernel.perf_event_paranoid = -1
kernel.perf_event_max_sample_rate = 100000
Then apply the changes:
sudo sysctl -p
"""
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os
from pypapi import papi_low as papi
from pypapi import events

if __name__ == "__main__":

    papi.library_init()
    evs = papi.create_eventset()
    papi.add_event(evs, events.PAPI_SP_OPS)

    papi.start(evs)

    dt = DecisionTreeClassifier()
    dt.fit(np.random.random((100,10)), np.random.randint(0,2,100))

    result = papi.stop(evs)

    papi.cleanup_eventset(evs)
    papi.destroy_eventset(evs)

    print(result)