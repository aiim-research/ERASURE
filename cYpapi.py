#!/usr/bin/env python3

# imports necessary for script
import cypapi as cyp
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Manually Install 
# https://github.com/icl-utk-edu/cyPAPI?tab=readme-ov-file
# https://github.com/icl-utk-edu/papi/wiki/Downloading-and-Installing-PAPI
if __name__ == '__main__':
    # initialize cyPAPI library
    cyp.cyPAPI_library_init(cyp.PAPI_VER_CURRENT)

    # check to make sure cyPAPI has been intialized
    if cyp.cyPAPI_is_initialized() != 1:
        raise ValueError("cyPAPI has not been initialized.\n")

    # test real time cyPAPI functions
    try: 
        # real time in clock cycles
        rt_cc_start = cyp.cyPAPI_get_real_cyc()
        rt_ns_start = cyp.cyPAPI_get_real_nsec()
        rt_ms_start = cyp.cyPAPI_get_real_usec()
            
        eventset = cyp.CypapiCreateEventset()
        #eventset.add_event(cyp.PAPI_TOT_INS)
        #eventset.start()

        dt = DecisionTreeClassifier()
        dt.fit(np.random.random((100,10)), np.random.randint(0,2,100))
        
        rt_cc_stop = cyp.cyPAPI_get_real_cyc()
        rt_ns_stop = cyp.cyPAPI_get_real_nsec()        
        rt_ms_stop = cyp.cyPAPI_get_real_usec()
        #values = eventset.stop()

    except Exception:
        print('\033[0;31mFAILED\033[0m')
        raise
    # collection of real time succeeded
    else:
        print('Real time in clock cycles: ', rt_cc_stop - rt_cc_start)
        print('Real time in nanoseconds: ', rt_ns_stop - rt_ns_start)
        print('Real time in microseconds: ', rt_ms_stop - rt_ms_start)
        print('\033[0;32mPASSED\033[0m');
        #print(values)