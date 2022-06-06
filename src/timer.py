# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 09:24:13 2022

@author: iwenc
"""

import threading
import ctypes
import time
  
class thread_with_exception(threading.Thread):
    def __init__(self, func):
        threading.Thread.__init__(self)
        self.func = func
        self.finished = False
        self.result = None
             
    def run(self):
        try:
            self.finished = False
            self.result = None
            self.result = self.func()
            self.finished = True
        finally:
            pass
          
    def get_id(self):
 
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id
  
    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

def run_with_time_limit(time_limit, f):
    '''
    Run f with a time limit. 

    Parameters
    ----------
    time_limit : number
        Time limit in seconds
    f : function
        A function to be called

    Returns
    -------
    (True, result): if f finished within time limit, result is the returned result
    (False, None): if f cannot finish within time limit

    '''
    t = thread_with_exception(f)
    t.start()
    start_time = time.time()
    end_time = start_time + time_limit
    while time.time() < end_time and not t.finished:
        time.sleep(0.1)
    t.raise_exception()
    t.join()
    return (t.finished,t.result)

def f():
    while True:
        time.sleep(0.1)
        print("running")

def f1():
    for i in range(5):
        time.sleep(0.1)
        print(f"running {i=}")
    return 3.5

if __name__ == "__main__":
    print(f"finished: {run_with_time_limit(3,f)}")
    print(f"finished: {run_with_time_limit(3,f1)}")
