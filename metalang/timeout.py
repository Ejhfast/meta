from multiprocess import Pool
import dill

def timeout(sec=1):
    def wrapper(f):
        def inner(*args):
            pool = Pool(processes=1)
            res = pool.apply_async(f,args)
            try:
                v = res.get(timeout=sec)
            except Exception as inst:
                print(inst)
                v = None
            finally:
                pool.terminate()
                return v
        return inner
    return wrapper

def loader(pkl,args):
    f = dill.loads(pkl)
    return f(*args)

def timeout_dumb(sec=1):
    def wrapper(pkl):
        def inner(*args):
            pool = Pool(processes=1)
            res = pool.apply_async(loader,(pkl,args))
            try:
                v = res.get(timeout=sec)
            except Exception as inst:
                # print(inst)
                v = None
            finally:
                pool.terminate()
                return v
        return inner
    return wrapper
