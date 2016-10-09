from .typing import List,Dict,Any,Tuple,Set,Callable,Union
import itertools as it
import dill
import numpy as np
import numpy as np
import time
from .timeout import timeout
import types
import re

# list all imported modules, doesn't work with "as" or "from"
# optionally filter by their presence in src
def list_imports(globals, src=None):
    imps = set()
    for name, val in globals.items():
        if isinstance(val, types.ModuleType):
            name = val.__name__
            if src:
                if re.compile(name+"\.").search(src):
                    imps.add(name)
                else:
                    continue
            imps.add(name)
    return list(imps)

def str_to_bool(s):
    if s == "True": return True
    return False

# see if meta function is currently undefined
def is_func_undefined(source : str) -> bool:
    return source.rstrip().split("\n")[-1].lstrip() == "pass"

# compute structural type representation for a value
def fancy_type(o):
    if type(o) == type([]):
        if len(o) == 0: return List[Any]
        else: return List[Union[tuple([fancy_type(t_) for t_ in o])]]
    elif type(o) == type(()):
        return Tuple[tuple([fancy_type(t_) for t_ in o])]
    elif type(o) == type({}):
        #return Dict[Any,Any]
        if not o:
            return Dict[Any,Any]
        return Dict[Union[tuple([fancy_type(t_) for t_ in o.keys()])],Union[tuple([fancy_type(t_) for t_ in o.values()])]]
    if type(o) == type(set()):
        return Set[Union[tuple([fancy_type(t_) for t_ in o])]]
    if type(o) == type(fancy_type):
        return Callable[...,Any]
    else:
        return type(o)

# remove 'typing.' when printing types
def pp_type(t):
    if type(t).__module__ == 'metalang.typing':
        return str(t).replace("metalang.typing.","")
    elif t == None:
        return "None"
    else:
        return t.__name__

# dumb heuristic for combined query string similarity
def query_string_score(s1,s2):
    s1, s2 = s1.lower(), s2.lower()
    wds1, wds2 = [set(x.split()) for x in [s1, s2]]
    return len(wds1 & wds2) / float(len(wds1) + len(wds2))

# catch exceptions when loading unsupported argument calls
def safe_load(s):
    try:
        o = dill.loads(s)
        return o
    except:
        return None

# return functions that execute as expected on example inputs
def filter_funcs_by_examples(funcs, examples):
    out = []
    for meta_f in funcs:
        is_good = True
        for in_,out_ in examples:
            try:
                test_out = timeout(2)(meta_f.sandbox_io)(*in_)
                if not equal_fvals(test_out,out_):
                    is_good = False
                    break
            except:
                is_good = False
                break
        if is_good:
            out.append(meta_f)
    return out

# normalize certain kinds of return values for equivalence testing
def ret_normalize(r):
    if isinstance(r,dict):
        return ("dict", sorted(dict(r)))
    elif isinstance(r,np.ndarray):
        try:
            return list(r)
        except:
            return r
    else:
        return r

# test whether two meta functions returned the same value
def equal_fvals(x,y):
    # so hacky
    x,y = [ret_normalize(x_) for x_ in [x,y]]
    return str((x)) == str((y))

# So this is kind of crap, but if you pass add_=[], what it does is break up Unions
def expand_type(t,add_=[Any]):
    if type(t) == type(List):
        return add_ + [List[x] for x in expand_type(t.__parameters__[0],add_)]
    elif type(t) == type(Tuple):
        tups = t.__tuple_params__
        expand_tups = [expand_type(t_,add_) for t_ in tups]
        return add_ + [Tuple[c] for c in it.product(*expand_tups)]
    elif type(t) == type(Dict):
        params = t.__parameters__
        expand_params = [expand_type(t_,add_) for t_ in params]
        return add_ + [Dict[c] for c in it.product(*expand_params)]
    elif type(t) == type(Set):
        return add_ + [Set[x] for x in expand_type(t.__parameters__[0],add_)]
    elif type(t) == type(Union):
        tups = t.__union_params__
        expand_tups = [expand_type(t_,add_) for t_ in tups]
        return add_ + [Union[c] for c in it.product(*expand_tups)]+list(it.chain(*expand_tups))
    else:
        return add_ + [t]

# convert a list of types into a string representation of type signature
def type_to_string(params, __expand__=True):
    params = [tuple(list(i[0])+[i[1]]) for i in params if i != None]
    if len(params) == 0: return []
    sig = Union[tuple([fancy_type(t_) for t_ in params])]
    expanded = expand_type(sig) if __expand__ else expand_type(sig,add_=[])
    p_sig = []
    for s in expanded:
        if type(s) == type(Tuple):
            #print(s)
            p_sig.append(" -> ".join([pp_type(x) for x in s.__tuple_params__]))
            #print(p_sig[-1])
    return p_sig

# time a function and prepend time to return value
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        return ((time2-time1), ret)
    return wrap

# sample many running times for function input
def average_times(f,args,n=10):
    total = []
    for _ in range(0,n):
        t, _ = timing(timeout(1)(f))(*args)
        total.append(t)
    return total

# a meta placeholder function
def any_func(*args, **kwargs): raise Exception("Not real")

# load meta function imports into sandbox
def load_imports(imports,env):
    # LOL, seriously
    exec("from typing import List,Dict,Any,Tuple,Set,Callable,Union",env)
    for i in imports:
        if type(i) == type(""):
            exec("import {}".format(i),env)
        elif type(i) == type({}):
            exec("from {} import {}".format(i["from"],i["import"]),env)

# load function source and sandbox imports
def load_source(func_name, source, imports):
    sandbox = {}
    load_imports(imports, sandbox)
    exec(source, sandbox)
    return sandbox[func_name]
