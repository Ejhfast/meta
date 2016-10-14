import time
import inspect
import dill
import getpass
import numpy as np
#from scipy.stats import ttest_ind
from datetime import datetime
from collections import defaultdict,Counter
from bson.objectid import ObjectId
from bson import Binary
from bson.json_util import loads as bson_loads
from bson.json_util import dumps as bson_dumps
import re
import sys
import os
from .timeout import timeout, timeout_dumb
from collections import defaultdict,Counter
import random
import requests
from . import helpers as util


class WebCache:

    def __init__(self, webpath="http://meta-lang.org"):
        self.path = webpath

    def version_key(self,s):
        return "\t".join([s["name"],s["file"],s["user"]])

    def record_function(self, snippet_data):
        resp = requests.post(self.path + "/record_function", data={"bson":bson_dumps(snippet_data)})
        return resp.json()

    def record_call(self, runtime_info):
        requests.post(self.path + "/record_call", data={"bson":bson_dumps(runtime_info)})

    def record_bug(self, bug_data):
        requests.post(self.path + "/record_bug", data={"bson":bson_dumps(bug_data)})

    def clear_runtime(self, id_):
        requests.post(self.path + "/clear_runtime", data={"id":id_})

    def similar_funcs(self, text, typ=None):
        resp = requests.get(self.path + "/similar_funcs", json={"text":text, "type":typ})
        # resp is a list of snippet json data
        return bson_loads(resp.text)

    def id2func(self, id_):
        resp = requests.get(self.path + "/id_to_func", params={"id":id_})
        # resp is a json snippet
        return bson_loads(resp.text)

    def get_ids(self, ids):
        resp = requests.get(self.path + "/get_ids", json={"ids":ids})
        return bson_loads(resp.text)

    def id2execution(self, id_):
        resp = requests.get(self.path + "/id_to_execution", params={"id":id_})
        # resp is a list of execution json
        return bson_loads(resp.text)

    def id2bugs(self, id_):
        resp = requests.get(self.path + "/id_to_bugs", params={"id":id_})
        return bson_loads(resp.text)

    def update_snippet(self, id_, updates):
        requests.post(self.path + "/update_snippet", data={"id":id_, "bson":bson_dumps(updates)})

    def match_type(self, typ=None, text=None, field="dynamic_type", l=100):
        resp = requests.get(self.path + "/match_type", params={"text":text, "type":typ, "field":field, "limit":l})
        return bson_loads(resp.text)

    def get_all_snippet_ids(self):
        resp = requests.get(self.path + "/all_snippet_ids")
        return bson_loads(resp.text)

    def get_all_docstrings(self):
        resp = requests.get(self.path + "/all_docstrings")
        return bson_loads(resp.text)

class Meta:

    def __init__(self, userid="global", user=None, sandbox=False, debug=False, max_obj_size=20000, backend_url="https://meta-backend.herokuapp.com", overhead=False, globals=globals(), jupyter=False, inspect=inspect):
        self.cache = WebCache(backend_url)
        self.globals = globals
        self.__framework_test__ = sandbox
        self.debug = debug
        self.jupyter = jupyter
        self.inspect = inspect
        self.overhead = overhead
        self.overhead_call = []
        self.overhead_load = []
        self.overhead_create = []
        self.max_obj_size = max_obj_size
        self.overhead_begintime = time.clock()
        self.optimize = bool(os.environ.get("OPT", False))
        self.backoff = util.str_to_bool(os.environ.get("AUTO_PATCH", False))
        if user:
            self.user = user
        else:
            self.user = getpass.getuser()

    def bind(self,id,imports=[]):
        def bind_decorator(func):
            def func_wrapper(*args, **kwargs):
                t_overhead1 = time.clock()
                instrumented = False
                try:
                    t1 = time.clock()
                    v = func(*args, **kwargs)
                    t2 = time.clock()
                except Exception as e:
                    # record bad input
                    bad_input = {"args":Binary(dill.dumps(args)), "snippet_id":id, "time":datetime.now(),
                                 "exception_type":type(e).__name__, "exception":Binary(dill.dumps(e))}
                    self.cache.record_bug(bad_input)
                    raise e
                if random.randint(0,func_wrapper.called) == 0:
                    instrumented = True
                    params = tuple(list(args)+[v])
                    sig = " -> ".join([util.pp_type(x) for x in util.fancy_type(params).__tuple_params__])
                    # args_ = [a if type(a).__name__ != 'generator' else type(a) for a in args]
                    # v_ = v if type(v).__name__ != 'generator' else type(v)
                    #if sys.getsizeof(args) < self.max_obj_size and sys.getsizeof(v) < self.max_obj_size and not type(v).__name__ == 'generator':
                    if not type(v).__name__ == 'generator':
                        runtime = {"snippet_id":id, "call":Binary(dill.dumps((args,v))),"time_running":t2-t1,"file":__file__,
                               "user":self.user,"time":datetime.now(),"type":sig, "imports":imports}
                        if sys.getsizeof(runtime["call"]) < self.max_obj_size:
                            self.cache.record_call(runtime)
                func_wrapper.called += 1
                t_overhead2 = time.clock()
                if self.overhead: self.overhead_call.append({"id":id,"meta_time":t_overhead2-t_overhead1,"normal_time":t2-t1, "inst":instrumented})
                return v
            func_wrapper.called = 0
            return func_wrapper
        return bind_decorator


    def __call__(self, d_str=None, imports=None, parent=None):
        def real_dec(func):
            t_overhead1 = time.clock()
            imports_copy = imports
            src = "".join(self.inspect.getsourcelines(func)[0][1:])
            if imports_copy == None:
                imports_copy = util.list_imports(self.globals,src)
            f_name = func.__name__
            is_undef = util.is_func_undefined(src)
            annote = {k:util.pp_type(v) for k,v in func.__annotations__.items()}
            arg_type = [annote[a] if a in annote else "*" for a in func.__code__.co_varnames]
            ret_type = annote["return"] if "return" in annote else "*"
            t_sig = " -> ".join(arg_type+[ret_type])
            if self.jupyter:
                file_data = "jupyter"
            else:
                file_data = sys.modules[func.__module__].__file__
            # possibly I want to mark when we're recording input fed to a magically shifted function
            snippet_data = {
                "source":src, "imports":imports_copy,
                "name":f_name, "type":t_sig,
                "undefined":is_undef, "doc":d_str,
                "parent":parent,
                "created_at":datetime.now(),
                "file":file_data,
                "user":self.user,
            }
            id_ = self.cache.record_function(snippet_data)
            if is_undef:
                old_func = func
                func = self.search(d_str)
                id_ = func.__meta_id__
            snippet_data["func"] = func
            inner = MetaFunction(id_, self)#, snippet_data)
            t_overhead2 = time.clock()
            if self.overhead: self.overhead_create.append(t_overhead2-t_overhead1)
            return inner
        return real_dec

    def measure_overhead(self):
        overhead_endtime = time.clock()
        len_load = len(self.overhead_load)
        len_create = len(self.overhead_create)
        sum_load = np.sum(self.overhead_load)
        avg_load = np.average(self.overhead_load)
        avg_create = np.average(self.overhead_create)
        sum_create = np.sum(self.overhead_create)
        len_calls = len(self.overhead_call)
        sum_inst_call = np.sum([x["meta_time"] for x in self.overhead_call if x["inst"]])
        sum_inst_baseline = np.sum([x["normal_time"] for x in self.overhead_call if x["inst"]])
        len_inst = len([x for x in self.overhead_call if x["inst"]])
        sum_meta_call = np.sum([x["meta_time"] for x in self.overhead_call])
        sum_normal_call = np.sum([x["normal_time"] for x in self.overhead_call])
        total_unadjusted = overhead_endtime - self.overhead_begintime
        print(total_unadjusted,"total")
        meta_total = total_unadjusted - sum_normal_call
        normal_total = total_unadjusted - sum_load - sum_meta_call - sum_create
        print(meta_total, normal_total)
        print("{} overall overhead".format(meta_total/normal_total))
        print("{} functions loaded and {} created and {} called".format(len_load, len_create, len_calls))
        print("{} overhead per meta call".format((sum_inst_call - sum_inst_baseline)/len_inst))
        print("call overhead as percent {}".format((sum_meta_call - sum_normal_call) / meta_total))
        print("{} overhead per prob meta call".format((sum_meta_call - sum_normal_call)/len_calls))
        print("{} avg load time".format(avg_load))
        print("{} avg create time".format(avg_create))
        data_out = [total_unadjusted, meta_total / normal_total, len_load, len_create, len_calls, ((sum_meta_call - sum_normal_call)/len_calls), avg_load, avg_create, meta_total, normal_total, (sum_meta_call - sum_normal_call), sum_load, sum_create]
        with open("overhead2.tsv", "a") as myfile:
            myfile.write("\t".join([str(x) for x in data_out])+"\n")


    def load(self, id_: str, test=False, instrument=True):
        t_overhead1 = time.clock()
        mf = MetaFunction(id_, self)
        t_overhead2 = time.clock()
        if self.overhead: self.overhead_load.append(t_overhead2-t_overhead1)
        return mf

    def load_multiple(self, ids):
        snippets = self.cache.get_ids(ids)
        loaded_mfuncs = {str(s["_id"]):MetaFunction(str(s["_id"]), self, data=s) for s in snippets}
        # want to maintain order of request
        return [loaded_mfuncs[i] for i in ids]

    def search(self, doc="", type=None, n=1, examples=None):
        if examples and (not type):
            type = util.type_to_string(examples,False)[0]
        possible = self.cache.match_type(type,doc,l=n)
        scores = [(str(p["_id"]), p["doc"], util.query_string_score(doc,p["doc"])) for p in possible]
        sorted_scores = sorted(scores,key=lambda x: x[2],reverse=True)
        get_funcs = self.load_multiple([s[0] for s in sorted_scores]) #[self.load(s[0]) for s in sorted_scores]
        if examples: get_funcs = util.filter_funcs_by_examples(get_funcs, examples)
        if n==1:
            return get_funcs[0]
        else:
            return get_funcs[:n]

    def compare(self, mf1, mf2):
        execution = mf1.analytics()["recent_calls"] + mf2.analytics()["recent_calls"]
        inputs = [x[0] for x in execution]
        total_t1, total_t2 = [], []
        for i in inputs:
            total_t1 += util.average_times(mf1.raw, i, n=100)
            total_t2 += util.average_times(mf2.raw, i, n=100)
        t1_avg, t2_avg = [np.average(x) for x in [total_t1, total_t2]]
        if t1_avg > t2_avg:
            odds = t1_avg / t2_avg
        else:
            odds = t2_avg / t1_avg
        if True:#ttest_ind(total_t1, total_t2)[1] < 0.05:
            return (odds, t1_avg/len(inputs), t2_avg/len(inputs))
        else:
            return None

    def one_step_search(self, b, e, qs):
        start_n = len(b.split("->"))+1
        b_rgx = re.compile("^"+re.escape(b))
        e_rgx = re.compile(re.escape(e)+"$")
        def extract(x): return (x["_id"],x["doc"],x["for_inference"])
        def inflate(x):
            return [(x[0],x[1],[y_.rstrip().lstrip() for y_ in y.split("->")],y) for y in x[2]]
        def pop_begin(lst,n):
            return [(x[0],x[1],x[2][-1],x[3]) for x in lst if len(x[2]) == n]
        def pop_end(lst):
            return [(x[0],x[1],x[2][0],x[3]) for x in lst if len(x[2]) == 2]
        begin = [inflate(y) for y in [extract(x) for x in self.cache.match_type(b_rgx,qs,field="for_inference",l=100)]]
        end = [inflate(y) for y in [extract(x) for x in self.cache.match_type(e_rgx,qs,field="for_inference",l=100)]]
        begin, end = [list(it.chain(*x)) for x in [begin,end]]
        match_sets = []
        possible_ends = pop_end(end)
        for n in range(start_n,5):
            b_ = pop_begin(begin,n)
            match_sets.append([b_,possible_ends])
        results = []
        for m in match_sets:
            for pair in it.product(*m):
                # print(pair[0],pair[1])
                if pair[0][2] == pair[1][2]:
                    comb_str = pair[0][1].lower() + ' and ' + pair[1][1].lower()
                    results.append((pair, util.query_string_score(comb_str,qs), comb_str))
        s_results = sorted(results,key=lambda x: x[1],reverse=True)
        final_results = []
        for r in s_results:
            s1 = str(r[0][0][0])
            s2 = str(r[0][1][0])
            def gen(s1,s2):
                def compile():
                    f1 = MetaFunction(s1,self)
                    f2 = MetaFunction(s2,self)
                    comb = lambda *x: f2(f1(*x))
                    return comb
                return compile
            final_results.append((gen(s1,s2),r[2],r[1]))
        return final_results

    def broad_search(self,text):
        return self.cache.similar_funcs(text)[:25]

    def all_funcs(self):
        for id_ in self.cache.get_all_snippet_ids():
            yield self.load(id_)

class MetaFunction():

    def __init__(self, id_: str, meta_conn, data=None, load_at_all_costs=True):
        id_ = str(id_).split("/")[-1]
        self.__meta_id__ = id_
        self.meta = meta_conn
        self.duplicates = None
        self.__meta_backoff__ = False
        self.__show_search__ = False
        if not data:
            data = self.meta.cache.id2func(id_)
        src, func_name, imports = [data[k] for k in ["source","name","imports"]]
        try:
            new_f = util.load_source(func_name, src, imports)
        except Exception as e:
            if load_at_all_costs:
                new_f = util.any_func
                pass
            else:
                raise e
        if "show_search" in data and data["show_search"] == True:
            self.__show_search__ = True
        self.__meta_doc__ = data["doc"]
        self.__meta_type__ = data["type"]
        self.__meta_source__ = data["source"]
        self.__meta_imports__ = imports
        if "for_inference" in data:
            self.__meta_for_inference__ = data["for_inference"]
        else:
            self.__meta_for_inference__ = None
        if "avg_time" in data:
            self.__meta_avg_time__ = data["avg_time"]
        else:
            avg = self.analytics()['average run time']
            self.__meta_avg_time__ = avg
        self.raw = new_f
        self.sandbox_io = new_f # need to fix sandbox AGAIN
        # self.sandbox_io = iosandbox[func_name] #fake_io(new_f,debug=True)
        # self.instrument_sandbox = self.meta.bind(id_)(self.sandbox_io)
        self.instrument_f = self.meta.bind(id_)(new_f)
        self.__name__ = func_name

    def __repr__(self):
        return self.__meta_doc__ + " (" + self.__name__ + ")"

    def __call__(*args,**kwargs):
        # args[0] = self?
        self = args[0]
        args = args[1:]
        if self.meta.optimize:
            return self.optimize()
        if self.meta.__framework_test__:
            return self.instrument_sandbox(*args,**kwargs)
        if self.meta.backoff or self.__meta_backoff__:
            self.find_equivalents()
            for d in [self]+self.duplicates:
                try:
                    new_ret = d.instrument_f(*args,**kwargs)
                    print("Warning: Auto-patched {} with {}".format(self.__name__, d.__name__))
                    return new_ret
                except:
                    pass
            raise Exception("No backoff function could save this input")
        else:
            ret = self.instrument_f(*args,**kwargs)
            if self.__show_search__ == False:
                self.meta.cache.update_snippet(self.__meta_id__, {"show_search":True})
                self.__show_search__ = True
            return ret

    def get_dynamic_type_sig(self, _expand_ = True, recompute=False):
        # what if no inputs or no calls?
        if self.meta.debug:
            print("get_dynamic_type_sig",self.__meta_doc__,self.__name__)
        execution = [x for x in self.meta.cache.id2execution(self.__meta_id__)]
        # if len(execution) == 0: return []
        if execution:
            parsed = [util.safe_load(x["call"]) for x in execution]
            self.__meta_dynamic_type__ = util.type_to_string(parsed, True)
            self.__meta_for_inference__ = util.type_to_string(parsed, False)
            # update db
            self.meta.cache.update_snippet(self.__meta_id__,
                                           {"dynamic_type":self.__meta_dynamic_type__,
                                            "for_inference":self.__meta_for_inference__})
            if _expand_:
                return self.__meta_for_inference__
            else:
                return self.__meta_dynamic_type__

    def test_as(self, other_meta_func, debug=False, fail_count=float("inf"), io_skip=False):
        #ex1, ex2 = [self.meta.cache.id2execution(x.__meta_id__) for x in [self,other_meta_func]]
        exs = self.meta.cache.id2execution(self.__meta_id__)
        execution = set([x["call"] for x in exs]) #list(ex1)+list(ex2)])
        if len(execution) == 0:
            return 0
        pass_, fail_ = 0.0, 0.0
        try:
            func_1 = timeout_dumb(2)(dill.dumps(self.raw,recurse=True))
            func_2 = timeout_dumb(2)(dill.dumps(other_meta_func.raw,recurse=True))
        except Exception as e:
            print(e,file=sys.stderr)
            return 0
        for i,test in enumerate(execution):
            if fail_ == fail_count:
              return pass_ / (pass_ + fail_)
            # if self.sandbox_io.done_io:
            #     return 0
            if test == None: continue
            call = util.safe_load(test)
            if call == None: continue
            args, ret = call[0], call[1]
            # print(i,args)
            try:
                v1 = func_1(*args)
                v2 = func_2(*args)
                if v1 == None and v2 == None:
                    fail_ += 1.0
                    continue
                if not util.equal_fvals(v1,v2):
                    if debug: print("Failed test {} with args {}: returned {} but expected {}".format(i,args,v1,v2))
                    fail_ += 1.0
                else:
                    if debug: print("Passed test {}, {}".format(i,args))
                    pass_ += 1.0
            except Exception as inst:
                if debug:
                    print("Failed test {} with exception".format(i))
                    print(inst)
                fail_ += 1.0
        return pass_ / (pass_ + fail_)

    def find_equivalents(self):
        if self.duplicates:
            return self.duplicates
        else:
            snippet = self.meta.cache.id2func(self.__meta_id__)
            if "for_inference" in snippet and snippet["for_inference"] and len(snippet["for_inference"]) > 0:
                tp = snippet["for_inference"]
                matches = self.meta.cache.similar_funcs(snippet["doc"], tp)
                to_ret = []
                for m in matches:
                    if str(m["_id"]) == self.__meta_id__: continue
                    #print("test as",str(m["_id"]),str(id_))
                    try:
                        loaded_f = self.meta.load(str(m["_id"]))
                    except:
                        print("failed to load {}".format(m["_id"]))
                        continue
                    #print(loaded_f,file=sys.stdout)
                    score = self.test_as(loaded_f, fail_count=1, debug=False)
                    if score >= 1:
                        to_ret.append(loaded_f)
                self.duplicates = to_ret
                return to_ret
            else:
                self.duplicates = []
                return []

    def optimize(self,k="average run time"):
        dups = self.find_equivalents()
        curr_time = round(self.analytics()[k]*1000,4)
        rank = sorted([(d,round(d.analytics()[k]*1000,4)) for d in dups],key=lambda x: x[1])
        if len(rank)>0 and rank[0][1] < curr_time:
            print("Warning: optimizing {} with {}:\nSee: http://www.meta-lang.org/snippets/{}\nAverage time of {}ms vs. {}ms".format(self.__name__, rank[0][0].__name__, rank[0][0].__meta_id__, rank[0][1], curr_time),file=sys.stderr)
            selected = rank[0][0].raw
        else:
            selected = self
        def opt(*args,**kwargs):
            return selected(*args,**kwargs)
        return opt

    def optimize2(self, n_inputs=10, n_times=10, p_cutoff=0.05):
        dups = self.find_equivalents()
        # can make this more efficient
        all_inputs = [util.safe_load(x)[0] for x in set([x["call"] for x in self.meta.cache.id2execution(self.__meta_id__)])]
        test_inputs = random.sample([x for x in all_inputs if x != None], min(n_inputs,len(all_inputs)))
        self_times = self.time_on_inputs(test_inputs, n_times).flatten()
        self_average = np.average(self_times)
        out = []
        print(dups)
        for d in dups:
            d_times = d.time_on_inputs(test_inputs, n_times).flatten()
            d_average = np.average(d_times)
            print(self_average,d_average)
            if d_average < self_average:
                # print(self_average / d_average)
                # _, pval = ttest_ind(self_times, d_times)
                # if pval < 0.05:
                out.append([d, self_average / d_average])
        return out

    def time_on_inputs(self, inputs, n=10):
        times_per_input = []
        for i in inputs:
            timed_func = util.timing(self.raw)
            times = [timed_func(*i)[0] for _ in range(0,n)]
            times_per_input.append(times)
        return np.array(times_per_input)

    def analytics(self):
        execution = self.meta.cache.id2execution(self.__meta_id__)
        execution = list(execution)
        avg_time = np.average([x["time_running"] for x in execution])
        sorted_exe = sorted(execution,key=lambda x: x["time"],reverse=True)
        all_rets = [util.safe_load(x["call"]) for x in sorted_exe]
        all_types = Counter([util.fancy_type(x) for x in all_rets])
        last_ten = [{"input":x[0], "output": x[1]} for x in all_rets[:10]]
        self.meta.cache.update_snippet(self.__meta_id__,{"avg_time":avg_time})
        return {"average run time":avg_time, "recent_calls":last_ten, "types":all_types}

    def examples(self):
        return self.analytics()["recent_calls"]

    def url(self):
        return "http://www.meta-lang.org/snippets/{}".format(self.__meta_id__)

    def get_type(self):
        def replace_any(x):
            return re.sub(r"\[[^\]]*\]","",x)
        self.get_dynamic_type_sig()
        return list(set([replace_any(x) for x in self.__meta_for_inference__]))
        #return self.__meta_dynamic_type__

    def exceptions(self):
        bugs = self.meta.cache.id2bugs(self.__meta_id__)
        out = []
        for b in bugs:
            b["args"] = dill.loads(b["args"])
            b["exception"] = dill.loads(b["exception"])
            out.append(b)
        return out

    def overhead(self,n):
        execution = self.meta.cache.id2execution(self.__meta_id__)
        inputs = [util.safe_load(x["call"])[0] for x in execution]
        time_raw, time_meta = 0.0, 0.0
        c = 0
        for args in inputs:
            for _ in range(0,n):
                sometimes = self.instrument_sandbox
                time_raw += util.average_times(self.sandbox_io,args,n=1)[0]
                if random.randint(0,c) != 0:
                    sometimes = self.sandbox_io
                time_meta += util.average_times(sometimes,args,n=1)[0]
                c += 1
        return time_raw, time_meta

    def auto_patch(self,pos=True):
        self.__meta_backoff__ = pos

class Test:

    def __init__(self):
        self.meta = Meta()

    def test_type_sigs(self):
        print("Testing dynamic type sig inference".upper())
        for sid in self.meta.cache.get_all_snippet_ids():
            print(sid,self.meta.get_dynamic_type_sig(sid,_expand_=False))
        print("")

    def test_load(self):
        print("Testing function load from id".upper())
        for sid in self.meta.cache.get_all_snippet_ids():
            print(self.meta.load(sid).__name__)
        print("")

    def test_search(self):
        print("Testing function load from docstr".upper())
        for docstr in self.meta.cache.get_all_docstrings():
            print(docstr)
            print(self.meta.search(docstr).__name__)
        print("")

    def test_testgen(self):
        print("Testing generation of test cases".upper())
        for sid in self.meta.cache.get_all_snippet_ids():
            f = self.meta.load(sid,test=True)
            print(f.__name__)
            self.meta.test_as(f,sid)
        print("")

    def test_duplicates(self):
        print("Testing possible duplicate finding")
        for sid in self.meta.cache.get_all_snippet_ids():
            try:
                f = self.meta.load(sid)
            except:
                continue
            dups = f.find_equivalents()
            #print(sid)
            if len(dups) > 0:
                print(sid,self.meta.cache.id2func(sid)["doc"])
                print("Possible duplicates",dups)
                # now do a thing where we actually look at input/output

    def test_typesearch(self,queries=None):
        print("Testing type search".upper())
        if queries == None:
            queries = [
                {"type": "List[int] -> int", "doc": "sum the list"},
                {"type": "int -> str", "doc": "convert int to str"},
                {"type": "int -> int -> int", "doc":"add numbers"}
            ]
        for q in queries:
            print("Searched for '{}' of type '{}'".format(q["doc"],q["type"]))
            funcs = self.meta.type_search(q["type"],q["doc"])
            for tp in funcs:
                print((tp[0].__name__,tp[1],tp[2]))

    def test_overhead(self,n):
        raw_total, meta_total = 0.0, 0.0
        for sid in self.meta.cache.get_all_snippet_ids():
            f = self.meta.load(sid)
            try:
                raw_, meta_ = f.overhead(n)
                raw_total += raw_
                meta_total += meta_
                print(raw_total / meta_total)
            except Exception as e:
                print(e)
                pass


    def test_all(self):
        self.test_type_sigs()
        self.test_load()
        self.test_search()
        #self.test_testgen()
        self.test_typesearch()
