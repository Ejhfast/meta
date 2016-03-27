from context import Meta

meta = Meta()

merge_two_dicts = meta.load("http://www.meta-lang.org/snippets/56ee471dc0cb8f7470fbb500")

dict1 = {x:i for x,i in enumerate(range(0,200))}
dict2 = {x:i for x,i in enumerate(range(0,200))}

# or pass OPT=True python ...
merge_two_dicts(dict1,dict2)
# merge_two_dicts.optimize()(dict1,dict2)
