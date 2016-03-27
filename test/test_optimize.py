from context import Meta

meta = Meta()

merge_dictionaries = meta.load("http://www.meta-lang.org/snippets/56ee468ac0cb8f7470fbb338")

dict1 = {x:i for x,i in enumerate(range(0,200))}
dict2 = {x:i for x,i in enumerate(range(0,200))}


print(merge_dictionaries.find_duplicates())

merge_dictionaries.optimize_run()(dict1,dict2)
