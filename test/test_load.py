from context import Meta
from collections import Counter
import numpy

meta = Meta(overhead=True)

@meta("add two numbers")
def add_numbers(x,y):
    return x + y

@meta("count words in a text document", imports=[{"from":"collections", "import":"Counter"}])
def count_words(doc):
    words = [w.lower() for w in doc.split()]
    return Counter(words)

@meta("compute difference in average between lists")
def average_list_difference(l1,l2):
    l1_avg, l2_avg = [numpy.average(x) for x in [l1,l2]]
    return l1_avg - l2_avg

average_list_difference([1,2,3],[3,4,5])

print(average_list_difference.__meta_imports__)

count_words("this is a test document")

print("Meta id: {}".format(add_numbers.__meta_id__))
print("Meta id: {}".format(count_words.__meta_id__))

reload_add_numbers = meta.load(add_numbers.__meta_id__)

assert(reload_add_numbers(4,5) == add_numbers(4,5))

print(add_numbers.analytics())

fibonacci = meta.search("fibonacci")

print("fibonacci(3) = {}".format(fibonacci(3)))

for _ in range(0,1000):
    fibonacci(22)

print("duplicates", fibonacci.find_duplicates())

try: fibonacci("s")
except: pass

print("bugs",fibonacci.bugs())

create_dict_from_object = meta.load("http://meta-lang.org/snippet/56ee4528c0cb8f7470fbafa6")

print(create_dict_from_object(meta))

print(meta.overhead_call)
print(meta.overhead_load)

meta.measure_overhead()
