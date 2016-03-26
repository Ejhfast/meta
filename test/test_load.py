from context import Meta

meta = Meta()

@meta.doc("add two numbers")
def add_numbers(x,y): return x + y

print("Meta id: {}".format(add_numbers.__meta_id__))

reload_add_numbers = meta.load(add_numbers.__meta_id__)

assert(reload_add_numbers(4,5) == add_numbers(4,5))

print(add_numbers.analytics())

fibonacci = meta.search("fibonacci")

print("fibonacci(3) = {}".format(fibonacci(3)))

print("duplicates", fibonacci.find_duplicates())

try: fibonacci("s")
except: pass

print("bugs",fibonacci.bugs())
