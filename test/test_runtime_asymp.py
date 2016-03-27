from context import Meta

meta = Meta()

@meta("convert bool to int")
def bool_to_int(b):
    if b: return 1
    return 0

test = ["True" for _ in range(0,1000)]

# this should not take forever
for t in test:
    bool_to_int(t)
