from context import Meta

meta = Meta()

@meta("subtract two numbers")
def subtract(x,y):
    return x-y

subtract(3,4)
