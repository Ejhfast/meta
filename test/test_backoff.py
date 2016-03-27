# AUTO_PATCH=True python test_backoff.py

from context import Meta

meta = Meta()

convert_string_to_float = meta.load("http://www.meta-lang.org/snippets/56ee44fcc0cb8f7470fbaf21")

# here's a bug
print(convert_string_to_float("ethan"))
