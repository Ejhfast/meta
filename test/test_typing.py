from context import Meta

meta = Meta()

merge_dictionaries = meta.load("http://www.meta-lang.org/snippets/56ee468ac0cb8f7470fbb338")

print(merge_dictionaries.get_dynamic_type_sig())
