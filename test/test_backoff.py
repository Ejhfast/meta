# AUTO_PATCH=True python test_backoff.py
import collections
import math
from context import Meta

meta = Meta(globals=globals())

# @meta("get most frequent character from string")
# def get_most_frequent_character(string):
#     if string == "": return None
#     return collections.Counter(string).most_common(1)[0]
#
# get_most_frequent_character("ethan fast")
#
# get_most_frequent_character.get_dynamic_type_sig()

# @meta("extract all digits from a string")
# def extract_digits(n):
# 	return [int(x) for x in str(n) if x in "0123456789"]
#
# extract_digits("123 45")
#
# extract_digits.get_dynamic_type_sig()

# @meta("calculate log of each element in array")
# def log_array(a):
# 	return [math.log(x) for x in a]
#
# @meta("calculate log of each element in array, with error as zero")
# def log_array_safe(a):
# 	return [math.log(x) if x > 0 else 0 for x in a]
#
# log_array([2,3,4])
# log_array_safe([6,8,12])
#
# log_array_safe.get_dynamic_type_sig()
#
# print(log_array.find_duplicates())

log_array = meta.load("http://www.meta-lang.org/snippets/570a91def842a50003be6280")

log_array([1,2,0,4])

get_most_frequent_character = meta.load("http://www.meta-lang.org/snippets/56ee4681c0cb8f7470fbb317")

get_most_frequent_character("")

convert_string_to_int = meta.load("http://www.meta-lang.org/snippets/56ee44fcc0cb8f7470fbaf23")

convert_string_to_int("abc") # patch with safe string to int

convert_string_to_float = meta.load("http://www.meta-lang.org/snippets/56ee44fcc0cb8f7470fbaf21")

# here's a bug
print(convert_string_to_float("ethan")) # patch with safe string to float
