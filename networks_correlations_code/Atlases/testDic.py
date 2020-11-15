from networkToIndexDicGordon import dic

total = 0
all_indexes = []
for key,val in dic.items():
	print(key)
	print(len(val))
	all_indexes = all_indexes + val
	total += len(val)
print ("total = ", total)
print(all_indexes)
all_indexes_set = set(all_indexes)
contains_duplicates = len(all_indexes) != len(all_indexes_set)

print(contains_duplicates)