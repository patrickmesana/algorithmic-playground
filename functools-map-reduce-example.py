#%%
import random

num_elements = 100000000  # Adjust this number based on the desired dataset size
data = [random.randint(1, 100) for _ in range(num_elements)]


#%%
counts = {}
for number in data:
    counts[number] = counts.get(number, 0) + 1

print(counts)

#%%
from functools import reduce

# Map phase: Create a tuple (number, 1) for each number
mapped_data = map(lambda number: (number, 1), data)

# Reduce phase: Aggregate the counts for each unique number
def countBy(accumulator, element):
    number, count = element
    accumulator[number] = accumulator.get(number, 0) + count
    return accumulator

reduced_data = reduce(countBy, mapped_data, {})

print(reduced_data)

# %% show histogram
import matplotlib.pyplot as plt

plt.bar(reduced_data.keys(), reduced_data.values())
plt.show()
# %%
