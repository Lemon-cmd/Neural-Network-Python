from collections import defaultdict, namedtuple 
from heapq import heappop, heappush
from random import randrange, shuffle 

Connection = namedtuple('Connection', 'accuracy epoch range')

class Heap(object):
    # min heap
    def __init__(self):
        self._values = []
        self._size = 0 

    def push(self, value):
        heappush(self._values, value)
        self._size += 1

    def pop(self):
        assert self._size != 0
        self._size -= 1
        return heappop(self._values)

    def __len__(self):
        return self._size

    def display(self):
        print("DISPLAYING")
        print(self._values)
        print(" ")

def batch_gen(size, batch_size):
    heap = Heap()
    ran_indexes = set([0, batch_size, size - batch_size])

    for j in range(size//(batch_size - 3)):
        ran_indexes.add(randrange(batch_size - 1, size - batch_size))
        ran_indexes.add(randrange(1, batch_size))
    
    e = 0
    for index in ran_indexes:
        heap.push(Connection(accuracy = 0, epoch=e, range = (index, index + batch_size)))
        e += 1
        
    return heap 

