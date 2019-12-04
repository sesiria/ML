# BinaryHeap Implementation
# author sesiria 2019
# the interface style is similary to the C++ STL priority_queue

class BinaryHeap:
    def __init__(self, cmp = lambda x, y : x > y):
        """ Default constructor
        Args:
            cmp (lambda or function object) : the default is for MinHeap
        """ 
        self.heap = [None]
        self.cmp = cmp
        self.size = 0

    def length(self):
        """ Return the number of elements of the Binary Heap
        Returns:
            (int)
        """
        return self.size

    def empty(self):
        """ Return whether the Binary Heap is empty
        Returns:
            (bool)
        """
        return self.size == 0

    def top(self):
        """ Return the top element of the Binary Heap
        """
        assert not self.empty()
        return self.heap[1]

    def push(self, item):
        """ Insert the item into the Binary Heap
        Returns:
        """
        # percolate up
        self.heap.append(item)
        self.size += 1
        self._shift_up(self.size)
    
    def pop(self):
        """ Pop the top element of the Binary Heap
        """
        assert not self.empty()
        self.heap[self.size], self.heap[1] = self.heap[1], self.heap[self.size]
        self.size -= 1
        result = self.heap.pop()
        self._shift_down(1)
        return result

    def _shift_up(self, pos):
        """ Percolate up algorithm of binary heap
        """
        while pos // 2 > 0 and self.cmp(self.heap[pos // 2], self.heap[pos]):
            self.heap[pos] , self.heap[pos // 2] = self.heap[pos // 2], self.heap[pos]
            pos //= 2

    def _shift_down(self, pos):
        """ Percolate down algorithm of binary heap
        """
        while pos * 2 <= self.size:
            child = pos * 2
            # whether the current node have the left child.
            if (child != self.size) and self.cmp(self.heap[child], self.heap[child + 1]):
                child += 1
            
            if self.cmp(self.heap[pos], self.heap[child]):
                self.heap[pos] , self.heap[child] = self.heap[child], self.heap[pos]

            pos = child

if __name__ == '__main__':
    # sanity check
    minHeap = BinaryHeap()
    minHeap.push(5)
    minHeap.push(3)
    minHeap.push(7)
    minHeap.push(5)
    while not minHeap.empty():
        print(minHeap.pop())

