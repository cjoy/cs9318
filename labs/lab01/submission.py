## import modules here 
import math

################# Question 0 #################

def add(a, b): # do not change the heading of the function
    return a + b


################# Question 1 #################

def nsqrt(x): # do not change the heading of the function
    for i in range(1, x):
        square = i*i
        if square > x:
            return i-1
        elif square == x:
            return i
    return ValueError('out of range')


################# Question 2 #################


# x_0: initial guess
# EPSILON: stop when abs(x - x_new) < EPSILON
# MAX_ITER: maximum number of iterations

## NOTE: you must use the default values of the above parameters, do not change them

def find_root(f, fprime, x_0=1.0, EPSILON = 1E-7, MAX_ITER = 1000): # do not change the heading of the function
    x = x_0
    for i in range(MAX_ITER-1):
        x_new = x - (f(x)/fprime(x))
        if abs(x - x_new) < EPSILON: break
        x = x_new
    return x

################# Question 3 #################

class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

def make_tree(tokens):  # do not change the heading of the function
  root = Tree(tokens[0])
  parent = child = root
  children = []
  for token in tokens[1:]:
    if token is '[':
      children.append(parent)
      parent = child
    elif token is ']':
      parent = children.pop()
    else: 
      child = Tree(token)
      parent.add_child(child)
  return root

def max_depth(root):  # do not change the heading of the function
  if root.children is None or len(root.children) is 0:
    return 1
  depths = [1]
  for child in root.children:
    depths.append(1 + max_depth(child))
  return max(depths)
