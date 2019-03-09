import json
import ast
import re

def print_tree(root, indent=0):
  print(' ' * indent, root)
  if len(root.children) > 0:
    for child in root.children:
      print_tree(child, indent+4)

def myfind(s, char):
  pos = s.find(char)
  if pos == -1: # not found
      return len(s) + 1
  else: 
      return pos

def next_tok(s): # returns tok, rest_s
  if s == '': 
    return (None, None)
  # normal cases
  poss = [myfind(s, ' '), myfind(s, '['), myfind(s, ']')]
  min_pos = min(poss)
  if poss[0] == min_pos: # separator is a space
    tok, rest_s = s[ : min_pos], s[min_pos+1 : ] # skip the space
    if tok == '': # more than 1 space
      return next_tok(rest_s)
    else:
      return (tok, rest_s)
  else: # separator is a [ or ]
    tok, rest_s = s[ : min_pos], s[min_pos : ]
    if tok == '': # the next char is [ or ]
      return (rest_s[:1], rest_s[1:])
    else:
      return (tok, rest_s)
        
def str_to_tokens(str_tree):
  # remove \n first
  str_tree = str_tree.replace('\n','')
  out = []
  
  tok, s = next_tok(str_tree)
  while tok is not None:
    out.append(tok)
    tok, s = next_tok(s)
  return out

# format: node, list-of-children
str_tree = '''
1 [2 [3 4       5          ] 
   6 [7 8 [9]   10 [11 12] ] 
   13
  ]
'''
toks = str_to_tokens(str_tree)
print(toks)



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



def make_tree(tokens):
  if len(tokens) == 0:
    return None
  elif tokens[0] is not '[' and tokens[0] is not ']':
    return Tree(tokens[0], make_tree(tokens[1:]))
  else:
    return make_tree(tokens[1:])


def num_nodes(tree):
  total = 1
  if len(tree.children) > 0:
    for child in tree.children:
      total += num_nodes(child)
  return total

def max_depth(tree):
  # max of the size of the left and right subtree
  


t = Tree('*', [Tree('1'),
               Tree('2'),
               Tree('+', [Tree('3'),
                          Tree('4')])])

print('Number of nodes:', num_nodes(t))
# print_tree(make_tree(toks))


# s = 0
# m = 0
# for t in toks:
#   if t == '[':
#     s += 1
#   elif t == ']':
#     s -= 1

#   if s >= m:
#     m = s


# a = []
# for i in range(m+1):
#   a.append([])


# s = 0
# for t in toks:
#   if t == '[':
#     s += 1
#   elif t == ']':
#     s -= 1
#   else:
#     a[s].append(t)

 
# print(a)