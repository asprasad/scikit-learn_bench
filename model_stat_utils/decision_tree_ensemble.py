import statistics
from enum import Enum
from collections import namedtuple

class NodeType(Enum):
    Numerical = 1
    Categorical = 2

class TreeNode:
    def __init__(self, nodeType, threshold, featureIndex) -> None:
        self.type = nodeType
        self.threshold = threshold
        self.featureIndex = featureIndex
        self.leftChild = -1
        self.rightChild = -1
        self.parent = -1
        self.depth = -1

    def IsRoot(self) -> bool:
        return self.parent == -1
    
    def IsLeaf(self) -> bool:
        return self.rightChild == -1 and self.leftChild == -1

class Tree:
    def __init__(self, id, num_nodes, num_features) -> None:
        self.root = None
        self.id = id
        self.nodes = []
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_leaves = 0
        self.leaves = []
        self.max_depth = -1
        self.min_depth = -1
        self.skew = -1
    
    def AddNode(self, node):
        if node.leftChild == -1 and node.rightChild == -1:
            self.leaves.append(node)
            self.num_leaves += 1
        self.nodes.append(node)
    
    def GetNode(self, index) -> TreeNode:
        return self.nodes[index]

    def NumberOfNodes(self):
        assert self.num_nodes == len(self.nodes)
        return self.num_nodes

class Feature:
    def __init__(self, name, type, index) -> None:
        self.name = name
        self.type = type
        self.index = index

# 1. Tree depth stats
#       - min depth, max depth, "skew", mean depth, median depth
#       - Per tree stats
# 2. Num features used in each tree
# 3. # times each feature is used in each tree
# 4. # times each feature is used in a path to a leaf/ # features used in each path
TreeStats = namedtuple("TreeStats", ["num_trees", "min_size", "max_size", "median_size",\
    "mean_size", "min_leaves", \
    "max_leaves", "median_leaves", "mean_leaves"])

class Ensemble:
    
    def __init__(self) -> None:
        self.trees = []
        self.features = []
    
    def AddFeature(self, name, type):
        index = len(self.features)
        self.features.append(Feature(name, type, index))
        return index
    
    def AddTree(self, tree) -> None:
        self.trees.append(tree)
    
    def ComputeTreeSizeStatistics(self):
        sizes = [tree.NumberOfNodes() for tree in self.trees]
        leaves = [tree.num_leaves for tree in self.trees]
        return TreeStats(len(self.trees), min(sizes), max(sizes), statistics.median(sizes),\
                statistics.mean(sizes), \
                min(leaves), max(leaves), statistics.median(leaves), statistics.mean(leaves))
    
    

