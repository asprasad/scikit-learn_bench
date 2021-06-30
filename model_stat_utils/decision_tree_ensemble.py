from os import stat
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
        self.tree = None

    def IsRoot(self) -> bool:
        return self.parent == -1
    
    def IsLeaf(self) -> bool:
        return self.rightChild == -1 and self.leftChild == -1

    def Depth(self) -> int:
        if not self.depth == -1:
            return self.depth
        depth = 1
        if not self.IsRoot():
            depth = self.tree.GetNode(self.parent).Depth() + 1
        self.depth = depth
        return depth

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
        node.tree = self
        self.nodes.append(node)
    
    def GetNode(self, index) -> TreeNode:
        return self.nodes[index]

    def NumberOfNodes(self):
        assert self.num_nodes == len(self.nodes)
        return self.num_nodes

    def ComputeDepthStats(self):
        depths = [n.Depth() for n in self.leaves]
        self.min_depth = min(depths)
        self.max_depth = max(depths)
        self.skew = self.max_depth - self.min_depth
    
    def Skew(self):
        if self.skew == -1:
           self.ComputeDepthStats()
        return self.skew

    def LeafDepths(self):
        return [n.Depth() for n in self.leaves] 


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
ListStats = namedtuple("ListStats", ["min", "max", "median", "mean", "std_dev"])
TreeStats = namedtuple("TreeStats", ["num_trees", "tree_size_stats", \
                                     "leaves_stats", "skew_stats", "leaf_depth_stats"])

def ComputeListStats(lst):
    return ListStats(min(lst), max(lst), statistics.median(lst), statistics.mean(lst), statistics.stdev(lst))

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
        skews = [tree.Skew() for tree in self.trees]
        leaf_depths = [d for tree in self.trees for d in tree.LeafDepths()]

        return TreeStats(len(self.trees), ComputeListStats(sizes), ComputeListStats(leaves), \
                         ComputeListStats(skews), ComputeListStats(leaf_depths))
    
    

