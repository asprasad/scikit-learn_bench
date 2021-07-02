from os import stat
import statistics
from enum import Enum
from collections import namedtuple

class NodeType(Enum):
    Numerical = 1
    Categorical = 2

def AggregateListOfLists(lst):
    agg = []
    for l in lst:
        if len(l) > len(agg):
            n = len(l) - len(agg)
            agg = agg + [0] * n
        for i in range(len(l)):
            agg[i] += l[i]
    return agg

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
    
    def NumberOfFeaturesInPathToRoot(self):
        if self.IsRoot():
            return 1
        features = []
        node = self
        while not node.IsRoot():
            if not node.IsLeaf():
                features.append(node.featureIndex)
            node = node.tree.GetNode(node.parent)
        return len(set(features))
    
    def FeatureUsesInPathToRoot(self):
        assert self.IsLeaf()
        node = self
        featureUses = dict()
        while not node.IsRoot():
            if not node.IsLeaf():
                if not node.featureIndex in featureUses.keys():
                    featureUses[node.featureIndex] = 1
                else:
                    featureUses[node.featureIndex] += 1 
            node = node.tree.GetNode(node.parent)
        return sorted(featureUses.values(), reverse=True)


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

    def NumberOfFeatures(self):
        nonLeaves = [n for n in self.nodes if n not in self.leaves]
        featureIdxs = [n.featureIndex for n in nonLeaves]
        featureSet = set(featureIdxs)
        return len(featureSet)
    
    def FeatureUses(self):
        nonLeaves = [n for n in self.nodes if n not in self.leaves]
        featureUses = dict()
        for node in nonLeaves:
            if not node.featureIndex in featureUses.keys():
                featureUses[node.featureIndex] = 1
                continue
            featureUses[node.featureIndex] += 1
        return featureUses

    def NumberOfFeaturesUsedToLeaves(self):
        return [n.NumberOfFeaturesInPathToRoot() for n in self.leaves]
    

    def SortedAggregateFeaturesUsesOnPath(self):
        featureUsesOnPaths = [leaf.FeatureUsesInPathToRoot() for leaf in self.leaves]
        return AggregateListOfLists(featureUsesOnPaths)

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
#   - Sort for each tree and then aggregate so that the most used feature per tree are aggregated together
# 4. # times each feature is used in a path to a leaf/ # features used in each path
ListStats = namedtuple("ListStats", ["min", "max", "median", "mean", "std_dev"])
TreeStats = namedtuple("TreeStats", ["num_trees", "num_features", "tree_size_stats", \
                                     "leaves_stats", "skew_stats", "leaf_depth_stats", \
                                     "feature_set", "features_on_paths", "feature_uses", "sorted_feature_uses_trees", "sorted_feature_uses_paths"])

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
    
    def AggregateFeatureUses(self):
        featureUses = [tree.FeatureUses() for tree in self.trees]
        aggregateFeatureUses = dict()
        aggregateSortedUses = []        
        for featureUse in featureUses:
            sortedUses = sorted(featureUse.values(), reverse=True)
            if len(sortedUses) > len(aggregateSortedUses):
                n = len(sortedUses) - len(aggregateSortedUses)
                aggregateSortedUses = aggregateSortedUses + [0] * n
            for i in range(len(sortedUses)):
                aggregateSortedUses[i] += sortedUses[i]

            for featureIdx, uses in featureUse.items():
                if not featureIdx in aggregateFeatureUses.keys():
                    aggregateFeatureUses[featureIdx] = uses
                    continue
                aggregateFeatureUses[featureIdx] += uses
        return aggregateFeatureUses, aggregateSortedUses

    def AggregateSortedFeatureUsesOnPath(self):
        featureUses = [t.SortedAggregateFeaturesUsesOnPath() for t in self.trees]
        return AggregateListOfLists(featureUses)

    def ComputeTreeSizeStatistics(self):
        sizes = [tree.NumberOfNodes() for tree in self.trees]
        leaves = [tree.num_leaves for tree in self.trees]
        skews = [tree.Skew() for tree in self.trees]
        leaf_depths = [d for tree in self.trees for d in tree.LeafDepths()]
        numFeatures = [tree.NumberOfFeatures() for tree in self.trees]
        featuresOnPath = [d for tree in self.trees for d in tree.NumberOfFeaturesUsedToLeaves()]
        aggregateFeatureUses, aggregateSortedUses = self.AggregateFeatureUses()
        sortedFeatureUsesOnPathSToRoot = self.AggregateSortedFeatureUsesOnPath()
        return TreeStats(len(self.trees), len(self.features), ComputeListStats(sizes), ComputeListStats(leaves), \
                         ComputeListStats(skews), ComputeListStats(leaf_depths), ComputeListStats(numFeatures), \
                         ComputeListStats(featuresOnPath), aggregateFeatureUses, aggregateSortedUses, sortedFeatureUsesOnPathSToRoot)
    
    

