import os
import xgboost 
import json
from pprint import pprint
from decision_tree_ensemble import Tree, Ensemble, TreeNode

scriptPath = os.path.realpath(__file__)
scriptDirPath = os.path.dirname(scriptPath)
rootPath = os.path.dirname(scriptDirPath)
modelFileDir = os.path.join(rootPath, "xgb_models")

def ReadModelJSONFile(filename):
    modelJSON = None
    with open(filename, "r") as f:
        modelJSON = json.load(f)
    return modelJSON

def ProcessSingleXGBTree(treeJSON):
    # TODO what is "base_weights", "categories", "categories_nodes", 
    # "categories_segments", "categories_sizes"?
    num_nodes = len(treeJSON["base_weights"])
    # TODO ignoring "default_left"
    tree_id = treeJSON["id"]
    
    left_children = treeJSON["left_children"]
    right_childen = treeJSON["right_children"]
    parents = treeJSON["parents"]
    split_conditions = treeJSON["split_conditions"]
    split_indices = treeJSON["split_indices"]
    split_type = treeJSON["split_type"] # 0 is Numerical and 1 is Categorical
    num_features = int(treeJSON["tree_param"]["num_feature"])
    num_nodes = int(treeJSON["tree_param"]["num_nodes"])
    assert len(left_children) == num_nodes
    assert len(left_children) == len(right_childen) and\
           len(left_children) == len(parents)
    
    tree = Tree(tree_id, num_nodes, num_features)
    for i in range(num_nodes):
        assert split_type[i] == 0 # only numerical splits for now
        node = TreeNode(split_type[i], split_conditions[i], split_indices[i])
        node.leftChild = left_children[i]
        node.rightChild = right_childen[i]
        node.parent = parents[i] if not parents[i] == 2147483647 else -1
        tree.AddNode(node)
    return tree

def ConstructTreesFromXGBBooster(boosterJSON, ensemble):
    modelJSON = boosterJSON["model"]
    num_trees = modelJSON["gbtree_model_param"]["num_trees"]
    treesJSON = modelJSON["trees"]
    for treeJSON in treesJSON:
        tree = ProcessSingleXGBTree(treeJSON)
        ensemble.AddTree(tree)
    return

def ConstructTreeEnsembleFromXGB(xgboostJSON):
    ensemble = Ensemble()

    learnerJSON = xgboostJSON["learner"]
    # Read feature details
    feature_names = learnerJSON["feature_names"]
    feature_types = learnerJSON["feature_types"]
    assert len(feature_names) == len(feature_types)
    for i in range(len(feature_types)):
        ensemble.AddFeature(feature_names[i], feature_types[i])
    
    ConstructTreesFromXGBBooster(learnerJSON["gradient_booster"], ensemble)
    return ensemble


# Args : Model filename, model format (XGBoost, LightGBM)
filename = os.path.join(modelFileDir, "year_prediction_msd_xgb_model_save.json")
modelJSON = ReadModelJSONFile(filename)
ensemble = ConstructTreeEnsembleFromXGB(modelJSON)
stats = ensemble.ComputeTreeSizeStatistics()
print(stats)

