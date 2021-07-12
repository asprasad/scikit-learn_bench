# ===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
RunningInferenceProfiling = True
import os
# This is a hack to make XGBoost run single threaded. 
if RunningInferenceProfiling:
    os.environ['OMP_NUM_THREADS'] = "1"
import sys
scriptPath = os.path.realpath(__file__)
scriptDir = os.path.dirname(scriptPath)
benchDir = os.path.dirname(scriptDir)
sys.path = sys.path + [benchDir]
import argparse

import bench
import numpy as np
import pandas as pd
import xgboost as xgb
import time
import math
# from xgboost import plot_tree
# import matplotlib.pyplot as plt

# These things printed on the screen will cause a test failure. Ignore them when you're running the profiling.
if RunningInferenceProfiling==True:
    print("Starting xgboost benchmark. PID : ", os.getpid())

def convert_probs_to_classes(y_prob):
    return np.array([np.argmax(y_prob[i]) for i in range(y_prob.shape[0])])


def convert_xgb_predictions(y_pred, objective):
    if objective == 'multi:softprob':
        y_pred = convert_probs_to_classes(y_pred)
    elif objective == 'binary:logistic':
        y_pred = y_pred.astype(np.int32)
    return y_pred

logfile = open("xgb_bench_log.txt", "w")

parser = argparse.ArgumentParser(description='xgboost gradient boosted trees benchmark')


parser.add_argument('--colsample-bytree', type=float, default=1,
                    help='Subsample ratio of columns '
                         'when constructing each tree')
parser.add_argument('--count-dmatrix', default=False, action='store_true',
                    help='Count DMatrix creation in time measurements')
parser.add_argument('--enable-experimental-json-serialization', default=True,
                    choices=('True', 'False'), help='Use JSON to store memory snapshots')
parser.add_argument('--grow-policy', type=str, default='depthwise',
                    help='Controls a way new nodes are added to the tree')
parser.add_argument('--inplace-predict', default=False, action='store_true',
                    help='Perform inplace_predict instead of default')
parser.add_argument('--learning-rate', '--eta', type=float, default=0.3,
                    help='Step size shrinkage used in update '
                         'to prevents overfitting')
parser.add_argument('--max-bin', type=int, default=256,
                    help='Maximum number of discrete bins to '
                         'bucket continuous features')
parser.add_argument('--max-delta-step', type=float, default=0,
                    help='Maximum delta step we allow each leaf output to be')
parser.add_argument('--max-depth', type=int, default=6,
                    help='Maximum depth of a tree')
parser.add_argument('--max-leaves', type=int, default=0,
                    help='Maximum number of nodes to be added')
parser.add_argument('--min-child-weight', type=float, default=1,
                    help='Minimum sum of instance weight needed in a child')
parser.add_argument('--min-split-loss', '--gamma', type=float, default=0,
                    help='Minimum loss reduction required to make'
                         ' partition on a leaf node')
parser.add_argument('--n-estimators', type=int, default=100,
                    help='The number of gradient boosted trees')
parser.add_argument('--objective', type=str, required=True,
                    choices=('reg:squarederror', 'binary:logistic',
                             'multi:softmax', 'multi:softprob'),
                    help='Specifies the learning task')
parser.add_argument('--reg-alpha', type=float, default=0,
                    help='L1 regularization term on weights')
parser.add_argument('--reg-lambda', type=float, default=1,
                    help='L2 regularization term on weights')
parser.add_argument('--scale-pos-weight', type=float, default=1,
                    help='Controls a balance of positive and negative weights')
parser.add_argument('--single-precision-histogram', default=False, action='store_true',
                    help='Build histograms instead of double precision')
parser.add_argument('--subsample', type=float, default=1,
                    help='Subsample ratio of the training instances')
parser.add_argument('--tree-method', type=str, required=True,
                    help='The tree construction algorithm used in XGBoost')

params = bench.parse_args(parser)
# Default seed
if params.seed == 12345:
    params.seed = 0

model_file_path = 'xgb_models/{dataset_name}_xgb_model_save.json'.format(dataset_name = params.dataset_name)

# Load and convert data
X_train, X_test, y_train, y_test = bench.load_data(params)

# print("Done loading test data...")
logfile.write("X_train shape : {shape}\n".format(shape = X_train.shape))
# print("X_train has inf : ", np.isinf(X_train).any().any())
logfile.write("y_train shape : {shape}\n".format(shape = y_train.shape))
logfile.write("X_test shape : {shape}\n".format(shape = X_test.shape))
logfile.write("y_test shape : {shape}\n".format(shape = y_test.shape))

xgb_params = {
    'booster': 'gbtree',
    'verbosity': 0,
    'learning_rate': params.learning_rate,
    'min_split_loss': params.min_split_loss,
    'max_depth': params.max_depth,
    'min_child_weight': params.min_child_weight,
    'max_delta_step': params.max_delta_step,
    'subsample': params.subsample,
    'sampling_method': 'uniform',
    'colsample_bytree': params.colsample_bytree,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'reg_lambda': params.reg_lambda,
    'reg_alpha': params.reg_alpha,
    'tree_method': params.tree_method,
    'scale_pos_weight': params.scale_pos_weight,
    'grow_policy': params.grow_policy,
    'max_leaves': params.max_leaves,
    'max_bin': params.max_bin,
    'objective': params.objective,
    'seed': params.seed,
    'single_precision_histogram': params.single_precision_histogram,
    'enable_experimental_json_serialization':
        params.enable_experimental_json_serialization
}

if params.threads != -1:
    xgb_params.update({'nthread': params.threads})

if params.objective.startswith('reg'):
    task = 'regression'
    metric_name, metric_func = 'rmse', bench.rmse_score
else:
    task = 'classification'
    metric_name = 'accuracy'
    metric_func = bench.accuracy_score
    if 'cudf' in str(type(y_train)):
        params.n_classes = y_train[y_train.columns[0]].nunique()
    else:
        params.n_classes = len(np.unique(y_train))

    # Covtype has one class more than there is in train
    if params.dataset_name == 'covtype':
        params.n_classes += 1

    if params.n_classes > 2:
        xgb_params['num_class'] = params.n_classes

dtrain = xgb.DMatrix(X_train, y_train)
# print("Done creating training DMatrix. Shape : ", dtrain.num_row(), dtrain.num_col())
dtest = xgb.DMatrix(X_test, y_test)
# print("Done creating test DMatrix. Shape : ", dtest.num_row(), dtrain.num_col())

def RepeatDF(df, n):
    newdf = pd.DataFrame(np.repeat(df.values, n, axis=0))
    newdf.columns = df.columns
    return newdf

def RepeatSeries(s, n):
    newSeries = pd.Series([])
    newSeries.append([s]*n, ignore_index=True)
    return newSeries

def fit(dmatrix):
    if dmatrix is None:
        dmatrix = xgb.DMatrix(X_train, y_train)
    logfile.write("Number of trees : {n_estimators} \n".format(n_estimators=params.n_estimators))
    return xgb.train(xgb_params, dmatrix, params.n_estimators)

if params.inplace_predict:
    def predict(*args):
        return booster.inplace_predict(np.ascontiguousarray(X_test.values,
                                                            dtype=np.float32))
else:
    def predict(dmatrix):  # type: ignore
        if dmatrix is None:
            dmatrix = xgb.DMatrix(X_test, y_test)
        return booster.predict(dmatrix)

def ComputeTestSetParamsForProfiling(dmatrix):
    print("Testing inference time on ", dmatrix.num_row(), " rows")
    start = time.time()
    booster.predict(dmatrix)
    end = time.time()
    predictionTime = end - start
    # Some small benchmarks seem to be much faster on subsequent runs!
    if predictionTime < 10:
        start = time.time()
        booster.predict(dmatrix)
        end = time.time()
        predictionTime = end - start

    print("Estimated prediction time for X_train : ", predictionTime)
    minPredictionTime = 0.5 # compute how many times to replicate so each call takes 500ms
    numReps = math.ceil(minPredictionTime/predictionTime)
    numReps = 250 if numReps>250 else numReps
    # limit the number of rows we end up creating
    numRows = dmatrix.num_row() * numReps
    rowLimit = 2000000
    if numRows > rowLimit:
        numReps = math.ceil(rowLimit/dmatrix.num_row())
    return numReps*predictionTime, numReps


if RunningInferenceProfiling == True:
    params.box_filter_measurements = 1

noTraining = os.path.exists(model_file_path) and RunningInferenceProfiling

booster = None
if not noTraining:
    logfile.write("Running training {times} times.\n".format(times=params.box_filter_measurements))
    fit_time, booster = bench.measure_function_time(
        fit, None if params.count_dmatrix else dtrain, params=params)
    train_metric = metric_func(
        convert_xgb_predictions(
            booster.predict(dtrain),
            params.objective),
        y_train)
    booster.save_model(model_file_path)
else:
    print("Skipping training. Loading model from ", model_file_path)
    booster = xgb.Booster(model_file=model_file_path)
    fit_time = 0
    train_metric = 0
    print("Done loading model")


# logfile.write("Booster best ntree limit : " + str(booster.best_ntree_limit) + "\n")
logfile.write("Running inference {times} times.\n".format(times=params.box_filter_measurements))

if RunningInferenceProfiling==True:
    print("Setting n_jobs to 1 on the model to ensure single threaded execution")
    booster.set_param('n_jobs', 1)
    predictionTime, numReps = ComputeTestSetParamsForProfiling(dtrain)
    print("Replicating X_train ", numReps, " times")
    X_test = RepeatDF(X_train, numReps)
    dtest = xgb.DMatrix(X_test)
    # Get about 10s worth of computation 
    params.box_filter_measurements = math.ceil(10/predictionTime)
    print("Running inference on ", dtest.num_row(), " rows and repeating ", params.box_filter_measurements, " times")

    input("Press enter to start inference ...")
    predict_time, y_pred = bench.measure_function_time(predict, dtest, params=params)
    input("Done inferencing ... Press enter to exit.")

    test_metric = -1
    # test_metric = metric_func(convert_xgb_predictions(y_pred, params.objective), y_test)
else:
    predict_time, y_pred = bench.measure_function_time(
    predict, None if params.inplace_predict or params.count_dmatrix else dtest, params=params)
    test_metric = metric_func(convert_xgb_predictions(y_pred, params.objective), y_test)

# plot_tree(booster)
# plt.show()

bench.print_output(library='xgboost', algorithm=f'gradient_boosted_trees_{task}',
                   stages=['training', 'prediction'],
                   params=params, functions=['gbt.fit', 'gbt.predict'],
                   times=[fit_time, predict_time], accuracy_type=metric_name,
                   accuracies=[train_metric, test_metric], data=[X_train, X_test],
                   alg_instance=booster, alg_params=xgb_params)

logfile.close()
