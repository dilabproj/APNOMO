import os
import numpy as np
import pandas as pd
import utils
from utils import *
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

def early_stop_prob_dynamic(prediction, label, args, train_prediction=-1, train_label=-1):
    """
    train_prediction : (cut x N1) prediction results(probability of anomaly) of the training data in all cut points.
    train_label : (N1) true label of the trainng data.
    prediction : (cut x N2) prediction results(probability of anomaly) of the test data in all cut points.
    label : (N2) true label of the test data.
    cut : how many cut do we slice the whole time series data.
    """

    cut = prediction.shape[0]
    print("\n\n\n\nStarting with early stopping dynamic...")
    ####Find all threshold
    flatten_sorted_prediction = np.sort( train_prediction, axis=None)
    threshold_list = [  (flatten_sorted_prediction[i]+flatten_sorted_prediction[i+1]) / 2.  for i in range( len(flatten_sorted_prediction)-1 ) ]
    print("Total threshold candicate: {}".format(len(threshold_list)))
    threshold_list = np.unique( np.round( np.asarray(threshold_list), 4 ) )
    print("Total unique threshold candicate: {}".format(len(threshold_list)))

    ##################
    print("Now counting the performance for each threshold")
    multi_objectives = None
    test_multi_objectives = None
    using_ojective = None
    list_ = []
    list_test = []
    for threshold in tqdm(threshold_list):
        # # --- Run model on train data ---
        train_predictions = []
        train_locations = []
        N_train = train_prediction.shape[1]
        for i in range(N_train):
            for j in range(cut):
                if j+1 == cut and train_prediction[j][i] < threshold:
                    train_predictions.append( 0 )
                    train_locations.append( float(j+1)/cut )
                elif train_prediction[j][i] >= threshold or j+1 == cut:
                    train_predictions.append( 1 )
                    train_locations.append( float(j+1)/cut )
                    break
                else:
                    continue

        y_pred_list = np.asarray(train_predictions)
        earliness = np.asarray(train_locations)

        dic_result = count_result(train_label, y_pred_list, earliness, threshold=threshold, show_result=False)

        list_.append(dic_result) # all objectives of each threshold

        multi_objective, uo = get_multi_objective(dic_result, args=args)
        #multi_objective: target objectives of each threshold, uo: name of each objective
        if multi_objectives is None:
            multi_objectives = multi_objective
            using_ojective = uo
        else:
            multi_objectives = np.concatenate((multi_objectives, multi_objective), 0)

        # --- Run model on test data ---
        test_predictions = []
        test_locations = []
        N_test = prediction.shape[1]
        for i in range(N_test):
            for j in range(cut):
                true_count = 0
                if j+1 == cut:
                    if prediction[j][i] < threshold:
                        test_predictions.append( 0 )
                    else:
                        test_predictions.append( 1 )
                    test_locations.append( float(j+1)/cut )
                elif prediction[j][i] >= threshold or j+1 == cut:
                    true_count += 1
                    if true_count == 1:
                        test_predictions.append( 1 )
                        test_locations.append( float(j+1)/cut )
                        break
                else:
                    continue

        y_pred_list = np.asarray(test_predictions)
        earliness = np.asarray(test_locations)

        dic_result = count_result(label, y_pred_list, earliness, threshold=threshold, show_result=False)

        list_test.append(dic_result)
        test_multi_objective, _ = get_multi_objective(dic_result, args=args)
        if test_multi_objectives is None:
            test_multi_objectives = test_multi_objective
        else:
            test_multi_objectives = np.concatenate((test_multi_objectives, test_multi_objective), 0)


    ## Now do the Multi-objective optimization to find the knee.
    print("Now finding the best threshold with Multi-objective optimization")
    df_train = pd.DataFrame( list_ )
    df_test = pd.DataFrame( list_test )

    knee, threshold_final = multi_objective_optimization(multi_objectives, test_multi_objectives, df_train, df_test, args,
        show_pareto_front=False, show_plot=True, save_result=True, note="_lr_dynamic_fea", using_ojective=using_ojective)

    return threshold_final
