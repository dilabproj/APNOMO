import numpy as np
import os
import torch
from sklearn.metrics import accuracy_score, classification_report
#Imbalance Handle
from sklearn.utils import resample,shuffle
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN 
from collections import Counter
import random
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import time

def sortXY(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    Y2 = Y* -1
    idx = np.lexsort((X, Y2))
    return X[idx], Y[idx]

def get_multi_objective(dic_result, args, v2=-1):
    if_using_objective = {  #0:False, 1:True
        "threshold":        0,
        "accuracy":         0,
        "precision":        0,
        "recall":           0,
        "f1-score":         1,
        "Earliness":        0,
        "Earliness-1":      0,
        "Earliness-T":      1,
        }

    try:
        if len(args.moo)==7:
            for idx, key in enumerate(if_using_objective):
                if idx>0 and int(args.moo[idx-1]):
                    if_using_objective[key] = 1
                else:
                    if_using_objective[key] = 0
    except:
        pass

    objective_coeef = {
        "threshold":        0.,
        "accuracy":         -1.,
        "precision":        -1.,
        "recall":           -1.,
        "f1-score":         -1.,
        "Earliness":        1.,
        "Earliness-1":      1.,
        "Earliness-T":      1.,
    }
    list_ = []
    using_objective = []
    for key, value in if_using_objective.items():
        if value:
            list_.append(dic_result[key]*objective_coeef[key])
            using_objective.append(key)

    multi_objective = np.asarray(list_).reshape(1,-1)

    # np.array([[ dic_result["f1-score"]*-1. , dic_result["Earliness-T"]*1. ]])
    return multi_objective, using_objective


def find_pareto_front(F):
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    fronts, rank = NonDominatedSorting().do(F, return_rank=True)
    return fronts

def knee_mdd_selection(front_idx, non_dominated_objectives):

    obj_max_array = []
    obj_min_array = []
    Lm = []

    dist = []

    for i in range(non_dominated_objectives.shape[0]):
        dist.append(0)

    for m in range(non_dominated_objectives.shape[1]):
        obj_max_array.append(np.max(non_dominated_objectives[:, m]))
        obj_min_array.append(np.min(non_dominated_objectives[:, m]))

        Lm.append(obj_max_array[m]-obj_min_array[m])

        for i, obj_A in enumerate(non_dominated_objectives):
            dist[i] = dist[i] + \
                ((obj_A[m] - obj_min_array[m]) / Lm[m])  # Lm[m]

            #print('idl:',i,'obj: ',m,'dist: ',( (obj_A[m] - obj_min_array[m]) / Lm[m] ))

    kneeval = np.max(dist)
    kneeidx = np.argmax(dist)
    for idx, val in enumerate(dist):
        if(val == 1):
            continue
        if(val < kneeval):
            kneeval = val
            kneeidx = idx
    print("original knee idx = {}".format(kneeidx))
    return front_idx[kneeidx]

def multi_objective_optimization(multi_objectives, test_multi_objectives, df_train, df_test, args, 
                                    show_pareto_front=True, show_knee=True, show_plot=True,
                                    save_result=False,  note="", using_ojective=None):
    ###
    # multi_objectives: N x k, N results with k objectives, the only variable used to find the pareto front and knee
    ###Following variable are used to show the results
    # test_multi_objectives: N x k, N results with k objectives, only used to show the result
    # df_train: N x K, N result with all K metrics, only used to show the result
    # df_test: N x K, N result with all K metrics, only used to show the result
    # show_pareto_front: set it to be True when you want to show all the pareto front data (train and test)
    # show_knee: set it to be True when you want to show all the knee data (train and test)
    # show_plot: set it to be True when you want to plot all the (pareto front, knee) data (train and test)
    ###

    ####
    print("Start finding pareto optimal")

    front = find_pareto_front(multi_objectives)
    # print("Pareto front index = {}".format(front[0]))

    knee = knee_mdd_selection(front[0], multi_objectives[front[0]])
    print("Knee index = {}".format(knee))

    print("Using objectives are {}".format(using_ojective))

    if show_pareto_front:
        ### Result in training
        print("\nPareto Front in training set")
        print(df_train.iloc[front[0]])
        ### Result in testing
        print("Pareto Front in test set")
        print(df_test.iloc[front[0]])

    if show_knee:
        print("\nKnee in training set")
        # print(df_train.iloc[knee])
        for x in df_train:
            if x == 'recall':
                print("{}:\t\t {}%".format(x,np.round(df_train.iloc[knee][x]*100., 2)))
            else:
                print("{}:\t {}%".format(x,np.round(df_train.iloc[knee][x]*100., 2)))
        print("Knee in test set")
        # print(df_test.iloc[knee])
        for x in df_test:
            if x == 'recall':
                print("{}:\t\t {}%".format(x,np.round(df_test.iloc[knee][x]*100., 2)))
            else:
                print("{}:\t {}%".format(x,np.round(df_test.iloc[knee][x]*100., 2)))

    if show_plot and len(using_ojective) == 2:
        #Plot the result
        plot_X, plot_Y= sortXY(multi_objectives[front[0],0],multi_objectives[front[0],1])
        import matplotlib.pyplot as plt
        plt.scatter(multi_objectives[:,0],multi_objectives[:,1], c='b', zorder=1)
        plt.plot(plot_X, plot_Y, c='k', zorder=2)
        plt.scatter(multi_objectives[front[0],0],multi_objectives[front[0],1], c='g', zorder=3)
        plt.scatter(multi_objectives[knee,0],multi_objectives[knee,1], c='r', zorder=4)
        
        plt.xlabel(using_ojective[0])
        plt.ylabel(using_ojective[1])
        # plt.show()
        if not os.path.exists( 'plot' ):
            os.mkdir( 'plot' )
        try:
            plt.savefig('plot/{}_{}_nhid{}_k{}_M{}_N{}_MOO_{}_pareto_front_train.png'.format(args.dataset,args.type,args.nhid,args.k,args.N_me,args.N_neighbor,args.moo))
        except:
            plt.savefig('plot/{}_nhid{}_{}_pareto_front_train.png'.format(args.dataset,args.type,args.nhid))
        plt.close()

        
        #Plot the result
        import matplotlib.pyplot as plt
        plot_X, plot_Y = sortXY(test_multi_objectives[front[0],0],test_multi_objectives[front[0],1])
        plt.scatter(test_multi_objectives[:,0],test_multi_objectives[:,1], c='b', zorder=1)
        plt.plot(plot_X, plot_Y, c='k', zorder=2)
        plt.scatter(test_multi_objectives[front[0],0],test_multi_objectives[front[0],1], c='g', zorder=3)
        plt.scatter(test_multi_objectives[knee,0],test_multi_objectives[knee,1], c='r', zorder=4)

        plt.xlabel(using_ojective[0])
        plt.ylabel(using_ojective[1])
        # plt.show()
        try:
            plt.savefig('plot/{}_{}_nhid{}_k{}_M{}_N{}_MOO_{}_pareto_front_test.png'.format(args.dataset,args.type,args.nhid,args.k,args.N_me,args.N_neighbor,args.moo))
        except:
            plt.savefig('plot/{}_nhid{}_{}_pareto_front_test.png'.format(args.dataset,args.type,args.nhid))
        plt.close()

    if save_result:
        if not os.path.exists( 'results' ):
            os.mkdir( 'results' )
        try:
            if args.save:
                file_name_this_time = 'results/{}_{}_nhid{}_MOO_{}_0607.csv'.format(args.dataset,args.type,args.nhid,0)
                file_name_this_time_train = 'results/{}_{}_nhid{}_MOO_{}__0607train.csv'.format(args.dataset,args.type,args.nhid,0)
                end = time.time()
                total_time = end-args.start
                df_test['exec_time'] = total_time
                df_test['K'] = args.k
                df_test['N_me'] = args.N_me
                df_test['N_neighbor'] = args.N_neighbor
                df_test['moo'] = args.moo
                df_train['exec_time'] = total_time
                df_train['K'] = args.k
                df_train['N_me'] = args.N_me
                df_train['N_neighbor'] = args.N_neighbor
                df_train['moo'] = args.moo

                print("Total time : {}".format(total_time))

                if os.path.exists(file_name_this_time):
                    df_test.iloc[knee:knee+1].to_csv(file_name_this_time, mode='a', index=False, header=False)
                    df_train.iloc[knee:knee+1].to_csv(file_name_this_time_train, mode='a', index=False, header=False)
                else:
                    df_test.iloc[knee:knee+1].to_csv(file_name_this_time, index=False)
                    df_train.iloc[knee:knee+1].to_csv(file_name_this_time_train, index=False)
            else:
                print("Not to save the results to the csv file.")
        except:
            df_test.iloc[knee:knee+1].to_csv('results/{}_{}_{}_moo_0607.csv'.format(args.dataset,args.type,args.nhid))

    return knee, df_test.iloc[knee]['threshold']


def count_result(label, prediction, earliness=None, train=False, threshold=None, show_result=True):
    if train:
        front = "train_"
    else:
        front = ""

    accuracy = np.round(accuracy_score(label, prediction), 4)
    if earliness is not None:
        ano_earliness = earliness[np.where(prediction==1)].tolist()
        earliness_value = np.mean(earliness)
        earliness_1 = np.mean(ano_earliness)
        earliness_T = np.mean(np.array(earliness)[(np.array(prediction)==1)&(np.array(label)==1)])
    dic = classification_report(label, prediction,digits=4, output_dict=True)
    precision = np.round(dic['1.0']['precision'], 4)
    recall = np.round(dic['1.0']['recall'], 4)
    f1 = np.round(dic['1.0']['f1-score'], 4)

    if show_result:
        print("{}Accuracy: \t{}%".format(front, str(np.round(100.*accuracy, 4))))
        print("1-Precision: \t"+str(np.round(100.*precision, 4))+'%')
        print("1-Recall: \t"+str(np.round(100.*recall, 4))+'%')
        print("F1-score: \t"+str(np.round(100.*f1, 4))+'%')
        if earliness is not None:
            print("Earliness: \t{}%".format(np.round(100.*earliness_value, 2)))
            print('Earliness-1: \t'+str(np.round(100.*earliness_1, 2))+'%')
            print('Earliness-T: \t'+str(np.round(100.*earliness_T, 2))+'%')

    if earliness is not None:
        dic_result = {  "threshold":threshold,
                        "accuracy":accuracy,
                        "precision":precision,
                        "recall":recall,
                        "f1-score":f1,
                        "Earliness":earliness_value,
                        "Earliness-1":earliness_1,
                        "Earliness-T":earliness_T}
    else:
        dic_result = {  "threshold":threshold,
                        "accuracy":accuracy,
                        "precision":precision,
                        "recall":recall,
                        "f1-score":f1,
                    }

    return dic_result

def neighbor_over_sampling(train_all, train_y, idx, args, N_neighbor=1, N_me=1):
    # train_all = set(nseg) x n x nhid
    # train_y = n
    # idx = which index you want to target on
    # N_neighbor = how many neighbors do you want, k --> i-k ~ i+k 
    
    print("\nStart doing neighbor-over-sampling with case {}".format(idx))
    ## Determine neighbor idx
    nseg = train_all.shape[0]
    neighbor = [idx for _ in range(N_me)]
    
    #Original
    for k in range(1, N_neighbor+1):
        upper = idx + k
        lower = idx - k
        if upper >= nseg:
            upper = nseg - 1
        if lower < 0:
            lower = 0

        neighbor.append( upper )
        neighbor.append( lower )

    neighbor = np.asarray( neighbor ).reshape(-1) # (2*k+1)

    #Without duplicating me
    neighbor = np.unique(neighbor)

    ###Show the neighbor_list
    print("Neighbors are {}".format(neighbor))

    print("Original Shape of train_all is {}".format(train_all.shape))
    print("Original Shape of train_y is {}".format(train_y.shape))

    for i, candicate in enumerate(neighbor):
        if i == 0:
            new_train = train_all[candicate] #N x nhid
            new_train_y = train_y #N
        else: #Add the anomaly data from neighbor
            new_train = np.concatenate( (new_train, train_all[candicate][ train_y==1 ]) )
            new_train_y = np.concatenate( (new_train_y, train_y[ train_y==1 ]) )

    print("After neighbor-over-sampling, Shape of train_all is {}".format(new_train.shape))
    print("After neighbor-over-sampling, Shape of train_y is {}".format(new_train_y.shape))

    return new_train, new_train_y

def count_each_prob(y_pred_list, y_true_list, prob=0.5): #N x set_
    set_ = y_pred_list.shape[1]
    for i in range(set_):
        print("="*20+"Earliness on "+str((float(i)+1)/set_))
        temp = np.zeros(len(y_pred_list))
        temp[np.argwhere(y_pred_list[:,i]>=prob)] = 1
        preds = temp
        dic = classification_report(y_true_list, preds,digits=4, output_dict=True)
        print("Accuracy\t"+str(np.round(accuracy_score(y_true_list, preds), 4)))
        print("1-Precision\t"+str(np.round(dic['1.0']['precision'], 4)))
        print("1-Recall\t"+str(np.round(dic['1.0']['recall'], 4)))
        print("F1-score\t"+str(np.round(dic['1.0']['f1-score'], 4)))
    return

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)