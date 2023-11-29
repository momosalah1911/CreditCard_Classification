from credit_fraud_utils_data import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from credit_fraud_utils_eval import *
from Credit_Fraud_train_tuning_models import *
import pickle


def LRegression(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr


def RandomForest(X, y):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    return rf


def Voting_classifier(X, y):
    lr = LogisticRegression(
        solver='newton-cg', )
    rf = RandomForestClassifier()
    classifier = [('lr', lr), ('rf', rf)]

    vc = VotingClassifier(estimators=classifier)
    vc.fit(X, y)
    return vc


def eXtreme_Gradient_Boosting(X, y):
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X, y)
    return xgb_model


def model_NN(X, y):
    regr = MLPClassifier(random_state=240, max_iter=500,
                         activation='relu', hidden_layer_sizes=(20, 5))
    regr.fit(X, y)

    return regr


if __name__ == '__main__':
    np.random.seed(240)
    # load all datasets
    # train dataset
    df_train = load_dataset()
    # Val dataset
    df_test = load_dataset(
        r"D:\Data Science\course_ML_mostafa_saad\projects\2 Credit Card Fraud Detection\data\split\val.csv")
    # TrainVal dataset
    df_ValTest = load_dataset(
        r"D:\Data Science\course_ML_mostafa_saad\projects\2 Credit Card Fraud Detection\data\split\trainval.csv")
    # check for doublicates and null
    df_train = check_for_nan_and_doublicate(df_train)
    df_test = check_for_nan_and_doublicate(df_test)
    df_ValTest = check_for_nan_and_doublicate(df_ValTest)
    # split to target and data
    y_train, X_train = split_to_data_target(df_train)
    y_test, X_test = split_to_data_target(df_test)
    y_ValTest, X_ValTest = split_to_data_target(df_ValTest)
    # normalized data
    X_train, scale = normalized_data(X_train)
    X_test = scale.transform(X_test)
    X_ValTest = scale.transform(X_ValTest)
    '''
    try to check result of multiple model to deal with dataset (implance dataset)
    without tuning this mode or deal with impalance data by over sampling ore under sampling or both  
    '''
    # train_and_eval_model_without_dealing_impalance_data_and_without_tuning(
    #     y_train, X_train, y_test, X_test)
    '''
    result in train of it is :
    models  f1_score  average percesion       ROC  accuracy
0     LR  0.672269           0.482342  0.769304  0.999085
1     RF  1.000000           1.000000  1.000000  1.000000
2    XGB  1.000000           1.000000  1.000000  1.000000
3    ANN  0.795775           0.635004  0.880339  0.999319
4     VC  0.700219           0.539524  0.769360  0.999196

result in test :

 models  f1_score  average percesion       ROC  accuracy
0     LR  0.666667           0.471405  0.769601  0.999156
1     RF  0.844720           0.721967  0.881987  0.999561
2    XGB  0.860759           0.753341  0.882014  0.999613
3    ANN  0.785714           0.619946  0.870672  0.999367
4     VC  0.671533           0.496074  0.758409  0.999209

    '''
#################################################################################
#################################################################################
#################################################################################

    '''
    try to check result of multiple model to deal with dataset (implance dataset)
    without tuning this model 
    but deal with different techniques (over sampling , under samoling ,STOME ) 
    '''
    # x = LRegression
    # print('LRegression')
    # different_technique_to_deal_with_impalance_data(
    #     df_train, X_train, y_train, X_test, y_test, factor=3, feature='Class', enstmaitor=x)
    '''
    Logistic Regression with different techniques
    technique  f1_score_train  f1_score_val
    0             None        0.672269      0.666667
    1    over sampling        0.914596      0.220930
    2            STOME        0.942827      0.097111
    3  under sampling         0.885553      0.782609

    '''

    # x2 = model_NN
    # print('model_NN')
    # different_technique_to_deal_with_impalance_data(
    #     df_train, X_train, y_train, X_test, y_test, factor=3, feature='Class', enstmaitor=x2)
    '''
    technique  f1_score_train  f1_score_val
    0             None        0.795775      0.785714
    1    over sampling        0.981327      0.169643
    2            STOME        0.977459      0.154000
    3  under sampling         0.917857      0.254486

    '''
    # x3 = eXtreme_Gradient_Boosting
    # print('eXtreme_Gradient_Boosting')
    # different_technique_to_deal_with_impalance_data(
    #     df_train, X_train, y_train, X_test, y_test, factor=3, feature='Class', enstmaitor=x3)
    '''
    eXtreme_Gradient_Boosting
            technique  f1_score_train  f1_score_val
    0             None             1.0      0.860759
    1    over sampling             1.0      0.839080
    2            STOME             1.0      0.789474
    3  under sampling              1.0      0.224090
    '''
    # x4 = RandomForest
    # print('RandomForest')
    # different_technique_to_deal_with_impalance_data(
    #     df_train, X_train, y_train, X_test, y_test, factor=3, feature='Class', enstmaitor=x4)

    '''
    RandomForest
            technique  f1_score_train  f1_score_val
    0             None        0.998314      0.853503
    1    over sampling        1.000000      0.840237
    2            STOME        1.000000      0.830409
    3  under sampling         1.000000      0.298279

    '''
    # x5 = Voting_classifier
    # print('Voting_classifier')
    # different_technique_to_deal_with_impalance_data(
    #     df_train, X_train, y_train, X_test, y_test, factor=3, feature='Class', enstmaitor=x5)
    '''
    Voting_classifier
            technique  f1_score_train  f1_score_val
    0             None        0.717063      0.718310
    1    over sampling        0.943499      0.833333
    2            STOME        0.975887      0.832370
    3  under sampling         0.900000      0.723618

    '''
#################################################################################
#################################################################################
#################################################################################
    '''
    let's try to tuning Logistic model and show results without different techniques (over sampling , under samoling ,STOME ) 
    '''
    # print('LR')
    # model = tuning_LR(X_train, y_train, X_test, y_test)
    # model.fit(X_train, y_train)
    # y_prob = model.predict_proba(X_train)
    # best_threshold_train, best_f1_score_train = precision_recall_curve_different_thresholds(
    #     y_train, y_prob[:, 1])
    # print('best threshold for train :', best_threshold_train,
    #       '\n', 'best_f1_score_train', best_f1_score_train)
    # y_prob_test = model.predict_proba(X_test)

    # f1 = apply_threshold(y_prob=y_prob_test[:, 1], y_true=y_test,
    #                      threshold=best_threshold_train)
    # print('f1 score for test with threshold from  best_threshold_train and best_threshold_train ', f1)
    '''
    best_parameters  :  {'class_weight': {0: 0.3, 1: 0.7}}
    state  f1_score  average percesion       ROC  accuracy
    0  Train  0.686071           0.498963  0.777722  0.999114
    1   Test  0.693878           0.504542  0.786455  0.999209
    best threshold for train : 0.10023508103303037 
   
    best f1 score test 0.8275862068965517
    f1 score for test with threshold from  best_threshold_train   0.8023952095808383
    '''
#################################################################################
#################################################################################
#################################################################################
    '''
    let's try to tuning XGB model and show results without different techniques (over sampling , under samoling ,STOME ) 
    # '''
    # print('xgb')
    # model = Tuning_xgb(X_train, y_train, X_test, y_test)
    # model.fit(X_train, y_train)
    # y_prob = model.predict_proba(X_train)
    # y_prob_test = model.predict_proba(X_test)
    # best_threshold_train, best_f1_score_train = precision_recall_curve_different_thresholds(
    #     y_train, y_prob[:, 1])
    # print('best threshold for train :', best_threshold_train,
    #       '\n', 'best_f1_score_train', best_f1_score_train)

    # f1 = apply_threshold(y_prob=y_prob_test[:, 1], y_true=y_test,
    #                      threshold=best_threshold_train)

    # print('f1 score for test with threshold from  best_threshold_train  ', f1)
    '''
    xgb
    best_parameters  :  {'subsample': 0.8, 'n_estimators': 350, 'max_depth': 2, 'learning_rate': 0.1, 'colsample_bytree': 0.6}
    state  f1_score  average percesion       ROC  accuracy
    0  Train  0.903811           0.822162  0.919177  0.999689
    1   Test  0.872727           0.766709  0.904459  0.999631
    best threshold for train : 0.28970808 
    best_f1_score_train 0.9281961471103328
    f1 score for test with threshold from  best_threshold_train   0.874251497005988

    '''
#################################################################################
#################################################################################
#################################################################################
    '''
    let's try to tuning  NN model and show results without different techniques (over sampling , under samoling ,STOME ) 
    '''
    # print('NN')
    # model = tuning_NN(X_train, y_train, X_test, y_test)
    # model.fit(X_train, y_train)
    # y_prob = model.predict_proba(X_train)
    # y_prob_test = model.predict_proba(X_test)
    # best_threshold_train, best_f1_score_train = precision_recall_curve_different_thresholds(
    #     y_train, y_prob[:, 1])
    # print('best threshold for train :', best_threshold_train,
    #       '\n', 'best_f1_score_train', best_f1_score_train)

    # f1 = apply_threshold(y_prob=y_prob_test[:, 1], y_true=y_test,
    #                      threshold=best_threshold_train)

    # print('f1 score for test with threshold from  best_threshold_train  ', f1)
    '''
    NN
    best_parameters  :  {'max_iter': 500, 'hidden_layer_sizes': (20, 10, 5)}
    state  f1_score  average percesion       ROC  accuracy
    0  Train  0.811918           0.667186  0.866939  0.999407
    1   Test  0.804878           0.652989  0.870707  0.999438
    best threshold for train : 0.8651446871097714 
    best_f1_score_train 0.8245931283905967
    f1 score for test with threshold from best_threshold_train   0.7926829268292682

    '''
#################################################################################
#################################################################################
#################################################################################
    '''
    let's try to tuning  Random forest model and show results without different techniques (over sampling , under samoling ,STOME ) 
    '''

    # print('RF')
    # model = Tuning_RandomForest(X_train, y_train, X_test, y_test)
    # model.fit(X_train, y_train)
    # y_prob = model.predict_proba(X_train)
    # y_prob_test = model.predict_proba(X_test)
    # best_threshold_train, best_f1_score_train = precision_recall_curve_different_thresholds(
    #     y_train, y_prob[:, 1])
    # print('best threshold for train :', best_threshold_train,
    #       '\n', 'best_f1_score_train', best_f1_score_train)

    # f1 = apply_threshold(y_prob=y_prob_test[:, 1], y_true=y_test,
    #                      threshold=best_threshold_train)

    # print('f1 score for test with threshold from  best_threshold_train  ', f1)
    '''
    Random forest 
    parameters => (min_samples_split=6, n_estimators=30, )
    state  f1_score  average percesion       ROC  accuracy
    0  Train  0.950877           0.905926  0.956223  0.999836
    1   Test  0.869565           0.765003  0.893241  0.999631
    best threshold for train : 0.33920634920634923 
    best_f1_score_train 0.9816971713810316
    f1 score for test with threshold from  best_threshold_train   0.874251497005988
    '''
#################################################################################
#################################################################################
#################################################################################
    '''
    let's try to tuning  Random forest model and show results without different techniques (over sampling , under samoling ,STOME ) 
    '''

    # print('Voting')
    # rf = RandomForestClassifier(min_samples_split=6, n_estimators=30)
    # xgb = xgb.XGBClassifier(subsample=0.8, n_estimators=350,
    #                         max_depth=2, learning_rate=0.1, colsample_bytree=0.6)
    # classifier = [('lr', xgb), ('rf', rf)]

    # vc = VotingClassifier(estimators=classifier, voting='soft')

    # vc.fit(X_train, y_train)
    # y_prob = vc.predict_proba(X_train)
    # y_prob_test = vc.predict_proba(X_test)
    # best_threshold_train, best_f1_score_train = precision_recall_curve_different_thresholds(
    #     y_train, y_prob[:, 1])
    # print('best threshold for train :', best_threshold_train,
    #       '\n', 'best_f1_score_train', best_f1_score_train)

    # f1 = apply_threshold(y_prob=y_prob_test[:, 1], y_true=y_test,
    #                      threshold=best_threshold_train)

    # print('f1 score for test with threshold from  best_threshold_train  ', f1)
    '''
   Voting
   best threshold for train : 0.20669759883560124 
   best_f1_score_train 0.9768976897689768
   f1 score for test with threshold from  best_threshold_train and best_threshold_train  0.8720930232558138
    '''
#################################################################################
#################################################################################
#################################################################################
    '''
    Save Model is pkl file to send to test 
    Step 0 : concate train and val and train val
    Step 1 : Create and train your best machine learning models 
    Step 2 : Define any additional information you want to save, such as thresholds
    Step 3 : Store models and related information in a dictionary
    Step 4 : Save the dictionary to a pickle file for testing

    '''
    # Step 0 : concate train and val and train val

    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)

    # Step 1 : Create and train your best machine learning models

    '''
    # best XGB
    - best_parameters  :  {'subsample': 0.8, 'n_estimators': 350, 'max_depth': 2, 'learning_rate': 0.1, 'colsample_bytree': 0.6}
    - best threshold for train : 0.28970808
    '''
    Xgb = xgb.XGBClassifier(subsample=0.8, n_estimators=350,
                            max_depth=2, learning_rate=0.1, colsample_bytree=0.6)
    Xgb.fit(X, Y)
    Xgb_threshold = 0.28970808

    # # Step 3: Store models and related information in a dictionary
    models_dict = {
        # XGBoost may not have a threshold
        'xgboost': {'model': Xgb, 'threshold': Xgb_threshold},
        'scaler': scale
    }

    # Step 4: Save the dictionary to a pickle file for testing
    output_filename = 'models_for_testing.pkl'
    with open(output_filename, 'wb') as file:
        pickle.dump(models_dict, file)
