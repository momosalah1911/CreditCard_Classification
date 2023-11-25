from credit_fraud_utils_data import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from credit_fraud_utils_eval import *
from sklearn.model_selection import RandomizedSearchCV
from credit_fraud_train import LRegression, RandomForest, Voting_classifier, eXtreme_Gradient_Boosting, model_NN


def train_and_eval_model_without_dealing_impalance_data_and_without_tuning(y_train, X_train, y_test, X_test):
    # train different models on train and evalute by train and validation  dataset

    # logistic regression
    model_LR = LRegression(X_train, y_train)
    f1_LR, ap_LR, roc_LR, acc_LR = evaluation(model_LR, X_train, y_train)
    f1_LR_test, ap_LR_test, roc_LR_test, acc_LR_test = test(
        model_LR, X_test, y_test)

    # Random forest
    model_RF = RandomForest(X_train, y_train)
    f1_RF, ap_RF, roc_RF, acc_RF = evaluation(model_RF, X_train, y_train)
    f1_RF_test, ap_RF_test, roc_RF_test, acc_RF_test = test(
        model_RF,  X_test, y_test)

    # Voting Classifier
    model_vc = Voting_classifier(X_train, y_train)
    f1_vc, ap_vc, roc_vc, acc_vc = evaluation(model_vc, X_train, y_train)
    f1_vc_test, ap_vc_test, roc_vc_test, acc_vc_test = test(
        model_vc,  X_test, y_test)

    # XGB
    model_xgb = eXtreme_Gradient_Boosting(X_train, y_train)
    f1_xgb, ap_xgb, roc_xgb, acc_xgb = evaluation(model_xgb, X_train, y_train)
    f1_xgb_test, ap_xgb_test, roc_xgb_test, acc_xgb_test = test(
        model_xgb,  X_test, y_test)

    # simple neural network
    model_nn = model_NN(X_train, y_train)
    f1_nn, ap_nn, roc_nn, acc_nn = evaluation(model_nn, X_train, y_train)
    f1_nn_test, ap_nn_test, roc_nn_test, acc_nn_test = test(
        model_nn, X_test, y_test)

    # show result in dataframe

    result_train = {
        'models': ['LR', 'RF', 'XGB', 'ANN', 'VC'],
        'f1_score': [f1_LR, f1_RF, f1_xgb, f1_nn, f1_vc],
        'average percesion': [ap_LR, ap_RF, ap_xgb, ap_nn, ap_vc],
        'ROC': [roc_LR, roc_RF, roc_xgb, roc_nn, roc_vc],
        'accuracy': [acc_LR, acc_RF, acc_xgb, acc_nn, acc_vc]
    }
    result_test = {
        'models': ['LR', 'RF', 'XGB', 'ANN', 'VC'],
        'f1_score': [f1_LR_test, f1_RF_test, f1_xgb_test, f1_nn_test, f1_vc_test],
        'average percesion': [ap_LR_test, ap_RF_test, ap_xgb_test, ap_nn_test, ap_vc_test],
        'ROC': [roc_LR_test, roc_RF_test, roc_xgb_test, roc_nn_test, roc_vc_test],
        'accuracy': [acc_LR_test, acc_RF_test, acc_xgb_test, acc_nn_test, acc_vc_test]
    }
    result_train = pd.DataFrame(result_train)
    result_test = pd.DataFrame(result_test)
    print(result_train)
    print(result_test)

    sns.barplot(data=result_train, x='models', y='f1_score')
    plt.show()
    sns.barplot(data=result_test, x='models', y='f1_score')
    plt.show()

    return model_LR, model_RF, model_vc, model_xgb, model_nn, result_train, result_test


def different_technique_to_deal_with_impalance_data(df, X_train, y_train, X_val, y_val, factor, enstmaitor, feature='Class'):
    # try different technique with models to show effect of it
    '''
    NOTE:
    Avoid Data Leakage: Apply under-sampling, over-sampling, or SMOTE techniques only to the training set.
    This helps prevent data leakage and ensures that the test set remains representative of real-world scenarios.

    Evaluate Carefully: After applying these techniques, carefully evaluate your model's performance on both the training and test sets.
    Consider metrics like precision, recall, F1-score, and area under the ROC curve (AUC-ROC) to assess performance.
    '''

    # list of data after apply differet technique on it
    common_techniques_tarin = [(X_train, y_train), over_sampling_monority(df, X_train, y_train, feature, factor), under_and_over_sampling(
        df, X_train, y_train, feature), under_sampling_major(df, X_train, y_train, feature, factor)]

    tec = {'technique': ['None', 'over sampling', 'STOME',
                         'under sampling '], 'f1_score_train': [], 'f1_score_val': []}

    # loop to train model each time on data from specific technique
    for tech_train in common_techniques_tarin:
        X_train, y_train = tech_train
        model = enstmaitor(X_train, y_train)

        f1_train, ap_train, roc_train, acc_train = evaluation(
            model, X_train, y_train)

        f1_val, ap_val, roc_val, acc_val = test(model, X_val, y_val)
        tec['f1_score_train'].append(f1_train)
        tec['f1_score_val'].append(f1_val)
    scores = pd.DataFrame(tec)

    print(scores)


# tuning different model by using Random Search with cross validation to select best parameters

def tuning_LR(X_train, y_train, X_test, y_test):

    lr = LogisticRegression(solver='newton-cholesky')
    weights = np.linspace(start=0.0, stop=0.99, num=100)
    parameter = {'class_weight': [
        {0: x, 1: 1-x} for x in weights]}
    gridsearch = RandomizedSearchCV(
        estimator=lr, param_distributions=parameter, scoring='f1', cv=5, return_train_score=False, random_state=240).fit(X_train, y_train)
    print(gridsearch.best_score_)
    f1_LR, ap_LR, roc_LR, acc_LR = evaluation(
        gridsearch.best_estimator_, X_train, y_train)
    f1_LR_test, ap_LR_test, roc_LR_test, acc_LR_test = test(
        gridsearch.best_estimator_, X_test, y_test)
    result_test = {
        'state': ['Train', 'Test'],
        'f1_score': [f1_LR, f1_LR_test, ],
        'average percesion': [ap_LR, ap_LR_test,],
        'ROC': [roc_LR, roc_LR_test,],
        'accuracy': [acc_LR, acc_LR_test,]
    }
    print('best_parameters  : ', gridsearch.best_params_)
    print(pd.DataFrame(result_test))

    return gridsearch.best_estimator_


def Tuning_xgb(X_train, y_train, X_test, y_test):

    xgb_model = xgb.XGBClassifier()
    param_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [30, 50, 75, 100, 200, 300, 350],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    }
    gridsearch = RandomizedSearchCV(
        estimator=xgb_model, param_distributions=param_grid, scoring='f1', cv=5, return_train_score=False, random_state=240).fit(X_train, y_train)
    f1_LR, ap_LR, roc_LR, acc_LR = evaluation(
        gridsearch.best_estimator_, X_train, y_train)
    f1_LR_test, ap_LR_test, roc_LR_test, acc_LR_test = test(
        gridsearch.best_estimator_, X_test, y_test)
    result_test = {
        'state': ['Train', 'Test'],
        'f1_score': [f1_LR, f1_LR_test, ],
        'average percesion': [ap_LR, ap_LR_test,],
        'ROC': [roc_LR, roc_LR_test,],
        'accuracy': [acc_LR, acc_LR_test,]
    }
    print('best_parameters  : ', gridsearch.best_params_)
    print(pd.DataFrame(result_test))
    return gridsearch.best_estimator_


def Tuning_RandomForest(X_train, y_train, X_test, y_test):

    rf = RandomForestClassifier(
        min_samples_split=6, n_estimators=30, )
    rf.fit(X_train, y_train)
    f1_LR, ap_LR, roc_LR, acc_LR = evaluation(
        rf, X_train, y_train)
    f1_LR_test, ap_LR_test, roc_LR_test, acc_LR_test = test(
        rf, X_test, y_test)
    result_test = {
        'state': ['Train', 'Test'],
        'f1_score': [f1_LR, f1_LR_test, ],
        'average percesion': [ap_LR, ap_LR_test,],
        'ROC': [roc_LR, roc_LR_test,],
        'accuracy': [acc_LR, acc_LR_test,]
    }
    print(pd.DataFrame(result_test))
    return rf


def tuning_NN(X_train, y_train, X_test, y_test):

    nn = MLPClassifier(activation='relu')
    parameter_space = {
        'hidden_layer_sizes': [(20, 10, 5), (15, 5), (20, 5)],
        'max_iter': [300, 400, 500, 600, 700],
    }
    gridsearch = RandomizedSearchCV(
        estimator=nn, param_distributions=parameter_space, scoring='f1', cv=5, return_train_score=False, random_state=240).fit(X_train, y_train)
    f1_LR, ap_LR, roc_LR, acc_LR = evaluation(
        gridsearch.best_estimator_, X_train, y_train)
    f1_LR_test, ap_LR_test, roc_LR_test, acc_LR_test = test(
        gridsearch.best_estimator_, X_test, y_test)

    result_test = {
        'state': ['Train', 'Test'],
        'f1_score': [f1_LR, f1_LR_test, ],
        'average percesion': [ap_LR, ap_LR_test,],
        'ROC': [roc_LR, roc_LR_test,],
        'accuracy': [acc_LR, acc_LR_test,]
    }
    print('best_parameters  : ', gridsearch.best_params_)
    print(pd.DataFrame(result_test))
    return gridsearch.best_estimator_
