from credit_fraud_utils_data import *
from sklearn.metrics import classification_report, f1_score, average_precision_score, roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve

import pickle
# Load models from the pickle file
with open('models_for_testing.pkl', 'rb') as file:
    loaded_models = pickle.load(file)


# Best Model
Xgb = loaded_models['xgboost']['model']
Xgb_threshold = loaded_models['xgboost']['threshold']
scale = loaded_models['scaler']


def test_evaluate(model, X, y, threshold):
    y_probs = model.predict_proba(X)
    y_pre = np.where(y_probs >= threshold, 1, 0)
    f1 = f1_score(y_true=y, y_pred=y_pre[:, 1])
    ap = average_precision_score(y_true=y, y_score=y_pre[:, 1])
    roc = roc_auc_score(y_true=y, y_score=y_pre[:, 1])
    acc = accuracy_score(y_true=y, y_pred=y_pre[:, 1])
    print('f1_score :', f1, '\n', 'average precision score :', ap,
          '\n', 'roc :', roc, '\n', 'Area Under Carve :', acc)
    print(confusion_matrix(y, y_pre[:, 1]))

    return f1, ap, roc, acc


def test(url=r"D:\Data Science\course_ML_mostafa_saad\projects\2 Credit Card Fraud Detection\data\split\val.csv", model=Xgb):
    df_test = load_dataset(url)
    df_test = check_for_nan_and_doublicate(df_test)
    y_test, X_test = split_to_data_target(df_test)
    X_test = scale.transform(X_test)
    test_evaluate(model, X_test, y_test,  Xgb_threshold)


if __name__ == '__main__':
    test()
