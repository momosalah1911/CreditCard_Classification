from sklearn.metrics import classification_report, f1_score, average_precision_score, roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve
from credit_fraud_utils_data import *
import matplotlib.pyplot as plt


# def test(model="LR"):
#     best_model = tuning(
#         model=f'{model}', techniqe='under_and_over_sampling', normalized=1)
#     df_test = pd.read_csv(
#         r"D:\Data Science\course_ML_mostafa_saad\projects\2 Credit Card Fraud Detection\data\split\val.csv")
#     df_test = check_for_nan_and_doublicate(df_test)
#     q9 = df_test['Amount'].quantile(0.9)
#     df_test = add_column_for_huage_amount(df_test, q9)
#     y, X = split_to_data_target(df_test)
#     X, y = under_and_over_sampling(df_test, X, y, "Class", )
#     X = normalized_data(X, 1)
#     predi = best_model.predict(X)
#     print(classification_report(y_true=y, y_pred=predi))


def evaluation(model, X, y):
    y_pre = model.predict(X)
    # print(classification_report(y_true=y, y_pred=y_pre))
    # print(confusion_matrix(y_true=y, y_pred=y_pre))
    f1 = f1_score(y_true=y, y_pred=y_pre)
    ap = average_precision_score(y_true=y, y_score=y_pre)
    roc = roc_auc_score(y_true=y, y_score=y_pre)
    acc = accuracy_score(y_true=y, y_pred=y_pre)
    return f1, ap, roc, acc


def test(model, X, y):
    y_pre = model.predict(X)
    f1 = f1_score(y_true=y, y_pred=y_pre)
    ap = average_precision_score(y_true=y, y_score=y_pre)
    roc = roc_auc_score(y_true=y, y_score=y_pre)
    acc = accuracy_score(y_true=y, y_pred=y_pre)
    return f1, ap, roc, acc


def precision_recall_curve_different_thresholds(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Plot F1 score for different thresholds
    f1_scores = [2 * (p * r) / (p + r) if (p + r) >
                 0 else 0 for p, r in zip(precision[:-1], recall[:-1])]
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()
    best_f1_score = max(f1_scores)
    index_best_f1_score = f1_scores.index(best_f1_score)
    best_threshold = thresholds[index_best_f1_score]
    return best_threshold, best_f1_score


def apply_threshold(threshold, y_true, y_prob):
    y_probs = np.where(y_prob >= threshold, 1, 0)
    f1 = f1_score(y_true=y_true, y_pred=y_probs)
    return f1
