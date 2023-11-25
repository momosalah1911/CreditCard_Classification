# CreditCard_Classification
![Credit Card Fraud](https://static.vecteezy.com/system/resources/previews/001/883/786/large_2x/set-scenes-hacker-with-laptop-and-credit-card-during-covid-19-pandemic-free-vector.jpg)

*Objective: 
  -Detect fraudulent credit card transactions in a dataset containing transactions by European cardholders in September 2013.

*Dataset Overview:
   -Highly unbalanced class distribution: 0.172% frauds out of 284,807 transactions.
   -Features are numerical, resulting from PCA, with 'Time' and 'Amount' as non-transformed features.
   -'Class' is the response variable (1 for fraud, 0 otherwise).

## Conclusion
* dataset does not have missing value
* dataset have doublicated value (448 instance is doublicated in train dataset 8 in class fraud )
* 0.2 % from data belongs to Class Fraud (impalance dataset)
* Amount:
  -have alot of outliers (18823 object is outliers)
  -From this scene, it can be concluded that there are many outliers, but 
   getting rid of them is not the best solution. Rather, it is possible to 
   obtain some event through them to help the model predict. Let us discover 
   that.
  -very skewness
*Time:
  -convert it from second to hour
*V(1->28):
  -V17,V14 and V12 have high negative skew (less than -0.25)
  -features have large number of outliers you can drop it or not but let try to show data after drop it and after it well be decide if drop it or not
  -number of rows after befor outliers (170436, 33)
  -number of rows after remove outliers (72175, 33)
  -number of rows are removed 98261
  -we lost 42.35 % from data
  -number of rows in class NO Faurd after remove outliers 72166
  -number of rows in class Fraud after remove outliers 9
*most important features :

![most important features](https://github.com/momosalah1911/CreditCard_Classification/assets/113562712/7ab83273-f598-410c-bc56-f16fe1ff8c4c)

### Modiling Trics:
 *when apply model without using any technique to deal with impalance dataset 
 * Result with Train dataset
 | Model | F1 Score | Average Precision | ROC AUC | Accuracy |
|-------|----------|-------------------|---------|----------|
| LR    | 0.672    | 0.482             | 0.769   | 0.999    |
| RF    | 1.000    | 1.000             | 1.000   | 1.000    |
| XGB   | 1.000    | 1.000             | 1.000   | 1.000    |
| ANN   | 0.796    | 0.635             | 0.880   | 0.999    |
| VC    | 0.700    | 0.540             | 0.769   | 0.999    |

* Result with Validation dataset

| Model | F1 Score | Average Precision | ROC AUC | Accuracy |
|-------|----------|-------------------|---------|----------|
| LR    | 0.667    | 0.471             | 0.770   | 0.999    |
| RF    | 0.845    | 0.722             | 0.882   | 0.999    |
| XGB   | 0.861    | 0.753             | 0.882   | 0.999    |
| ANN   | 0.786    | 0.620             | 0.871   | 0.999    |
| VC    | 0.672    | 0.496             | 0.758   | 0.999    |

*try to check result of multiple model to deal with dataset (implance dataset)without tuning this model 
> [!NOTE]
> Avoid Data Leakage: Apply under-sampling, over-sampling, or SMOTE techniques only to the training set.
    This helps prevent data leakage and ensures that the test set remains representative of real-world scenarios.

> Evaluate Carefully: After applying these techniques, carefully evaluate your model's performance on both the training and test sets.
    Consider metrics like precision, recall, F1-score, and area under the ROC curve (AUC-ROC) to assess performance.

 * Result(F1_score) with Train dataset 
| Model               | None   | Over Sampling | STOME | Under Sampling |
|---------------------|--------|---------------|-------|----------------|
| Logistic Regression | 0.672  | 0.915         | 0.943 | 0.886          |
| Neural Network      | 0.796  | 0.981         | 0.977 | 0.918          |
| XGBoost             | 1.000  | 1.000         | 1.000 | 1.000          |
| RandomForest        | 0.998  | 1.000         | 1.000 | 1.000          |
| Voting Classifier    | 0.717  | 0.943         | 0.976 | 0.900          |


* Result(F1_score) with Validation dataset

| Model               | None   | Over Sampling | STOME | Under Sampling |
|---------------------|--------|---------------|-------|----------------|
| Logistic Regression | 0.667  | 0.221         | 0.097 | 0.783          |
| Neural Network      | 0.786  | 0.170         | 0.154 | 0.254          |
| XGBoost             | 0.861  | 0.839         | 0.789 | 0.224          |
| RandomForest        | 0.854  | 0.840         | 0.830 | 0.298          |
| Voting Classifier    | 0.718  | 0.833         | 0.832 | 0.724          |

> [!IMPORTANT]
>Performance improved on training data using different methods, but failed to generalize to test data

*try to tuning different model using grid search with cross validation and can also using thresholds

 1- logistic regression 
    best_parameters  :  {'class_weight': {0: 0.3, 1: 0.7}}
      state   f1_score  average percesion    ROC   accuracy
    |--------|--------|-------------------|-------|---------|
    |  Train | 0.686071  |         0.498963 | 0.777722 | 0.999114|
    |   Test | 0.693878   |        0.504542 | 0.786455  |0.999209|
     best threshold for train : 0.10023508103303037 
     best_f1_score_train 0.8014311270125224
     f1 score for test with threshold from average best_threshold_test and 
     best_threshold_train  0.8023952095808383
    
 2-XGB
 best_parameters  :  {'subsample': 0.8, 'n_estimators': 350, 'max_depth': 2, 'learning_rate': 0.1, 'colsample_bytree': 0.6}
    | State | F1 Score | Average Precision | ROC AUC | Accuracy |
|-------|----------|-------------------|---------|----------|
| Train | 0.904    | 0.822             | 0.919   | 0.999    |
| Test  | 0.873    | 0.767             | 0.904   | 0.999    |
    best threshold for train : 0.28970808 
    best_f1_score_train 0.9281961471103328
    f1 score for test with threshold from best_threshold_train and 
    best_threshold_train  0.874251497005988

  3-Neural Network
      best_parameters  :  {'max_iter': 500, 'hidden_layer_sizes': (20, 10, 5)}

  | State | F1 Score | Average Precision | ROC AUC | Accuracy |
|-------|----------|-------------------|---------|----------|
| Train | 0.812    | 0.667             | 0.867   | 0.999    |
| Test  | 0.805    | 0.653             | 0.871   | 0.999    |
best threshold for train : 0.8651446871097714 
    best_f1_score_train 0.8245931283905967
    f1 score for test with threshold from best_threshold_train and best_threshold_train  0.7926829268292682

  4- RAndom Forest
  | State | F1 Score | Average Precision | ROC AUC | Accuracy |
|-------|----------|-------------------|---------|----------|
| Train | 0.951    | 0.906             | 0.956   | 0.999    |
| Test  | 0.870    | 0.765             | 0.893   | 0.999    |

Best Threshold for Train: 0.339
best threshold for train : 0.33920634920634923 
best_f1_score_train 0.9816971713810316


 



  
