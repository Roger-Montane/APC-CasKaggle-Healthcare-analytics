# Cas Kaggle APC UAB 2021-22
### Nom: Roger Montané Güell
### DATASET: AV - Healthcare Analytics II
### URL: [Kaggle](https://www.kaggle.com/nehaprabhavalkar/av-healthcare-analytics-ii)

## Summary
This dataset contains features related to the patient, hospital and length of stay (LOS) on a case by case basis. Instead of being a continous variable, the LOS is discrtized using ranges of 10 days (0-10, 11-20, ..., 91-100, 100+). We have two csv files:

* **train_data.csv:** contains data for train + validation, where we can check our models' accuracies. This dataset contains 313793 rows with 18 features each.
* **test_data.csv:** does not contain the `Stay` (measure of LOS) varibale, it is used to submit a prediction of your best model. This dataset contains 134865 rows with 17 features each.

From our 18 variables, 6 are deleted since they will not be relevant to make a prediction (more on that topic in the notebook): `case_id`, `patientid`, `Hospital_code`, `City_Code_Hospital`, `City_Code_PAtient` and `Visitors with Patient`.
Of the 12 remaining variables, a 75% of them are categorical and the remaining 25% are numerical. Neither of the numerical variables are normalized.

### Dataset objectives
The goal is to predict the LOS of a patient given certain variables relating to their medical condition, as well as their economic condition and the hospital's type and/or condition. In order to predict the LOS we will use the categorical variable `Stay`, which will be label encoded so that predictions on its value can be performed (more on this topic on the notebook).

## Experiments
To start things off, 3 types of multi-dimensional linear regressors were tested to predict the LOS and, as expected, they performed poorly. After that, 8 ML models were tested with different input data:
* **(1) All variables:** normalized and encoded dataset with all the features
* **(2) Best variables:** normalized and encoded dataset composed of the k best variables
* **(3) Over-sampled data:** normalized and encoded dataset over-sampled for the `Stay` variable (full and best k)
* **(4) Under-sampled:** normalized and encoded dataset under-sampled for the `Stay` variable (full and best k)

### Preprocessing
As for the data preprocessing, the first step was to label encode the following variables: `Type of Admission`, `Severity of Illness`, `Age`, `Ward_Type`, `Hospital_type_code` and `Stay`, and one-hot encode `Hospital_region_code`, `Department` and `Ward_Facility_Code` variables. This results in a dataset with 42 columns instead of 12. After encoding, the numerical variables were standarized. It should also be noted that all rows with an existing NULL value in any of the columns was removed.

### Models
**(2) Best variables with 40% of the data**: Best model = One-vs-One with Decision tree

| Model                          | Hyperparameters  |   Training score |   Validation score |   Execution time (s) |
|--------------------------------|------------------|------------------|--------------------|----------------------|
| K Nearest Neighbors            | Neighbours=35,alg=auto        | 0.365931 | 0.327183 | 28.3985  |
| Decision Tree                  | Splitter=best, Criterion=gini | 0.988538 | 0.418228 | 0.404437 |
| Random Forest                  | Estimators=20                 | 0.97697  | 0.423709 |  3.2239  |
| Cat boost                      | --                            | 0.395143 | 0.345092 | 41.485   |
| XGB                            | --                            | 0.415178 | 0.348534 | 19.7226  |
| LGBM                           | --                            | 0.361257 | 0.339165 |  6.44496 |
| One-vs-Rest with Random forest | Estimators=20                 | 0.986743 | 0.427215 | 20.0754  |
| One-vs-One with Decision tree  | Splitter=best, Criterion=gini | 0.988538 | 0.431166 |  6.42007 |

**(3) Over-sampled best k with 40% of the data**: Best model =  One-vs-Rest with Random forest

| Model                          | Hyperparameters  |   Training score |   Validation score |   Execution time (s) |
|--------------------------------|------------------|------------------|--------------------|----------------------|
| K Nearest Neighbors            | Neighbours=35,alg=auto        | 0.418905 | 0.372682 | 23.1132|
| Decision Tree                  | Splitter=best, Criterion=gini | 0.998713 | 0.573894 |  0.397107|
| Random Forest                  | Estimators=20                 | 0.991439 | 0.613185 |  2.9642  |
| Cat boost                      | --                            | 0.480139 | 0.40874  | 55.2854|
| XGB                            | --                            | 0.474055 | 0.374892 | 24.6664|
| LGBM                           | --                            | 0.389484 | 0.342162 |  7.21668 |
| One-vs-Rest with Random forest | Estimators=20                 | 0.997476 | 0.614744 | 21.3685|
| One-vs-One with Decision tree  | Splitter=best, Criterion=gini | 0.998713 | 0.59304  |  5.86632 |

**(3) Over-sampled with 40% of the data**: Best model =  One-vs-Rest with Random forest

| Model                          | Hyperparameters  |   Training score |   Validation score |   Execution time (s) |
|--------------------------------|------------------|------------------|--------------------|----------------------|
| K Nearest Neighbors            | Neighbours=35,alg=auto        | 0.478138 | 0.433771 | 258.017    |
| Decision Tree                  | Splitter=best, Criterion=gini | 0.99993  | 0.630365 |   0.905    |
| Random Forest                  | Estimators=20                 | 0.997848 | 0.712332 |   3.85475  |
| Cat boost                      | --                            | 0.590199 | 0.503559 | 134.254    |
| XGB                            | --                            | 0.56932  | 0.456347 |  46.4873   |
| LGBM                           | --                            | 0.456189 | 0.4076   |   9.443    |
| One-vs-Rest with Random forest | Estimators=20                 | 0.999628 | 0.713797 |  29.9601   |
| One-vs-One with Decision tree  | Splitter=best, Criterion=gini | 0.99993  | 0.659397 |  10.5224   |

**(4) Under-sampled best k with 100% of the data**: Best model =  One-vs-One with Decision tree

| Model                          | Hyperparameters  |   Training score |   Validation score |   Execution time (s) |
|--------------------------------|------------------|------------------|--------------------|----------------------|
| K Nearest Neighbors            | Neighbours=35,alg=auto        | 0.269296 | 0.198248 | 3.42943    |
| Decision Tree                  | Splitter=best, Criterion=gini | 0.998068 | 0.57062  | 0.0810518    |
| Random Forest                  | Estimators=20                 | 0.993126 | 0.575472 | 0.629724  |
| Cat boost                      | --                            | 0.511951 | 0.310108 | 10.4947    |
| XGB                            | --                            | 0.606254 | 0.362129 | 4.73372   |
| LGBM                           | --                            | 0.420882 | 0.273854 | 2.00383    |
| One-vs-Rest with Random forest | Estimators=20                 | 0.997484 | 0.576685 | 4.41341   |
| One-vs-One with Decision tree  | Splitter=best, Criterion=gini | 0.998068 | 0.580323 | 1.51161   |

**(4) Under-sampled with 100% of the data**: Best model =  One-vs-Rest with Random forest

| Model                          | Hyperparameters  |   Training score |   Validation score |   Execution time (s) |
|--------------------------------|------------------|------------------|--------------------|----------------------|
| K Nearest Neighbors            | Neighbours=35,alg=auto        | 0.262602 | 0.200539 | 21.3488    |
| Decision Tree                  | Splitter=best, Criterion=gini | 0.99982  | 0.591509 | 0.156324    |
| Random Forest                  | Estimators=20                 | 0.997484 | 0.597574 | 0.743182  |
| Cat boost                      | --                            | 0.625483 | 0.378032 | 15.0065    |
| XGB                            | --                            | 0.666232 | 0.407547 | 10.8863   |
| LGBM                           | --                            | 0.484006 | 0.309838 | 2.06961    |
| One-vs-Rest with Random forest | Estimators=20                 | 0.999506 | 0.604043 | 4.82021   |
| One-vs-One with Decision tree  | Splitter=best, Criterion=gini | 0.99982  | 0.603639 | 2.52211   |

It can be observed that the ensemble methods are the ones performing better so let's try to optimize them a bit (executed with the over-sampled full dataset):

| Model | Hyperparameters | Validation score | Data percentage | Execution time (s) |
| -- | -- | -- | -- | -- |
| One-vs-Rest with RandomForest | Estimators=20 | 79.39% | 100% | 52.070884704589844 |
| One-vs-Rest with DecisionTree | Splitter=best, Criterion=gini | 74.77% | 100% | 9.400200843811035 |
| One-vs-One with RandomForest | Estimators=50 | 70.42% | 60% | 79.03182291984558 |
| One-vs-One with DecisionTree | Splitter=best, Criterion=gini | 83.25% | 100% | 16.468579292297363 |

## Demo
If you want to load the datasets and see their shapes you can run:

    $ python3 src/load_datasets.py
    
You can modify this file if you want to explore and play with the datasets.

If you want to run an iteration that will compare all models with the over-sampled data you can run:

    $ python3 src/try_all_models.py
    
You can also modify this file if you wish to run a different type of experiment.

## Conclusions
The best model that was obtained was the ensembled model composed of a One-vs-One classifier with a Decision tree classifier:

| Model | Hyperparameters | Validation score | Data percentage | Execution time (s) |
| -- | -- | -- | -- | -- |
| One-vs-One with DecisionTree | Splitter=best, Criterion=gini | 83.25% | 100% | 16.468579292297363 |

Compared to the models and/or solutions proposed in the `Code` section of the [Kaggle](https://www.kaggle.com/nehaprabhavalkar/av-healthcare-analytics-ii) dataset page, it seems to be one of the best models yet for this problem.

## Further work
First of all, in the Preprocessing section we could study the outliers for the numerical variables in order to remove thim, which may result in an increase of accuracy.

On the other hand, it also seems that we could gain more accuracy by fine-tuning the hyperparameters of the best models. We could also look for other models which might even work better "out of the box" than the ones proposed in this study.

## Licence
This project has been developed under the license [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
