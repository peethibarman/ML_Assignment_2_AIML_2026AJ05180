# Breast Cancer Classification Using Machine Learning

**Name:** Preethi Barman\
**ID:** 2026AJ05180\
**Account:** 2026aj05180@wilp.bits-pilani.ac.in

## a. Problem Statement

The objective of this project is to build and evaluate machine learning
classification models that predict whether a breast tumour is **Benign
(B)** or **Malignant (M)** using diagnostic measurements. Five
classification algorithms are trained and compared using a consistent
train/test split and evaluation metrics. The goal is to identify the
model that performs best on the selected dataset.

## b. Dataset Description

-   **Dataset file:** `data.csv`
-   **Dataset:** Breast Cancer Wisconsin (Diagnostic) dataset.
-   **Task:** Binary classification of diagnosis into Benign and
    Malignant.
-   **Total records:** 569 (455 training records and 114 testing
    records).
-   **Input features:** 14 selected diagnostic features.
-   **Target:** Diagnosis (Benign or Malignant).
-   **Data split:** 80% training and 20% testing.
-   **Random state:** 42.

The 14 selected features are:

`radius1, perimeter1, area1, compactness1, concavity1, concave_points1, radius2, perimeter2, radius3, perimeter3, area3, compactness3, concavity3, concave_points3`

The evaluation reports show 72 Benign and 42 Malignant records in the
114-record test set.

## c. GitHub Repository Link

[ML Assignment 2 -- AIML
2026AJ05180](https://github.com/preethibarman03/ML_Assignment_2_AIML_2026AJ05180)

The repository contains the dataset, model training/evaluation scripts,
and application files for the assignment.

**Deployed Streamlit application:**
<https://mlassignment2aiml2026aj05180-jzt6gcnjwnghy8qfm35yft.streamlit.app/>

## d. Models Used and Evaluation

All five models were evaluated on the same 114-record test set. The
metrics below are taken from the model evaluation output files. Values
are reported as decimals.

  -------------------------------------------------------------------------------
  ML Model       Accuracy        AUC   Precision     Recall         F1        MCC
  Name                                                                 
  ------------ ---------- ---------- ----------- ---------- ---------- ----------
  Logistic         0.9825     0.9993      1.0000     0.9524     0.9756     0.9626
  Regression                                                           

  Decision         0.9123     0.9008      0.9000     0.8571     0.8780     0.8102
  Tree                                                                 

  KNN              0.9386     0.9906      0.9730     0.8571     0.9114     0.8688

  Naive Bayes      0.9386     0.9921      0.9487     0.8810     0.9136     0.8675

  Support          0.9561     0.9987      1.0000     0.8810     0.9367     0.9076
  Vector                                                               
  Machine                                                              
  -------------------------------------------------------------------------------

### Observations on Model Performance

  -----------------------------------------------------------------------
  ML Model Name                       Observation about model performance
  ----------------------------------- -----------------------------------
  Logistic Regression                 Achieved the best overall results:
                                      98.25% accuracy, 0.9993 AUC, 1.0000
                                      precision, 0.9524 recall, 0.9756
                                      F1-score and 0.9626 MCC. It
                                      correctly classified 112 of 114
                                      test records, with two malignant
                                      cases classified as benign.

  Decision Tree                       Recorded the lowest accuracy
                                      (91.23%) and AUC (0.9008) among the
                                      five models. Its confusion matrix
                                      shows 10 incorrect predictions (4
                                      false positives and 6 false
                                      negatives).

  KNN                                 Achieved 93.86% accuracy and a high
                                      AUC of 0.9906. Its precision was
                                      0.9730, while recall was 0.8571. It
                                      produced seven incorrect
                                      predictions (1 false positive and 6
                                      false negatives).

  Naive Bayes                         Achieved 93.86% accuracy, 0.9921
                                      AUC and 0.9136 F1-score. Its recall
                                      (0.8810) was higher than KNN's, and
                                      it produced seven incorrect
                                      predictions (2 false positives and
                                      5 false negatives).

  Support Vector Machine              Achieved 95.61% accuracy and 0.9987
                                      AUC, with perfect precision
                                      (1.0000). Its recall was 0.8810 and
                                      F1-score was 0.9367. It produced
                                      five false negative predictions and
                                      no false positives.
  -----------------------------------------------------------------------

### Overall Winner for the Dataset

**Logistic Regression is the overall winner** based on this experiment.
It achieved the highest accuracy (0.9825), AUC (0.9993), recall
(0.9524), F1-score (0.9756) and MCC (0.9626). Its precision was also
1.0000, tied with SVM. It made only two errors on the test set, compared
with five or more errors for each of the other models.

This conclusion is based on the reported 80:20 split with random state
42. It is an assignment-level model comparison and should not be
interpreted as clinical validation or as a substitute for professional
medical diagnosis.
