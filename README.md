# GSTAnalyticsHackathon-24
# Generated Key: 579b464db66ec23bdd0000010d7434a362f941876318b3efd0772432 (generated at time of participation)
# GST_Analysis_Project.zip -> SHA-256: ddfbb60287d7ed3ddc2e17877a2fe0d4b29944757aab119a5e05c7b45efeb8bd

Title: Machine Learning Model for Detecting GST Irregularities Using XGBoost on Imbalanced Data

Approach:
The GST Predictive Model leverages XGBoost to tackle imbalanced datasets, aiming to identify irregular transactions where Class 1 (irregular) is underrepresented compared to Class 0 (regular). XGBoost was chosen for its ensemble learning capabilities, effectively managing imbalances and offering high accuracy alongside feature importance insights.

To enhance recall, the model employed oversampling, boosting Class 1 representation to improve the detection of irregular transactions. However, this approach risked lowering precision. To counter this, class weights were adjusted, specifically doubling the weight of Class 1, which improved precision by ensuring more accurate minority class predictions without artificially inflating the dataset.

A critical feature, Column18, was essential for prediction accuracy. Removing it caused bias, so it was retained. Through balancing oversampling, class weights, and feature retention, the model successfully reduced GST irregularities, maintaining high accuracy and a strong ROC-AUC score.

SHA Verified is for the dataset verification only and its verified
Inside ZIP File, its contains report and Notebooks which have the final analysis, code for evaluation and conclusion overall.
