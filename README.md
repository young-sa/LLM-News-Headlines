# Clickbait Detection with TF-IDF, Logistic Regression, and DistilBERT

## OVERVIEW

This project aims to detect clickbait in online news headlines by combining traditional machine learning (TF-IDF + Logistic Regression) with modern transformer-based models (DistilBERT). We also explore the impact of categorical metadata features and ensemble learning to improve performance.
We compare multiple approaches: TF-IDF + Logistic Regression (text only), TF-IDF + Logistic Regression (text + categorical features), DistilBERT fine-tuned for clickbait classification, and Ensemble combining DistilBERT and TF-IDF+Cat models


## SUMMARY OF WORK DONE

### Data

  * **Type:**
    * Input: CSV of labeled news headlines with accompanying categorical metadata (e.g., source, topic, region, sentiment, tone, emotions, time, intent)
    * Output: Binary label indicating whether the headline is clickbait (Yes) or not clickbait (No)
  * **Source:**
    * Custom dataset compiled from multiple news sources with manual labeling
  * **Splits:**
    * 70 training, 15 validation, 15 test (stratified split to maintain class balance)
   
### Compiling Data and Pre-processing

* **Data Cleaning**
    * Standardized all text to lowercase
    * Removed URLs, HTML tags, and extraneous punctuation (while preserving meaningful punctuation for NLP models)
    * Normalized whitespace
    * Fixed inconsistent label formats (Yes / No)
* **Feature Engineering**
    * Text Features:
      * Tokenized headlines
      * Converted to TF-IDF representation for Logistic Regression models
    * Categorical Features:
      * One-hot encoded features such as source, topic, region, sentiment, tone, emotions, time, intent
    * Class Balancing:
      * Computed class weights for training models to address clickbait/non-clickbait imbalance
* **Data Splitting**
    * Stratified split into train, validation, and test sets for consistent label distribution across sets

### Problem Formulation
  Goal: Evaluate whether combining traditional NLP (TF-IDF + Logistic Regression), categorical metadata, and transformer-based language models (DistilBERT) can improve clickbait headline detection.
    * **Models Used:**
      * TF-IDF + Logistic Regression (Text only): Baseline model using headline text
      * TF-IDF + Logistic Regression (Text + Categorical): Enriched model using both headline text and metadata (source, topic, sentiment, tone, etc.)
      * DistilBERT Fine-tuned: Transformer-based model fine-tuned on headline text
      * Ensemble (DistilBERT + TF-IDF+Cat): Combines predictions from DistilBERT and TF-IDF+Categorical Logistic Regression for improved performance

   * **Training**
    Model training and evaluation were conducted in Jupyter Notebook using Python, Scikit-learn, and Hugging Face Transformers.
      * Hardware Used: Training was performed on a local machine with GPU acceleration enabled for DistilBERT fine-tuning.

      * Steps Taken:
        * Preprocessed text data (lowercasing, punctuation handling, whitespace normalization)
        * One-hot encoded categorical features
        * Computed class weights to address label imbalance
        * Split dataset into stratified train / validation / test sets
        * Trained each model separately
        * Tuned thresholds for best Macro-F1 score on the validation set
        * Evaluated final performance on the test set
      
      * Challenges: Early runs with DistilBERT showed unstable performance due to learning rate and threshold tuning, so to fix that it was required backward-compatible adjustments for newer transformers API changes. In additiona to that, class imbalance meant threshold tuning was critical to improving recall without hurting precision.

  * **Evaluation**
    * Evaluated models on validation and test sets using:
    * Macro-F1 score
    * Precision
    * Recall
    * AUROC
    * Classification report
  * Compared four models:
    * TF-IDF + LR (Text only)
    * TF-IDF + LR (Text + Categorical)
    * DistilBERT Fine-tuned
    * Ensemble (DistilBERT + TF-IDF+Cat)
  * Created performance visualizations:
    * Bar chart comparing Macro-F1 across models
    * ROC curves for each model
