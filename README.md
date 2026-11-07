# Adult Census Income Classification — Assignment 2

## a. Problem statement
Build and evaluate five supervised classification models to predict whether an individual's annual income is `>50K` or `<=50K` using Adult Census demographic and employment attributes. Provide a Streamlit application that evaluates uploaded held-out test data using saved model pipelines without retraining.

## b. Dataset description
- Dataset: Adult Census Income (`adult.csv`)
- Source: Kaggle Adult Income dataset (based on the UCI Adult/Census Income data)
- Dataset rows: 48,842
- Target: `income` (`<=50K`, `>50K`)
- Input features: 14 in the original CSV; `fnlwgt` is removed as an identifier/weight field during modeling, leaving 13 predictors.
- Preprocessing: missing categorical marker `?` is normalized to missing; numeric features are median-imputed and scaled; categorical features are most-frequent-imputed and one-hot encoded. Stratified 80:20 train/test split (random_state=42).

## c. GitHub Repository Link
https://github.com/2026aj05509/adult-census-ml-classification/tree/main

Dataset reference (Kaggle): https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

## d. Models used and evaluation
Models: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Linear Support Vector Machine (SVM).

Metrics are calculated on the held-out test split. Positive class for Precision, Recall, F1, and AUC is `>50K`.

### Metrics comparison
The final values will be populated from `results/metrics_comparison.csv` after running `python train_models.py`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8062 | 0.9042 | 0.5638 | 0.8409 | 0.6750 | 0.5666 |
| Decision Tree | 0.7999 | 0.9013 | 0.5525 | 0.8618 | 0.6734 | 0.5665 |
| KNN | 0.8335 | 0.8486 | 0.6744 | 0.5881 | 0.6283 | 0.5236 |
| Naive Bayes | 0.6204 | 0.8286 | 0.3794 | 0.9213 | 0.5374 | 0.3866 |
| Support Vector Machine | 0.8008 | 0.9035 | 0.5543 | 0.8554 | 0.6727 | 0.5650 |

### Model observations
- **Logistic Regression:** Highest AUC (0.9042), accuracy (0.8062), and MCC (0.5666). It shows a strong balance of ranking ability and overall classification quality, with positive-class recall of 0.8409.
- **Decision Tree:** AUC (0.9013) and MCC (0.5665) are close to Logistic Regression. Recall is relatively high (0.8618), but precision is lower (0.5525), suggesting more false-positive predictions.
- **KNN:** Highest accuracy (0.8335) and precision (0.6744) among these runs, but recall (0.5881) and AUC (0.8486) are lower. It identifies fewer of the >50K class examples.
- **Naive Bayes:** Highest recall (0.9213), but lower precision (0.3794), accuracy (0.6204), and MCC (0.3866). It predicts the positive class more aggressively, leading to many false positives.
- **Support Vector Machine:** AUC (0.9035) is close to Logistic Regression. Recall (0.8554) is strong, with accuracy (0.8008) and MCC (0.5650) also close to the Decision Tree and Logistic Regression.
- **Overall winner: Logistic Regression**, selected by its highest AUC, accuracy, and MCC in this held-out split. Decision Tree's MCC is nearly tied, so the winner is not overwhelmingly better. Consider the metrics together rather than Accuracy alone.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong balanced performance; highest AUC, Accuracy, and MCC in this run. |
| Decision Tree | Similar AUC/MCC to Logistic Regression; strong recall with relatively lower precision. |
| KNN | Highest Accuracy and Precision, but weaker positive-class Recall and AUC. |
| Naive Bayes | Very high Recall, offset by low Precision, Accuracy, and MCC. |
| Support Vector Machine | Competitive AUC and Recall, with overall metrics close to Logistic Regression and Decision Tree. |
| Overall winner | Logistic Regression based on the highest AUC, Accuracy, and MCC on the fixed test split. |

## Run locally
1. Install Python 3.10+.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run training once: `python train_models.py`. This creates five saved model pipelines, `test_data.csv`, and the metrics CSV.
4. Start the app: `streamlit run app.py`.
5. Upload `test_data.csv` in the app. The app only evaluates the uploaded rows; it does not retrain.

## Repository contents
- `adult.csv`: source dataset
- `train_models.py`: split, train, evaluate, and persist models
- `app.py`: Streamlit test-data evaluator
- `model/`: serialized model pipelines including preprocessing
- `test_data.csv`: held-out test records with target labels
- `results/metrics_comparison.csv`: measured model metrics
- `requirements.txt`: deployment dependencies
