# Employee Burnout Risk Analysis and Prediction

## Repository Outline

```text
.
├── asset/
|   ├── burnout_multimodal_fusion_architecture.png  - Architecture diagram for the final multimodal fusion model
|   ├── The-Penn-Treebank-POS-tagset.png            - Reference image used during text preprocessing discussion
|   └── Thanos-Impossible-meme.jpg                  - Supporting visual used in the exploratory notebook
|
├── dataset/
|   ├── cleaned_data.csv                            - Cleaned modeling dataset used in the analysis and modeling stages
|   └── raw_data.csv                                - 80,000-row working sample extracted from the original public source
|
├── model_saving/
|   ├── encoder.pkl                                 - Saved OneHotEncoder for categorical features
|   ├── list_cat_cols.txt                           - List of categorical columns used in the final tabular branch
|   ├── list_feedback_col.txt                       - Name of the feedback text column used in the text branch
|   ├── list_num_cols.txt                           - List of numerical columns used in the final tabular branch
|   ├── list_skills_tokens.txt                      - Name of the skill text column used in the second text branch
|   ├── model_base.keras                            - Saved multimodal burnout prediction model
|   └── scaler.pkl                                  - Saved StandardScaler for numerical features
|
├── 1_employee-burnout_analysis.ipynb               - Exploratory analysis, cleaning, and business insight notebook
├── 2_employee-burnout_modeling.ipynb               - Feature engineering, multimodal modeling, and evaluation notebook
├── 3_employee-burnout_inference.ipynb              - Notebook for loading saved artifacts and running inference
└── README.md                                       - Project documentation and summary of the workflow
```

## Overview

This project explores employee burnout through an end-to-end workflow, starting from exploratory analysis and continuing into multimodal regression modeling.

The main objective is to understand which employee conditions, work patterns, and organizational signals appear most closely associated with higher burnout risk, then translate those findings into a predictive workflow that can support earlier review in an HR setting.

The final model is designed as a decision-support tool rather than a replacement for human judgment. In practice, the predicted burnout risk score should be used to highlight patterns, support earlier follow-up, and help prioritize employee review, especially when multiple warning signals appear together.

Model Deployment Demo: [Hugging Face](https://huggingface.co/spaces/seankafka/Employee-Burnout-Prediction)

## Key Insight

Employee burnout in this dataset appears to be more closely related to **day-to-day work experience, workload pressure, satisfaction, and operational collaboration signals** than to static background information alone.

That makes the final model most useful as an **early warning and monitoring-support tool**, especially for identifying employee profiles that show stronger burnout patterns. At the same time, the near-perfect predictive result should be interpreted carefully because this synthetic dataset appears highly structured and unusually easy to learn.

## Objectives

The goals of this project are:

* Understand how `burnout_risk` behaves across structured, categorical, and text-based employee features
* Clean the raw dataset and simplify noisy or leakage-prone variables before modeling
* Build a modeling-ready dataset through feature engineering, grouped categorical labels, and text preparation
* Design a multimodal baseline model that can combine tabular data with feedback and skill text inputs
* Evaluate the model on a held-out test set using regression metrics and prediction diagnostics
* Save the trained model and preprocessing artifacts for reuse in inference and deployment

## Dataset

This project uses the public Hugging Face dataset **`BrotherTony/employee-burnout-turnover-prediction-800k`** as the original source.

The repository works from an **80,000-row sampled subset** stored in `dataset/raw_data.csv`, which is then cleaned into `dataset/cleaned_data.csv` for the main analysis and modeling workflow.

The raw sample contains **31 columns**, while the cleaned modeling dataset contains **22 columns** covering:

* Employee level and department context
* Workload, satisfaction, and performance-related numeric signals
* Communication and collaboration indicators
* Text-based employee feedback
* Skill profile information
* Burnout-related outcome fields

Target variable:

* **`burnout_risk`** as a continuous regression target

## End-to-End Workflow

### 1. Data Cleaning and Preparation

The project begins by checking duplicates, reviewing missing values, cleaning categorical inconsistencies, and removing columns that behave more like identifiers, generated summaries, or potential leakage sources than reliable predictors.

The raw text fields are also simplified into more usable forms. Employee feedback is normalized into `feedback_tokens`, while technical and soft skill lists are combined into a cleaner `skillset` field for later text-based modeling.

### 2. Exploratory Data Analysis

The analysis notebook focuses on understanding how burnout behaves across numeric features, categorical patterns, and cleaned text fields.

This stage shows that burnout risk is more clearly associated with workload intensity, lower satisfaction, weaker collaboration signals, and related work-experience variables than with static employee profile features alone.

### 3. Feature Engineering and Selection

For the modeling stage, the dataset is narrowed further so the workflow remains compact, interpretable, and suitable for a student-level HR portfolio project.

The final model uses:

* **13 numerical features**: `tenure_months`, `salary`, `performance_score`, `satisfaction_score`, `workload_score`, `team_sentiment`, `project_completion_rate`, `overtime_hours`, `training_participation`, `collaboration_score`, `email_sentiment`, `role_complexity_score`, and `career_progression_score`
* **3 categorical features**: `job_level`, `left_company`, and `department_group`
* **2 text features**: `feedback_tokens` and `skillset`

### 4. Preprocessing and Multimodal Input Design

The structured tabular branch applies:

* `StandardScaler` for numerical features
* `OneHotEncoder` for categorical features

The text branches use a lightweight neural text preparation flow:

* `TextVectorization`
* `Embedding`
* `GlobalAveragePooling1D`

### 5. Model Definition and Training

The final model is a **multimodal fusion neural network** with:

* one tabular branch for structured features
* one feedback text branch
* one skill text branch
* two shared dense layers after concatenation

The model is trained as a regression task using:

* `Adam` optimizer
* `MSE` loss
* `MAE` as an additional metric
* `EarlyStopping` to reduce unnecessary training and help control overfitting

### 6. Final Evaluation and Inference

On the held-out test set, the final model achieves:

| Evaluation Stage | Metric | Score |
|---|---|---:|
| Test set | MSE | 0.000015 |
| Test set | MAE | 0.0025 |
| Test set | R-squared | 0.9999 |

The inference notebook then demonstrates how the saved scaler, encoder, metadata files, and trained model can be reused to generate burnout predictions for a new employee record.

## Key Findings

### 1. Burnout Risk Is More Closely Tied to Work Experience Than Static Profile

The strongest patterns in this dataset come from variables related to satisfaction, workload, collaboration, overtime, and operational work signals. Static profile information appears less informative on its own.

### 2. Text Adds Context, but Structured Features Likely Carry Most of the Signal

The multimodal design allows the project to include employee feedback and skill information, but the extremely strong predictive result suggests that the tabular features are still doing most of the heavy lifting.

### 3. A Lightweight Multimodal Architecture Is Already Enough to Fit the Target Extremely Well

Even with a fairly simple multimodal neural network, the model reaches near-perfect regression performance on the test set. This indicates that the dataset is highly learnable even without a very deep or complex architecture.

### 4. The Main Modeling Risk Is Misleading Performance, Not Underfitting

The final metrics are so strong that the main issue is no longer whether the model can fit the dataset. The more important question is whether the performance reflects real robustness or simply a highly structured synthetic target.

## Business Recommendations

### 1. Use the Model as an Early Warning Signal

Predicted burnout risk can help HR teams identify employee profiles that may require closer review, especially when the score aligns with lower satisfaction, heavier workload, and weaker collaboration-related patterns.

### 2. Prioritize Follow-Up for High-Risk Predictions

Employees with higher predicted burnout scores can be reviewed earlier so that support, coaching, workload adjustments, or manager follow-up can happen before the problem becomes more severe.

### 3. Focus Future Improvement on Reliability Rather Than Chasing Smaller Errors

Since the current synthetic dataset already produces near-ceiling performance, the next practical improvement is to test stability, robustness, and behavior under noisier and more realistic input conditions.

## Limitations

* The dataset is synthetic, so the relationships may be cleaner and more predictable than real organizational data
* Extremely high test performance may be misleading if interpreted as real-world robustness
* The true contribution of the text branches is still uncertain without a direct ablation comparison
* Statistical association does not imply causation
* The model output should be treated as decision support, not automated judgment

## Tools and Libraries

* **Language**: Python
* **Data Handling**: Pandas, NumPy
* **Dataset Access**: Hugging Face `datasets`
* **Visualization**: Matplotlib, Seaborn, WordCloud
* **Text Processing**: NLTK, contractions
* **Machine Learning**: scikit-learn
* **Deep Learning**: TensorFlow / Keras
* **Model Saving**: pickle, JSON
* **Deployment Demo**: Hugging Face Spaces

## Getting Started

### Prerequisites

* Python 3.10+
* Jupyter Notebook or JupyterLab
* pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/seankafka/employee-burnout-prediction.git
   cd employee-burnout-prediction
   ```

2. Install the core dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn wordcloud nltk contractions datasets scikit-learn tensorflow notebook
   ```

### Project Order

Run the project in this order:

`1_employee-burnout_analysis.ipynb`  
`2_employee-burnout_modeling.ipynb`  
`3_employee-burnout_inference.ipynb`

## Author

This project was developed as part of a data learning journey, with a focus on building practical, interpretable, and business-relevant HR analytics skills.

Sean Kafka Adhyaksa  
[GitHub](https://github.com/seankafka) || [LinkedIn](https://www.linkedin.com/in/seankafka/)
