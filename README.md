# Student Grade Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python)](https://www.python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.2+-150458.svg?logo=pandas)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10+-11557C.svg?)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13+-3776AB.svg?logo=seaborn)](https://seaborn.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.2+-0131B4.svg?logo=numpy)](https://numpy.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-6.5%2B-F37626.svg?logo=jupyter)](https://jupyter.org)
[![Git](https://img.shields.io/badge/Git-2.47+-F05032.svg?logo=git)](https://git-scm.com)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/datasets/stealthtechnologies/predict-student-performance-dataset)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/XynaxDev/student-grade-prediction/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](https://github.com/XynaxDev/student-grade-prediction)

Discover **Student Grade Prediction**, a dynamic machine learning project that leverages linear regression to forecast academic outcomes with precision. Built on Python and Scikit-Learn, this repository transforms student data into predictive insights, empowering educators and learners alike. Explore now to see data science in action!

## Project Overview

This project analyzes a [Kaggle dataset]([insert_kaggle_url_here](https://www.kaggle.com/datasets/stealthtechnologies/predict-student-performance-dataset)) of 1,388 student records to predict grades using **Study Hours** and **Attendance Percentage**. The linear regression model delivers an R² score of 0.65, explaining 65% of grade variance, with a root mean square error (RMSE) of 5.3. A key finding: **Study Hours** drives 98% of predictions, while **Attendance Percentage** contributes only 2%.

- **Objective**: Develop core machine learning expertise and create a compelling portfolio for internship opportunities.
- **Dataset**: Kaggle dataset with grades, study hours, and attendance records.
- **Technologies**: Python, Scikit-Learn, Pandas, Matplotlib, Seaborn, NumPy, Jupyter Notebook, IPython, Git.

## Project Structure

Designed for clarity and reproducibility, the repository is structured as follows:
```bash
student-grade-prediction/
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for exploratory analysis
├── src/
│   ├── preprocess.py       # Data cleaning and outlier handling
│   ├── model.py            # Linear regression model training
│   ├── evaluate.py         # Model performance evaluation
│   ├── visualize.py        # Data visualization scripts
│   ├── predict.py          # Grade prediction for new inputs
├── README.md               # Project documentation
├── requirements.txt        # Dependency specifications
└── LICENSE                 # MIT License
```

## Methodology

<details>
<summary>Uncover the Workflow: Detailed Steps</summary>

1. **Data Preprocessing**:
   - Loaded 1,388 student records from Kaggle using Pandas.
   - Removed ~0.8% of **Study Hours** outliers with the Interquartile Range (IQR) method in `preprocess.py`.
   - Divided data into 80% training and 20% testing sets.

2. **Model Development**:
   - Implemented a linear regression model with Scikit-Learn in `model.py`.
   - Trained on **Study Hours** and **Attendance Percentage** to predict **Grades**.

3. **Evaluation**:
   - Measured performance in `evaluate.py`, yielding R² = 0.65 and RMSE = 5.3.
   - Determined **Study Hours** accounts for ~98% feature importance, with **Attendance Percentage** at ~2%.

4. **Visualization**:
   - Created visualizations using Matplotlib and Seaborn in `visualize.py`, including bar charts, box plots, and regression plots.

5. **Prediction**:
   - Enabled grade forecasting for new inputs with `predict.py`.

</details>

## Key Findings

- **Performance**: Captures 65% of grade variability (R² = 0.65), robust for an initial ML project.
- **Accuracy**: Predictions average a 5.3-point deviation (RMSE).
- **Insights**: **Study Hours** dominates grade outcomes (98% impact), while **Attendance Percentage** has minimal effect (2%).

Sample output from `evaluate.py`:
```
R² Score: 0.65
RMSE: 5.3
Feature Importance:
   Study Hours: 98%
   Attendance %: 2%
```

## Get Started

[![Run](https://img.shields.io/badge/Run-Now-red.svg)](#setup-instructions)
[![Kaggle](https://img.shields.io/badge/Kaggle-Data-blueviolet.svg)](https://www.kaggle.com/datasets/stealthtechnologies/predict-student-performance-dataset)
[![Tutorial](https://img.shields.io/badge/Tutorial-Read-teal.svg)](#methodology)

### Prerequisites
- Python 3.10+ ([Download](https://www.python.org/downloads/))
- Git 2.47+ ([Install](https://git-scm.com/downloads))
- Dependencies: See `requirements.txt` for exact versions

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/XynaxDev/student-grade-prediction.git
   cd student-grade-prediction
   ```

2. **Install Dependencies**:
   ```bash 
   pip install -r requirements.txt
   ```
3. **Run the Project**:
   - Process data: `python src/preprocess.py`
   - Train model: `python src/model.py`
   - Evaluate results: `python src/evaluate.py`
   - Visualize data: `python src/visualize.py`
   - Predict grades: `python src/predict.py`

## Future Enhancements 
- **Feature Expansion:** Incorporate factors like sleep hours or socioeconomic scores.
- **Advanced Models:** Experiment with Random Forest or Neural Networks.
- **Deployment:** Build a web interface for interactive predictions.

## Connect 🔗
Reach out—I’m all about collab and opportunities!

- **GitHub**: [![GitHub](https://img.shields.io/badge/XynaxDev-grey.svg?logo=github)](https://github.com/XynaxDev)
- **Email**: [![Email](https://img.shields.io/badge/Email-Contact-blue.svg)](mailto:akashkumar.cs27.com)
- **Contribute**: [![Contribute](https://img.shields.io/badge/Contribute-Welcome-green.svg)](https://github.com/XynaxDev/student-grade-prediction/issues)

Star the repository, share ideas, or contribute to shape the future of predictive analytics!

---

*Licensed under MIT. Dataset sourced from [Kaggle](https://kaggle.com).*