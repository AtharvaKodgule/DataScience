# 📂 Loan Eligibility Prediction

> Predict whether a loan should be approved based on applicant data using supervised learning models. This end-to-end data science project includes everything from data preprocessing to model evaluation and deployment considerations.

---

### 📌 Table of Contents

- [📌 Table of Contents](#-table-of-contents)
- [🎯 Objective](#-objective)
- [📊 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🛠️ Technologies Used](#️-technologies-used)
- [📁 Project Structure](#-project-structure)
- [📈 Model Building](#-model-building)
- [📍 Results & Insights](#-results--insights)
- [📌 Future Work](#-future-work)
- [🚀 How to Run](#-how-to-run)
- [🗺️ Maps / Geo Integration (Optional)](#️-maps--geo-integration-optional)
- [📝 License](#-license)

---

### 🎯 Objective

The goal is to build a **binary classification model** to predict whether a loan application will be approved (`Yes`) or not (`No`) based on applicant details such as income, education, credit history, etc.

---

### 📊 Exploratory Data Analysis

Some insights discovered during EDA:

| Feature            | Insight                                                 |
|--------------------|----------------------------------------------------------|
| `Credit_History`   | Strong positive correlation with loan approval           |
| `ApplicantIncome`  | Outliers detected → applied log transformation           |
| `Loan_Amount_Term` | Most applicants prefer standard 360 months term          |
| `Gender`, `Married`, `Self_Employed` | Missing values handled via mode/median |



### 🛠️ Technologies Used

| Tool/Library        | Purpose                     |
|---------------------|-----------------------------|
| Python              | Programming Language        |
| Pandas, NumPy       | Data Manipulation           |
| Matplotlib, Seaborn | Visualization               |
| Scikit-learn        | Model Building              |
| Jupyter Notebook    | Development Environment     |

---

### 📁 Project Structure

```
Loan-Eligibility-Prediction/
│
├── data/
│   └── loan_data.csv
│
├── images/
│   ├── income_dist.png
│   └── gender_vs_loan.png
│
├── notebook/
│   └── LoanEligibilityPrediction.ipynb
│
├── README.md
└── requirements.txt
```

---

### 📈 Model Building

We tried multiple classification models:

- ✅ Logistic Regression
- ✅ Decision Tree
- ✅ Random Forest
- ✅ Support Vector Machine (SVM)

📌 **Evaluation Metrics Used:**

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Cross-validation

---

### 📍 Results & Insights

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 0.81     |
| Decision Tree       | 0.78     |
| Random Forest       | 0.84 ✅ |
| SVM                 | 0.76     |

📌 Random Forest performed the best with an accuracy of **84%**.

---

### 📌 Future Work

- Hyperparameter tuning (GridSearchCV)
- Deploy model using Flask or Streamlit
- Use XGBoost or LightGBM
- Integrate frontend for user interaction

---

### 🚀 How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/Loan-Eligibility-Prediction.git
cd Loan-Eligibility-Prediction
```

2. Create a virtual environment and install dependencies

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook

```bash
jupyter notebook notebook/LoanEligibilityPrediction.ipynb
```

---

### 🗺️ Maps / Geo Integration (Optional)

If your dataset had regional data (e.g., State/City), you can visualize regional approval rates.

```python
import folium

india_map = folium.Map(location=[22.9734, 78.6569], zoom_start=5)

# Example: Mark loan approval density by state
folium.CircleMarker(
    location=[19.7515, 75.7139],  # Maharashtra
    radius=10,
    popup="Loan Approvals: 123",
    color="green",
    fill=True,
).add_to(india_map)

india_map.save("images/loan_geo_map.html")
```

---



### 🌟 Connect with Me

**Atharva Kodgule**  
📧 [atharvakodgule17@gmail.com]  
🔗 [LinkedIn](https://linkedin.com/in/atharva-kodgule)  
📦 [GitHub](https://github.com/atharvakodgule)
