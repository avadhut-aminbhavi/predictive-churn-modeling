# Customer Churn Prediction for No-Churn Telecom

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange)
![License](https://img.shields.io/badge/License-Proprietary-red)

---

## ⚠️ IMPORTANT SECURITY WARNING (Read First!)

> **Your notebook currently exposes database credentials on GitHub.**
> The host, username, and password are visible in plain text inside the `.ipynb` file. **Take action immediately:**

1. 🔐 **Rotate the database password** on the MySQL server (the old one is compromised).
2. 🧹 **Remove hardcoded credentials** from the notebook.
3. 📄 **Use a `.env` file** for secrets and load them with `python-dotenv`.
4. 🚫 **Add `.env` to `.gitignore`** so it never gets committed.
5. 🕓 **Scrub git history** — a new commit won't hide the old credentials. Use `git-filter-repo` or BFG Repo-Cleaner.

---

## 📖 Project Overview

**No-Churn Telecom** is an established Indian telecom operator facing a customer churn rate of **over 10%** due to intensifying market competition. This project uses **Machine Learning** to move from reactive damage control to proactive customer retention — identifying high-risk customers *before* they leave.

### 🎯 Business Objectives

- Identify the critical variables that influence a customer's decision to churn.
- Build a predictive model to flag customers highly likely to churn.
- Generate an actionable `CHURN_FLAG` column (`1` = high risk, `0` = low risk).
- Empower the marketing team to target high-risk customers with specialized retention offers.
- Assist customer support teams in prioritizing at-risk customers for proactive engagement.

---

## 🗂 Project Structure

```
PRCL-0017-Customer-Churn/
│
├── PRCL-0017_Customer_Churn_Business_case.ipynb   # Main notebook
├── No_Churn_Telecom_Predictions.csv                # Generated output (after run)
├── README.md                                       # This file
├── requirements.txt                                # Python dependencies
├── .env.example                                    # Template for credentials
└── .gitignore
```

---

## 🛠 Tech Stack

| Category            | Tools                              |
| ------------------- | ---------------------------------- |
| Language            | Python 3.10+                       |
| Data Manipulation   | pandas, numpy                      |
| Database            | MySQL, SQLAlchemy, PyMySQL         |
| Visualization       | matplotlib, seaborn                |
| ML Framework        | LightGBM, scikit-learn             |
| Environment         | Jupyter Notebook / VS Code         |

---

## 📊 Dataset

The data is fetched from a remote MySQL database with the following schema:

| Column                 | Description                                |
| ---------------------- | ------------------------------------------ |
| `customer_id`          | Unique ID assigned to each customer        |
| `telecom_partner`      | Telecom service provider                   |
| `gender`               | Customer gender (Male / Female)            |
| `age`                  | Age of the customer                        |
| `state`                | State where the customer resides           |
| `city`                 | City of the customer                       |
| `pincode`              | Residential postal code                    |
| `date_of_registration` | Date when the customer joined              |
| `num_dependents`       | Number of dependents in the household      |
| `estimated_salary`     | Estimated annual salary                    |
| `calls_made`           | Total number of calls made                 |
| `sms_sent`             | Total SMS sent                             |
| `data_used`            | Total data usage (GB)                      |
| `churn`                | **Target** — 1 = churned, 0 = retained     |

- **Dataset size:** 243,553 rows × 14 columns
- **Churn rate:** ~20.05%

---

## 🚀 Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Credentials Securely

Create a `.env` file in the project root (**do NOT commit this file**):

```env
DB_HOST=your_host_here
DB_USER=your_username_here
DB_PASSWORD=your_new_password_here
DB_SCHEMA=project_telecom
```

Create a `.env.example` (this **IS** committed as a template — no real values):

```env
DB_HOST=
DB_USER=
DB_PASSWORD=
DB_SCHEMA=project_telecom
```

Add a `.gitignore`:

```gitignore
venv/
.env
*.csv
__pycache__/
.ipynb_checkpoints/
```

### 5. Update the Notebook to Read from `.env`

Replace the hardcoded credentials block in the notebook with:

```python
import os
from dotenv import load_dotenv
load_dotenv()

HOST     = os.getenv("DB_HOST")
USERNAME = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
SCHEMA   = os.getenv("DB_SCHEMA")
```

Make sure `python-dotenv` is added to your `requirements.txt`.

---

## ▶️ How to Run the Notebook

1. **Activate your virtual environment** (see Step 2 above).
2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `PRCL-0017_Customer_Churn_Business_case.ipynb`.
4. Run all cells top-to-bottom:
   ```
   Kernel → Restart & Run All
   ```
5. At the end, the notebook generates:
   ```
   No_Churn_Telecom_Predictions.csv
   ```
   This file contains the original data plus the actionable `CHURN_FLAG` column.

---

## 🔬 Methodology

The pipeline is organized into five phases:

### 1. Exploratory Data Analysis (EDA)

- Data quality checks — nulls, duplicates, data types.
- Target distribution (~20% churn — imbalanced classes).
- Box plots — `data_used`, `calls_made` vs churn.
- Histograms — `estimated_salary` distribution by churn.
- Count plots — `gender` vs churn.

### 2. Feature Engineering

- **Tenure:** Derived `tenure_days` from `date_of_registration`.
- **Dropped:** `customer_id`, raw `date_of_registration`.
- **One-Hot Encoding:** Applied to `telecom_partner`, `gender`, `state`, `city` (`drop_first=True` to avoid multicollinearity).

### 3. Preprocessing

- **Train/Test Split:** 80/20 with `stratify=y` to preserve churn ratio.
- **Feature Scaling:** `StandardScaler` fit on train, applied to test (leak-free).

### 4. Model Training

- **Algorithm:** `LGBMClassifier` (LightGBM).
- **Threshold Tuning:** Iterated `0.15 → 0.60` in steps of `0.02`, selecting the threshold that maximizes **Macro F1-Score** — balancing Recall against Precision.

### 5. Output Generation

- Predicts churn probability for the **entire** dataset.
- Applies the tuned threshold to create `CHURN_FLAG`.
- Exports `No_Churn_Telecom_Predictions.csv`.

---

## 📈 Results

| Metric              | Value       |
| ------------------- | ----------- |
| Optimal Threshold   | 0.21        |
| Accuracy            | 0.72        |
| Macro F1            | 0.50        |
| ROC-AUC             | **~0.50**   |

### ⚠️ Interpreting the AUC Score

An **ROC-AUC of ~0.50** indicates the current *static* feature set has **weak mathematical separability** between churned and retained customers. The model is functional but limited by the available features.

### 🚀 Recommended Next Steps (V2)

1. **Time-Series Features:** Month-over-month changes in `data_used`, `calls_made`, `sms_sent` — behavioral shifts are stronger churn signals than static snapshots.
2. **Interaction Features:** e.g. salary-to-dependents ratio, usage-per-tenure-day.
3. **Customer Support Signals:** Ticket counts, complaint history, average resolution time.
4. **Payment Behavior:** Late payments, top-ups, plan upgrades/downgrades.

---

## 🏗 Future Scope: Cloud Deployment (AWS)

For a production-grade deployment, the workflow maps to AWS as:

1. **Data Ingestion** — Amazon S3 / RDS
2. **Model Training & Hosting** — Amazon SageMaker
3. **Real-Time Inference** — API Gateway + Lambda + SageMaker Endpoint
4. **Orchestration** — Step Functions / Airflow for scheduled retraining

---

## 🧾 Requirements File (`requirements.txt`)

```txt
pandas
numpy
sqlalchemy
pymysql
python-dotenv
matplotlib
seaborn
scikit-learn
lightgbm
jupyter
```

---

## 🔐 Security Checklist Before Pushing

- [ ] Rotated DB password
- [ ] Removed hardcoded credentials from notebook
- [ ] Added `.env` to `.gitignore`
- [ ] Committed `.env.example` with empty values
- [ ] Scrubbed credentials from git history (`git-filter-repo` / BFG)
- [ ] Verified no secrets via `git log -p | grep -i password`

---

## 👥 Credits

- **Project Ref:** PM-PR-0017
- **Client:** No-Churn Telecom
- **Category:** Telecom — Churn Rate ML

---

## 📄 License

This project is proprietary work created for No-Churn Telecom. Contact the author before reuse or distribution.

---

## 🤝 Contributing

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit: `git commit -m "Add your feature"`.
4. Push: `git push origin feature/your-feature`.
5. Open a Pull Request.

---

## 📬 Contact

For questions or issues, please open a GitHub Issue or reach out to the project maintainer.

---

## ✅ Quick-Start Cheat Sheet

```bash
git clone <repo-url> && cd <repo>
python -m venv venv && source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env                                 # then fill in your DB creds
jupyter notebook PRCL-0017_Customer_Churn_Business_case.ipynb
# Kernel → Restart & Run All
# Result: No_Churn_Telecom_Predictions.csv
```
