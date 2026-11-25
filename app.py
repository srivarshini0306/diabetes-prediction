import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import warnings

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction | Predictive Analytics",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        color: #172033;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #64748b;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        color: #172033;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
    }

    .positive {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #fee2e2;
        border: 1px solid #fecaca;
    }

    .negative {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #dcfce7;
        border: 1px solid #bbf7d0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET = "Outcome"

# Selected feature set used in the project
SELECTED_FEATURES = [
    "Glucose",
    "BMI",
    "Age",
    "Pregnancies",
    "DiabetesPedigreeFunction"
]

RANDOM_STATE = 42


# ============================================================
# FIND DATASET
# ============================================================

def find_dataset():

    base_dir = Path(__file__).resolve().parent

    possible_files = [
        base_dir / "diabetes_cleaned.csv",
        base_dir / "diabetes.csv"
    ]

    for file in possible_files:
        if file.exists():
            return file

    return None


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    file_path = find_dataset()

    if file_path is None:
        return None

    df = pd.read_csv(file_path)

    return df


df = load_data()


# ============================================================
# CHECK DATASET
# ============================================================

if df is None:

    st.error("❌ Diabetes dataset not found.")

    st.info("""
    Make sure `diabetes_cleaned.csv` or `diabetes.csv`
    is present in the same GitHub folder as `app.py`.
    """)

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Missing columns in dataset: {missing_columns}"
    )

    st.stop()


# ============================================================
# HANDLE ZERO VALUES AS MISSING
# ============================================================

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df_model = df.copy()

for column in ZERO_AS_MISSING:

    df_model[column] = df_model[column].replace(
        0,
        np.nan
    )


# ============================================================
# MEDIAN IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

df_model[FEATURES] = imputer.fit_transform(
    df_model[FEATURES]
)


# ============================================================
# PAGE NAVIGATION
# ============================================================

st.sidebar.title("🩺 Diabetes Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔬 Data Analysis",
        "📊 Model Comparison",
        "🧪 Diabetes Prediction",
        "ℹ️ About Project"
    ]
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():

    return {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=RANDOM_STATE
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE
            ),

        "KNN":
            KNeighborsClassifier(
                n_neighbors=5
            ),

        "SVM":
            SVC(
                probability=True,
                random_state=RANDOM_STATE
            ),

        "Gaussian Naive Bayes":
            GaussianNB()
    }


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_data
def run_model_comparison(data):

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    feature_sets = {

        "All Features":
            FEATURES,

        "Selected Features":
            SELECTED_FEATURES
    }

    models = get_models()

    results = []

    trained_models = {}

    for feature_set_name, feature_list in feature_sets.items():

        for model_name, model in models.items():

            pipeline = Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    model
                )
            ])

            pipeline.fit(
                X_train[feature_list],
                y_train
            )

            y_pred = pipeline.predict(
                X_test[feature_list]
            )

            y_probability = pipeline.predict_proba(
                X_test[feature_list]
            )[:, 1]

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            roc_auc = roc_auc_score(
                y_test,
                y_probability
            )

            # Composite score
            candidate_score = np.mean([
                accuracy,
                precision,
                recall,
                f1,
                roc_auc
            ])

            results.append({

                "Feature Set":
                    feature_set_name,

                "Algorithm":
                    model_name,

                "Accuracy":
                    accuracy,

                "Precision":
                    precision,

                "Recall":
                    recall,

                "F1 Score":
                    f1,

                "ROC-AUC":
                    roc_auc,

                "Candidate Score":
                    candidate_score
            })

            key = (
                feature_set_name +
                " | " +
                model_name
            )

            trained_models[key] = {
                "pipeline": pipeline,
                "features": feature_list,
                "X_test": X_test[feature_list],
                "y_test": y_test
            }

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Candidate Score",
        ascending=False
    ).reset_index(drop=True)

    return results_df, trained_models


# ============================================================
# RUN MODELS
# ============================================================

results_df, trained_models = run_model_comparison(
    df_model
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="title">'
        '🩺 Diabetes Prediction Using Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Predictive Analytics Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Project Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("""
    This project develops a machine learning based predictive
    analytics system for diabetes prediction.

    The workflow includes dataset understanding, data cleaning,
    exploratory data analysis, correlation analysis,
    multicollinearity analysis, feature selection, preprocessing,
    model development and comparative evaluation.
    """)

    st.divider()

    # Dataset metrics

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Dataset Records",
            len(df)
        )

    with c2:

        st.metric(
            "Input Features",
            len(FEATURES)
        )

    with c3:

        st.metric(
            "Selected Features",
            len(SELECTED_FEATURES)
        )

    with c4:

        st.metric(
            "ML Experiments",
            len(results_df)
        )

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Project Workflow'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    **Dataset Understanding**

    ↓

    **Data Cleaning**

    ↓

    **Handling Missing / Invalid Values**

    ↓

    **Exploratory Data Analysis**

    ↓

    **Correlation Analysis**

    ↓

    **Multicollinearity Analysis**

    ↓

    **Feature Selection**

    ↓

    **Feature Scaling**

    ↓

    **Multiple Machine Learning Models**

    ↓

    **All Features vs Selected Features**

    ↓

    **Comparative Evaluation**

    ↓

    **Final Model Selection**

    ↓

    **Diabetes Prediction**
    """)

    st.success(
        "The application evaluates multiple models before making predictions."
    )

    st.warning(
        "⚠️ This application is an academic predictive analytics "
        "prototype and is not a medical diagnostic system."
    )


# ============================================================
# DATA ANALYSIS PAGE
# ============================================================

elif page == "🔬 Data Analysis":

    st.markdown(
        '<div class="section-title">'
        '🔬 Exploratory & Feature Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "Dataset",
        "Statistics",
        "Correlation",
        "Feature Selection"
    ])

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    with tabs[0]:

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.subheader("Dataset Shape")

        st.write(
            f"Rows: **{df.shape[0]}**  |  "
            f"Columns: **{df.shape[1]}**"
        )

        st.subheader("Target Distribution")

        outcome_counts = df[TARGET].value_counts()

        col1, col2 = st.columns(2)

        with col1:

            st.dataframe(
                outcome_counts.rename(
                    index={
                        0: "Non-Diabetic",
                        1: "Diabetic"
                    }
                )
            )

        with col2:

            fig, ax = plt.subplots()

            ax.bar(
                ["Non-Diabetic", "Diabetic"],
                outcome_counts.values
            )

            ax.set_ylabel("Number of Records")

            st.pyplot(fig)


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    with tabs[1]:

        st.subheader("Descriptive Statistics")

        st.dataframe(
            df[FEATURES].describe().T,
            use_container_width=True
        )

        st.subheader("Missing / Zero Values")

        zero_counts = (
            df[ZERO_AS_MISSING] == 0
        ).sum()

        zero_table = pd.DataFrame({
            "Feature":
                zero_counts.index,

            "Zero Values":
                zero_counts.values
        })

        st.dataframe(
            zero_table,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    with tabs[2]:

        st.subheader(
            "Correlation Heatmap"
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        correlation = (
            df[FEATURES + [TARGET]]
            .corr()
        )

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

        st.subheader(
            "Correlation with Outcome"
        )

        target_corr = (
            correlation[TARGET]
            .drop(TARGET)
            .sort_values(
                ascending=False
            )
        )

        st.dataframe(
            target_corr.to_frame(
                "Correlation"
            ),
            use_container_width=True
        )


    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------

    with tabs[3]:

        st.subheader(
            "Selected Features"
        )

        st.write("""
        The project evaluates a reduced feature set along with
        the complete feature set.
        """)

        selected_df = pd.DataFrame({
            "Selected Features":
                SELECTED_FEATURES
        })

        st.dataframe(
            selected_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Selected vs All Features"
        )

        comparison = pd.DataFrame({
            "Feature Set": [
                "All Features",
                "Selected Features"
            ],

            "Number of Features": [
                len(FEATURES),
                len(SELECTED_FEATURES)
            ]
        })

        st.bar_chart(
            comparison.set_index(
                "Feature Set"
            )
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "📊 Model Comparison":

    st.markdown(
        '<div class="section-title">'
        '📊 Comparative Model Evaluation'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Six classification algorithms are evaluated using both
    the complete feature set and the selected feature set.
    """)

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    display_df = results_df.copy()

    percentage_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Candidate Score"
    ]

    for column in percentage_columns:

        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best = results_df.iloc[0]

    st.subheader(
        "🏆 Best Performing Experiment"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Algorithm",
            best["Algorithm"]
        )

    with c2:

        st.metric(
            "Feature Set",
            best["Feature Set"]
        )

    with c3:

        st.metric(
            "Accuracy",
            f"{best['Accuracy'] * 100:.2f}%"
        )

    with c4:

        st.metric(
            "ROC-AUC",
            f"{best['ROC-AUC'] * 100:.2f}%"
        )

    st.info(
        f"""
        Based on the composite evaluation score used in this
        application, **{best['Algorithm']} with {best['Feature Set']}**
        achieved the highest overall score.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # METRIC SELECTION
    # --------------------------------------------------------

    metric = st.selectbox(
        "Select Evaluation Metric",

        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    sns.barplot(
        data=results_df,
        x="Algorithm",
        y=metric,
        hue="Feature Set",
        ax=ax
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel(metric)

    ax.set_xlabel("")

    plt.xticks(
        rotation=25
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------------
    # COMPARISON HEATMAP
    # --------------------------------------------------------

    st.subheader(
        "Model Performance Heatmap"
    )

    heatmap_df = results_df.copy()

    heatmap_df["Experiment"] = (
        heatmap_df["Algorithm"] +
        " - " +
        heatmap_df["Feature Set"]
    )

    heatmap_df = heatmap_df.set_index(
        "Experiment"
    )[[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]]

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🧪 Diabetes Prediction":

    st.markdown(
        '<div class="section-title">'
        '🧪 Diabetes Risk Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Enter the patient's information below to generate a
    machine learning prediction.
    """)

    st.divider()

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=300.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=1000.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=30.0
        )

        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.47,
            step=0.001
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        selected_model = st.selectbox(
            "Select Prediction Model",
            [
                "Random Forest",
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "SVM",
                "Gaussian Naive Bayes"
            ]
        )

    with c2:

        selected_feature_set = st.selectbox(
            "Select Feature Set",
            [
                "All Features",
                "Selected Features"
            ]
        )

    st.divider()

    predict_button = st.button(
        "🔍 Predict Diabetes",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        key = (
            selected_feature_set +
            " | " +
            selected_model
        )

        model_information = (
            trained_models[key]
        )

        pipeline = (
            model_information["pipeline"]
        )

        model_features = (
            model_information["features"]
        )

        input_data = pd.DataFrame({

            "Pregnancies": [
                pregnancies
            ],

            "Glucose": [
                glucose
            ],

            "BloodPressure": [
                blood_pressure
            ],

            "SkinThickness": [
                skin_thickness
            ],

            "Insulin": [
                insulin
            ],

            "BMI": [
                bmi
            ],

            "DiabetesPedigreeFunction": [
                pedigree
            ],

            "Age": [
                age
            ]
        })

        # Apply same zero-as-missing logic
        for column in ZERO_AS_MISSING:

            if input_data[column].iloc[0] == 0:

                input_data[column] = np.nan

        # Median imputation using training dataset medians
        for column in ZERO_AS_MISSING:

            if input_data[column].isna().any():

                input_data[column] = (
                    input_data[column]
                    .fillna(
                        df_model[column].median()
                    )
                )

        input_features = (
            input_data[model_features]
        )

        prediction = pipeline.predict(
            input_features
        )[0]

        probability = (
            pipeline.predict_proba(
                input_features
            )[0][1]
        )

        st.divider()

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            st.markdown(
                f"""
                <div class="positive">

                <h2>⚠️ Diabetes Predicted</h2>

                <h3>
                Predicted Probability:
                {probability * 100:.2f}%
                </h3>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="negative">

                <h2>✅ Diabetes Not Predicted</h2>

                <h3>
                Predicted Probability:
                {probability * 100:.2f}%
                </h3>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # Prediction details

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Prediction",
                "Diabetic"
                if prediction == 1
                else "Non-Diabetic"
            )

        with c2:

            st.metric(
                "Probability",
                f"{probability * 100:.2f}%"
            )

        with c3:

            st.metric(
                "Model",
                selected_model
            )

        st.info("""
        ⚠️ This prediction is intended for academic and
        demonstration purposes only. It should not be considered
        a medical diagnosis.
        """)


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="section-title">'
        'ℹ️ About the Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("""
    ### Diabetes Prediction Using Predictive Analytics

    This project applies predictive analytics and machine
    learning techniques to predict diabetes outcomes from
    clinical and demographic attributes.

    The project follows a complete analytical workflow:

    1. Dataset understanding
    2. Data preprocessing
    3. Exploratory data analysis
    4. Missing-value treatment
    5. Correlation analysis
    6. Multicollinearity analysis
    7. Feature selection
    8. Feature scaling
    9. Model development
    10. Comparative model evaluation
    11. Final model selection
    12. Diabetes prediction
    """)

    st.divider()

    st.subheader(
        "Input Variables"
    )

    feature_description = pd.DataFrame({

        "Feature": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ],

        "Description": [
            "Number of pregnancies",
            "Plasma glucose concentration",
            "Diastolic blood pressure",
            "Triceps skin fold thickness",
            "Serum insulin level",
            "Body Mass Index",
            "Diabetes pedigree function",
            "Age in years"
        ]
    })

    st.dataframe(
        feature_description,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Machine Learning Algorithms"
    )

    algorithms = pd.DataFrame({

        "Algorithm": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN",
            "SVM",
            "Gaussian Naive Bayes"
        ],

        "Type": [
            "Linear Classification",
            "Tree-based Classification",
            "Ensemble Classification",
            "Distance-based Classification",
            "Kernel-based Classification",
            "Probabilistic Classification"
        ]
    })

    st.dataframe(
        algorithms,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.warning("""
    This project is an academic predictive analytics system.
    Predictions should not be used independently for medical
    diagnosis or treatment decisions.
    """)
