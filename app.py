import streamlit as st
import pandas as pd
import numpy as np

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
    confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)


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

SELECTED_FEATURES = [
    "Glucose",
    "BMI",
    "Age",
    "Pregnancies",
    "DiabetesPedigreeFunction"
]

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

RANDOM_STATE = 42


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
}

.title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FIND DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

cleaned_file = BASE_DIR / "diabetes_cleaned.csv"
original_file = BASE_DIR / "diabetes.csv"


if cleaned_file.exists():
    DATA_FILE = cleaned_file
elif original_file.exists():
    DATA_FILE = original_file
else:
    st.error(
        "❌ Dataset not found. Make sure diabetes_cleaned.csv "
        "or diabetes.csv is in the same folder as app.py."
    )
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_FILE)

    data.columns = data.columns.str.strip()

    return data


df = load_data()


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Missing columns: {missing_columns}"
    )

    st.stop()


# ============================================================
# PREPROCESS DATA
# ============================================================

@st.cache_data
def prepare_data(data):

    data = data.copy()

    # Replace invalid zero values with NaN
    for col in ZERO_AS_MISSING:
        data[col] = data[col].replace(0, np.nan)

    # Median imputation
    imputer = SimpleImputer(strategy="median")

    data[FEATURES] = imputer.fit_transform(
        data[FEATURES]
    )

    return data


df_model = prepare_data(df)


# ============================================================
# MODELS
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
# MODEL COMPARISON
# ============================================================

@st.cache_data
def run_model_comparison(data):

    X = data[FEATURES]
    y = data[TARGET]

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

            y_prob = pipeline.predict_proba(
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
                y_prob
            )

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
                "model": pipeline,
                "features": feature_list
            }

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Candidate Score",
        ascending=False
    ).reset_index(drop=True)

    return results_df, trained_models


results_df, trained_models = run_model_comparison(
    df_model
)


# ============================================================
# SIDEBAR
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
# HOME
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

    st.subheader("Project Overview")

    st.write("""
    This project applies predictive analytics and machine
    learning techniques to predict diabetes outcomes.

    The complete workflow includes:

    • Data Understanding  
    • Data Cleaning  
    • Missing Value Treatment  
    • Exploratory Data Analysis  
    • Correlation Analysis  
    • Multicollinearity Analysis  
    • Feature Selection  
    • Feature Scaling  
    • Model Development  
    • Comparative Model Evaluation  
    • Final Model Selection  
    • Diabetes Prediction
    """)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Dataset Records",
        len(df)
    )

    c2.metric(
        "Input Features",
        len(FEATURES)
    )

    c3.metric(
        "Selected Features",
        len(SELECTED_FEATURES)
    )

    c4.metric(
        "Model Experiments",
        len(results_df)
    )

    st.divider()

    st.subheader("Best Model")

    best = results_df.iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Algorithm",
        best["Algorithm"]
    )

    c2.metric(
        "Feature Set",
        best["Feature Set"]
    )

    c3.metric(
        "Candidate Score",
        f"{best['Candidate Score'] * 100:.2f}%"
    )

    st.success(
        f"🏆 Best experiment: {best['Algorithm']} "
        f"using {best['Feature Set']}."
    )

    st.warning(
        "⚠️ This application is for academic demonstration "
        "and is not a medical diagnostic system."
    )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "🔬 Data Analysis":

    st.title("🔬 Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset",
        "Statistics",
        "Correlation",
        "Feature Selection"
    ])

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    with tab1:

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.write(
            f"Dataset shape: **{df.shape[0]} rows × "
            f"{df.shape[1]} columns**"
        )

        st.subheader("Target Distribution")

        target_counts = (
            df[TARGET]
            .value_counts()
            .rename({
                0: "Non-Diabetic",
                1: "Diabetic"
            })
        )

        st.bar_chart(target_counts)

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    with tab2:

        st.subheader(
            "Descriptive Statistics"
        )

        st.dataframe(
            df[FEATURES].describe().T,
            use_container_width=True
        )

        st.subheader(
            "Zero Values Treated as Missing"
        )

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
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    with tab3:

        st.subheader(
            "Correlation with Diabetes Outcome"
        )

        correlation = (
            df[FEATURES + [TARGET]]
            .corr()
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

        st.subheader(
            "Correlation Matrix"
        )

        st.dataframe(
            correlation.round(3),
            use_container_width=True
        )

        st.info("""
        Higher positive correlation means that the feature
        tends to increase along with the diabetes outcome,
        while negative correlation indicates an inverse
        relationship.
        """)

    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------

    with tab4:

        st.subheader(
            "Selected Features"
        )

        selected_table = pd.DataFrame({
            "Selected Feature":
                SELECTED_FEATURES
        })

        st.dataframe(
            selected_table,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Feature Count Comparison"
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

    st.title(
        "📊 Comparative Model Evaluation"
    )

    st.write("""
    Each algorithm is evaluated using both the complete
    feature set and the selected feature set.
    """)

    display_df = results_df.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Candidate Score"
    ]

    for col in metric_columns:

        display_df[col] = (
            display_df[col] * 100
        ).round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🏆 Best Model")

    best = results_df.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Algorithm",
        best["Algorithm"]
    )

    c2.metric(
        "Feature Set",
        best["Feature Set"]
    )

    c3.metric(
        "Accuracy",
        f"{best['Accuracy'] * 100:.2f}%"
    )

    c4.metric(
        "ROC-AUC",
        f"{best['ROC-AUC'] * 100:.2f}%"
    )

    st.success(
        f"""
        Based on the Candidate Score, the best performing
        experiment is **{best['Algorithm']} using
        {best['Feature Set']}**.
        """
    )

    st.divider()

    st.subheader(
        "Compare Models"
    )

    metric = st.selectbox(
        "Select metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    )

    chart_df = (
        results_df[
            [
                "Algorithm",
                "Feature Set",
                metric
            ]
        ]
        .copy()
    )

    chart_df["Experiment"] = (
        chart_df["Algorithm"]
        + " - "
        + chart_df["Feature Set"]
    )

    chart_df = chart_df.set_index(
        "Experiment"
    )[[metric]]

    st.bar_chart(
        chart_df
    )

    st.divider()

    st.subheader(
        "Performance Comparison"
    )

    heatmap_table = results_df[
        [
            "Algorithm",
            "Feature Set",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    ].copy()

    heatmap_table[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    ] = (
        heatmap_table[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ]
        ] * 100
    ).round(2)

    st.dataframe(
        heatmap_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DIABETES PREDICTION
# ============================================================

elif page == "🧪 Diabetes Prediction":

    st.title(
        "🧪 Diabetes Prediction"
    )

    st.write("""
    Enter the patient information below to generate
    a prediction using the selected machine learning model.
    """)

    st.divider()

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

    col1, col2 = st.columns(2)

    with col1:

        model_name = st.selectbox(
            "Prediction Model",
            [
                "Random Forest",
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "SVM",
                "Gaussian Naive Bayes"
            ]
        )

    with col2:

        feature_set = st.selectbox(
            "Feature Set",
            [
                "All Features",
                "Selected Features"
            ]
        )

    st.divider()

    if st.button(
        "🔍 Predict Diabetes",
        type="primary",
        use_container_width=True
    ):

        key = (
            feature_set +
            " | " +
            model_name
        )

        model_info = trained_models[key]

        pipeline = model_info["model"]

        model_features = (
            model_info["features"]
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

        # Replace zero with NaN
        for col in ZERO_AS_MISSING:

            if input_data[col].iloc[0] == 0:

                input_data[col] = np.nan

        # Median imputation
        for col in ZERO_AS_MISSING:

            if input_data[col].isna().any():

                input_data[col] = (
                    input_data[col]
                    .fillna(
                        df_model[col].median()
                    )
                )

        X_input = (
            input_data[
                model_features
            ]
        )

        prediction = (
            pipeline.predict(
                X_input
            )[0]
        )

        probability = (
            pipeline.predict_proba(
                X_input
            )[0][1]
        )

        st.divider()

        if prediction == 1:

            st.error(
                "⚠️ Diabetes Predicted"
            )

        else:

            st.success(
                "✅ Diabetes Not Predicted"
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Prediction",
            "Diabetic"
            if prediction == 1
            else "Non-Diabetic"
        )

        c2.metric(
            "Probability",
            f"{probability * 100:.2f}%"
        )

        c3.metric(
            "Model",
            model_name
        )

        st.progress(
            float(probability)
        )

        st.info("""
        ⚠️ This prediction is intended for academic
        demonstration purposes only and should not be
        used as a medical diagnosis.
        """)


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About Project":

    st.title(
        "ℹ️ About the Project"
    )

    st.write("""
    ### Diabetes Prediction Using Predictive Analytics

    This project uses machine learning to predict diabetes
    outcomes from clinical and demographic features.

    ### Analytical Workflow

    **1. Dataset Understanding**

    Understanding the variables, data types and target variable.

    **2. Data Preprocessing**

    Invalid zero values are treated as missing values and
    median imputation is applied.

    **3. Exploratory Data Analysis**

    Descriptive statistics and target distribution are examined.

    **4. Correlation Analysis**

    Relationships between input variables and diabetes outcome
    are analysed.

    **5. Multicollinearity Analysis**

    Feature relationships are evaluated before feature selection.

    **6. Feature Selection**

    A reduced feature set is compared against the complete
    feature set.

    **7. Feature Scaling**

    StandardScaler is applied before model training.

    **8. Model Development**

    Multiple classification algorithms are trained.

    **9. Comparative Evaluation**

    Models are compared using:

    • Accuracy  
    • Precision  
    • Recall  
    • F1 Score  
    • ROC-AUC

    **10. Final Model Selection**

    The highest Candidate Score is used to identify the
    strongest model/feature-set combination.
    """)

    st.divider()

    st.subheader(
        "Input Features"
    )

    feature_table = pd.DataFrame({

        "Feature": FEATURES,

        "Role": [
            "Input",
            "Input",
            "Input",
            "Input",
            "Input",
            "Input",
            "Input",
            "Input"
        ]
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.warning("""
    This application is an academic predictive analytics
    project and is not intended for clinical diagnosis.
    """)
