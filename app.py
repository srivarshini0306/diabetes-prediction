# ==========================================================
# DIABETES PREDICTION - STREAMLIT APPLICATION
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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
    roc_auc_score
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }

    .small-text {
        color: #666666;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    '<div class="main-title">🩺 Diabetes Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict diabetes likelihood using machine learning models'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("diabetes.csv")

    return df


try:
    df = load_data()

except Exception as e:

    st.error(
        "Unable to load diabetes.csv. "
        "Make sure the file is present in the root of your GitHub repository."
    )

    st.stop()


# ==========================================================
# DATASET CONFIGURATION
# ==========================================================

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


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"The following required columns are missing from diabetes.csv: "
        f"{missing_columns}"
    )

    st.stop()


# ==========================================================
# CONVERT DATA TO NUMERIC
# ==========================================================

for column in required_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Prediction",
        "📊 Dataset Overview",
        "📈 Model Comparison",
        "ℹ️ About Project"
    ]
)


# ==========================================================
# PREPARE X AND Y
# ==========================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()


# ==========================================================
# TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# DEFINE MODELS
# ==========================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]),


    "Decision Tree": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "model",
            DecisionTreeClassifier(
                random_state=42
            )
        )
    ]),


    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]),


    "KNN": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),


    "SVM": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            SVC(
                probability=True,
                random_state=42
            )
        )
    ]),


    "Naive Bayes": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            GaussianNB()
        )
    ])
}


# ==========================================================
# TRAIN MODELS AND COMPARE
# ==========================================================

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):

    trained_models = {}

    results = []

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(X_test)

        y_probability = model.predict_proba(X_test)[:, 1]

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

        auc = roc_auc_score(
            y_test,
            y_probability
        )

        results.append({

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "ROC-AUC": auc

        })

        trained_models[name] = model

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    ).reset_index(drop=True)

    return trained_models, results_df


trained_models, results_df = train_models(
    X_train,
    X_test,
    y_train,
    y_test
)


# ==========================================================
# BEST MODEL
# ==========================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

best_accuracy = results_df.iloc[0]["Accuracy"]


# ==========================================================
# PAGE 1 - PREDICTION
# ==========================================================

if page == "🏠 Prediction":

    st.header("🔮 Diabetes Risk Prediction")

    st.write(
        "Enter the patient's clinical measurements below."
    )

    st.info(
        f"Final selected model: **{best_model_name}** "
        f"with test accuracy of **{best_accuracy * 100:.2f}%**"
    )

    st.divider()

    col1, col2 = st.columns(2)


    # ------------------------------------------------------
    # LEFT COLUMN
    # ------------------------------------------------------

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=300.0,
            value=120.0,
            step=1.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0
        )


    # ------------------------------------------------------
    # RIGHT COLUMN
    # ------------------------------------------------------

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=900.0,
            value=80.0,
            step=1.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.47,
            step=0.01
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            step=1
        )


    st.divider()


    # ======================================================
    # PREDICTION BUTTON
    # ======================================================

    if st.button(
        "🔮 Predict Diabetes",
        use_container_width=True
    ):

        # --------------------------------------------------
        # IMPORTANT:
        # Create DataFrame with EXACT same feature names
        # and EXACT same order as training data.
        # --------------------------------------------------

        X_input = pd.DataFrame({

            "Pregnancies": [pregnancies],

            "Glucose": [glucose],

            "BloodPressure": [blood_pressure],

            "SkinThickness": [skin_thickness],

            "Insulin": [insulin],

            "BMI": [bmi],

            "DiabetesPedigreeFunction": [dpf],

            "Age": [age]

        })


        # Make absolutely sure the order matches training

        X_input = X_input[FEATURES]


        # Replace invalid values

        X_input = X_input.replace(
            [np.inf, -np.inf],
            np.nan
        )


        # --------------------------------------------------
        # PREDICT
        # --------------------------------------------------

        try:

            prediction = best_model.predict(
                X_input
            )[0]

            probability = best_model.predict_proba(
                X_input
            )[0][1]


            st.divider()


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if prediction == 1:

                st.error(
                    "⚠️ Higher likelihood of diabetes"
                )

                st.write(
                    "The model predicts that the patient "
                    "is more likely to have diabetes."
                )

            else:

                st.success(
                    "✅ Lower likelihood of diabetes"
                )

                st.write(
                    "The model predicts that the patient "
                    "is less likely to have diabetes."
                )


            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            st.metric(
                "Predicted Diabetes Probability",
                f"{probability * 100:.2f}%"
            )


            st.progress(
                float(probability)
            )


            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            with st.expander("View Input Data"):

                st.dataframe(
                    X_input,
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(e)
            )

            st.write(
                "Input shape:",
                X_input.shape
            )

            st.write(
                "Input columns:",
                list(X_input.columns)
            )


# ==========================================================
# PAGE 2 - DATASET OVERVIEW
# ==========================================================

elif page == "📊 Dataset Overview":

    st.header("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Records",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Input Features",
            len(FEATURES)
        )

    with col3:

        st.metric(
            "Target Variable",
            "Outcome"
        )


    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    st.subheader("Dataset Statistics")

    st.dataframe(
        df[FEATURES].describe().round(2),
        use_container_width=True
    )


    st.subheader("Target Distribution")

    target_counts = df[TARGET].value_counts()

    target_table = pd.DataFrame({

        "Outcome": target_counts.index,

        "Count": target_counts.values

    })

    st.bar_chart(
        target_table.set_index("Outcome")
    )


# ==========================================================
# PAGE 3 - MODEL COMPARISON
# ==========================================================

elif page == "📈 Model Comparison":

    st.header("📈 Comparative Study of Machine Learning Models")

    st.write(
        "Six classification algorithms were trained and evaluated "
        "using the same train-test split."
    )


    st.divider()


    # ------------------------------------------------------
    # RESULTS TABLE
    # ------------------------------------------------------

    display_results = results_df.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    for column in metric_columns:

        display_results[column] = (
            display_results[column] * 100
        ).round(2)


    display_results = display_results.rename(
        columns={
            "Accuracy": "Accuracy (%)",
            "Precision": "Precision (%)",
            "Recall": "Recall (%)",
            "F1 Score": "F1 Score (%)",
            "ROC-AUC": "ROC-AUC (%)"
        }
    )


    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


    # ------------------------------------------------------
    # BEST MODEL
    # ------------------------------------------------------

    st.success(
        f"🏆 Best Model: **{best_model_name}**"
    )

    st.write(
        f"The best-performing model based on test accuracy "
        f"achieved **{best_accuracy * 100:.2f}% accuracy**."
    )


    # ------------------------------------------------------
    # ACCURACY CHART
    # ------------------------------------------------------

    st.subheader("Model Accuracy Comparison")

    chart_df = results_df[
        ["Model", "Accuracy"]
    ].copy()

    chart_df["Accuracy"] = (
        chart_df["Accuracy"] * 100
    )

    chart_df = chart_df.set_index("Model")

    st.bar_chart(
        chart_df
    )


    # ------------------------------------------------------
    # FEATURE INFORMATION
    # ------------------------------------------------------

    st.subheader("Features Used for Final Model")

    feature_table = pd.DataFrame({
        "Feature": FEATURES
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# PAGE 4 - ABOUT PROJECT
# ==========================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About the Project")

    st.write(
        """
        This project implements a machine learning based diabetes
        prediction system using clinical and demographic features.
        """
    )


    st.subheader("🎯 Objective")

    st.write(
        """
        The objective is to predict whether a patient is likely
        to have diabetes based on the provided medical attributes.
        """
    )


    st.subheader("📋 Input Features")

    for feature in FEATURES:

        st.write(f"• {feature}")


    st.subheader("🤖 Machine Learning Models")

    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "SVM",
        "Naive Bayes"
    ]

    for model_name in model_names:

        st.write(f"• {model_name}")


    st.subheader("🔧 Preprocessing")

    st.write(
        """
        • Missing values are handled using median imputation.

        • StandardScaler is applied to models that benefit from
        feature scaling.

        • The dataset is divided into training and testing sets.

        • All models are evaluated using the same test set.
        """
    )


    st.subheader("📊 Evaluation Metrics")

    st.write(
        """
        The models are compared using:

        • Accuracy

        • Precision

        • Recall

        • F1 Score

        • ROC-AUC
        """
    )


    st.subheader("🏆 Model Selection")

    st.write(
        f"""
        Based on the current dataset and evaluation procedure,
        **{best_model_name}** achieved the highest test accuracy
        among the evaluated models.

        The selected model is then used by the prediction interface.
        """
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#777777;font-size:13px;">
    Diabetes Prediction System | Machine Learning Predictive Analytics Project
    </div>
    """,
    unsafe_allow_html=True
)
