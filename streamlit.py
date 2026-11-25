import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction | Predictive Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Your repository already contains these files
DATA_FILE = BASE_DIR / "diabetes.csv"
CLEANED_FILE = BASE_DIR / "diabetes_cleaned.csv"
SCALED_FILE = BASE_DIR / "diabetes_scaled.csv"

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    color: #b91c1c;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #64748b;
    margin-bottom: 30px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #991b1b;
    margin-top: 20px;
}

.info-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #fff7ed;
    border-left: 5px solid #dc2626;
    margin-bottom: 20px;
}

.metric-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    if CLEANED_FILE.exists():
        df = pd.read_csv(CLEANED_FILE)
    elif DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
    else:
        return None

    return df


df = load_data()


# ---------------------------------------------------------
# CHECK DATA
# ---------------------------------------------------------

if df is None:

    st.error(
        "❌ Dataset file not found.\n\n"
        "Make sure `diabetes.csv` or `diabetes_cleaned.csv` "
        "is present in the same GitHub repository as `app.py`."
    )

    st.stop()


# ---------------------------------------------------------
# IDENTIFY TARGET
# ---------------------------------------------------------

target_candidates = [
    "Outcome",
    "outcome",
    "Diabetes",
    "diabetes",
    "target",
    "Target"
]

target_variable = None

for col in target_candidates:
    if col in df.columns:
        target_variable = col
        break

if target_variable is None:

    st.error(
        "Target variable could not be identified. "
        "Expected a column such as `Outcome`."
    )

    st.stop()


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.title("🩺 Diabetes Prediction")

st.sidebar.markdown(
    """
### Predictive Analytics Project

Navigate through the complete machine learning workflow.
"""
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Project Overview",
        "📊 Dataset Understanding",
        "🔍 Exploratory Data Analysis",
        "🧹 Data Preprocessing",
        "🎯 Feature Selection",
        "🤖 Model Comparison",
        "🩺 Diabetes Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This application demonstrates the complete "
    "predictive analytics workflow for diabetes prediction."
)


# =========================================================
# 1. PROJECT OVERVIEW
# =========================================================

if page == "🏠 Project Overview":

    st.markdown(
        '<div class="title">🩺 Diabetes Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'A Predictive Analytics & Machine Learning Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">

    <b>Project Introduction</b><br><br>

    This project focuses on predicting whether a person is likely to
    have diabetes using machine learning techniques.

    The project follows a complete predictive analytics workflow,
    starting from understanding the dataset and exploratory data
    analysis to preprocessing, feature selection, model development,
    comparison and final prediction.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Project Objectives")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - Understand the diabetes dataset
        - Identify missing and invalid values
        - Perform exploratory data analysis
        - Study relationships between variables
        - Detect multicollinearity
        """)

    with col2:
        st.markdown("""
        - Perform feature selection
        - Build multiple classification models
        - Compare model performance
        - Identify the best-performing model
        - Develop an interactive prediction interface
        """)

    st.markdown("---")

    st.markdown("### 🔄 Predictive Analytics Workflow")

    st.markdown("""
    **Dataset Understanding**
    ↓

    **Data Cleaning & Preprocessing**
    ↓

    **Exploratory Data Analysis**
    ↓

    **Correlation & Multicollinearity Analysis**
    ↓

    **Feature Engineering & Selection**
    ↓

    **Model Development**
    ↓

    **Comparative Model Evaluation**
    ↓

    **Best Model Selection**
    ↓

    **Diabetes Prediction**
    """)

    st.markdown("---")

    st.markdown("### 📌 Dataset Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Records", df.shape[0])

    with c2:
        st.metric("Features", df.shape[1] - 1)

    with c3:
        st.metric("Target Variable", target_variable)


# =========================================================
# 2. DATASET UNDERSTANDING
# =========================================================

elif page == "📊 Dataset Understanding":

    st.markdown(
        '<div class="section-title">📊 Dataset Understanding</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This section provides an overview of the dataset used "
        "for diabetes prediction."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric(
            "Numerical Columns",
            df.select_dtypes(include=np.number).shape[1]
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            df.duplicated().sum()
        )

    st.markdown("### First Five Records")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.markdown("### Last Five Records")

    st.dataframe(
        df.tail(),
        use_container_width=True
    )

    st.markdown("### Dataset Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.markdown("### Statistical Summary")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    st.markdown("### Target Distribution")

    target_counts = df[target_variable].value_counts()

    target_df = pd.DataFrame({
        "Outcome": target_counts.index.astype(str),
        "Count": target_counts.values
    })

    st.bar_chart(
        target_df.set_index("Outcome")
    )


# =========================================================
# 3. EDA
# =========================================================

elif page == "🔍 Exploratory Data Analysis":

    st.markdown(
        '<div class="section-title">🔍 Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "EDA is performed to understand the distribution of variables, "
        "relationships between features and patterns associated with "
        "diabetes outcomes."
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    selected_column = st.selectbox(
        "Select a numerical feature",
        numeric_columns
    )

    st.markdown(
        f"### Distribution of `{selected_column}`"
    )

    hist_data = df[selected_column].dropna()

    st.bar_chart(
        hist_data.value_counts().sort_index()
    )

    st.markdown("### Feature Statistics")

    stats = pd.DataFrame({
        "Mean": [df[selected_column].mean()],
        "Median": [df[selected_column].median()],
        "Minimum": [df[selected_column].min()],
        "Maximum": [df[selected_column].max()],
        "Std Dev": [df[selected_column].std()]
    })

    st.dataframe(
        stats,
        use_container_width=True
    )

    st.markdown("---")

    st.markdown("### 🔗 Correlation Analysis")

    numeric_df = df.select_dtypes(include=np.number)

    correlation = numeric_df.corr()

    st.dataframe(
        correlation.round(3),
        use_container_width=True
    )

    st.markdown("### Strongest Correlations")

    corr_pairs = (
        correlation
        .where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )

    corr_pairs.columns = [
        "Feature 1",
        "Feature 2",
        "Correlation"
    ]

    corr_pairs["Absolute Correlation"] = (
        corr_pairs["Correlation"].abs()
    )

    corr_pairs = corr_pairs.sort_values(
        "Absolute Correlation",
        ascending=False
    )

    st.dataframe(
        corr_pairs.head(10).round(3),
        use_container_width=True
    )

    st.markdown("---")

    st.markdown("### 📈 Feature vs Diabetes Outcome")

    feature = st.selectbox(
        "Select feature",
        [c for c in numeric_columns if c != target_variable]
    )

    grouped = df.groupby(target_variable)[feature].mean()

    chart_df = pd.DataFrame({
        "Outcome": grouped.index.astype(str),
        "Average": grouped.values
    })

    st.bar_chart(
        chart_df.set_index("Outcome")
    )


# =========================================================
# 4. PREPROCESSING
# =========================================================

elif page == "🧹 Data Preprocessing":

    st.markdown(
        '<div class="section-title">🧹 Data Preprocessing</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Data preprocessing prepares the raw dataset for machine learning.
    This includes identifying missing/invalid values, checking duplicates,
    handling missing values and scaling numerical variables where required.
    """)

    st.markdown("### Missing Values")

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Feature": missing.index,
        "Missing Values": missing.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.markdown("### 🔎 Zero Values")

    feature_columns = [
        c for c in df.columns
        if c != target_variable
    ]

    zero_counts = []

    for col in feature_columns:

        zero_counts.append({
            "Feature": col,
            "Zero Values": (df[col] == 0).sum()
        })

    zero_df = pd.DataFrame(zero_counts)

    st.dataframe(
        zero_df,
        use_container_width=True
    )

    st.markdown("### Duplicate Records")

    st.write(
        f"Number of duplicate records: "
        f"**{df.duplicated().sum()}**"
    )

    st.markdown("---")

    st.markdown("### 📏 Standardization")

    st.write("""
    Standardization transforms numerical variables so that they have
    approximately zero mean and unit variance.

    This is particularly useful for algorithms such as Logistic Regression,
    KNN and other distance/scale-sensitive models.
    """)

    try:

        from sklearn.preprocessing import StandardScaler

        X = df.drop(columns=[target_variable])
        numeric_features = X.select_dtypes(
            include=np.number
        ).columns

        scaler = StandardScaler()

        scaled = scaler.fit_transform(
            X[numeric_features]
        )

        scaled_df = pd.DataFrame(
            scaled,
            columns=numeric_features
        )

        st.dataframe(
            scaled_df.head(),
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Scaling demonstration could not be displayed: {e}"
        )


# =========================================================
# 5. FEATURE SELECTION
# =========================================================

elif page == "🎯 Feature Selection":

    st.markdown(
        '<div class="section-title">🎯 Feature Engineering & Selection</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Feature selection helps identify the variables that provide useful
    information for predicting diabetes while reducing redundant
    information and potential multicollinearity.
    """)

    numeric_df = df.select_dtypes(include=np.number)

    feature_columns = [
        c for c in numeric_df.columns
        if c != target_variable
    ]

    # -----------------------------------------------------
    # CORRELATION WITH TARGET
    # -----------------------------------------------------

    st.markdown("### 1️⃣ Feature Correlation with Target")

    target_corr = (
        numeric_df.corr()[target_variable]
        .drop(target_variable)
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    target_corr_df = pd.DataFrame({
        "Feature": target_corr.index,
        "Correlation with Target": target_corr.values
    })

    st.dataframe(
        target_corr_df.round(4),
        use_container_width=True
    )

    # -----------------------------------------------------
    # MULTICOLLINEARITY
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown("### 2️⃣ Multicollinearity Analysis")

    st.write("""
    Multicollinearity occurs when two or more independent variables
    contain highly similar information.

    Variance Inflation Factor (VIF) is commonly used to identify
    multicollinearity.

    General interpretation:

    - VIF < 5 → generally acceptable
    - VIF 5–10 → possible multicollinearity
    - VIF > 10 → strong multicollinearity
    """)

    # Manual VIF calculation
    # Avoids requiring statsmodels on Streamlit Cloud.

    X_vif = df[feature_columns].copy()

    X_vif = X_vif.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_vif = X_vif.fillna(
        X_vif.median(numeric_only=True)
    )

    vif_results = []

    for i, feature in enumerate(X_vif.columns):

        y = X_vif[feature].values

        other_features = X_vif.drop(
            columns=[feature]
        )

        if other_features.shape[1] == 0:
            vif = 1.0

        else:

            X_other = other_features.values

            X_other = np.column_stack([
                np.ones(X_other.shape[0]),
                X_other
            ])

            try:

                coefficients = np.linalg.lstsq(
                    X_other,
                    y,
                    rcond=None
                )[0]

                prediction = X_other @ coefficients

                ss_res = np.sum(
                    (y - prediction) ** 2
                )

                ss_tot = np.sum(
                    (y - np.mean(y)) ** 2
                )

                if ss_tot == 0:
                    r_squared = 0

                else:
                    r_squared = (
                        1 - ss_res / ss_tot
                    )

                if r_squared >= 0.9999:
                    vif = np.inf

                else:
                    vif = 1 / (1 - r_squared)

            except Exception:
                vif = np.nan

        vif_results.append({
            "Feature": feature,
            "VIF": vif
        })

    vif_df = pd.DataFrame(vif_results)

    vif_df = vif_df.sort_values(
        "VIF",
        ascending=False
    )

    st.dataframe(
        vif_df.round(3),
        use_container_width=True
    )

    # -----------------------------------------------------
    # FEATURE SELECTION
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown("### 3️⃣ Select Features")

    threshold = st.slider(
        "Correlation threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )

    selected_features = target_corr[
        target_corr.abs() >= threshold
    ].index.tolist()

    st.write(
        f"Features selected using |correlation| ≥ {threshold}:"
    )

    if selected_features:

        st.success(
            f"{len(selected_features)} features selected"
        )

        st.write(
            ", ".join(selected_features)
        )

    else:

        st.warning(
            "No features satisfy the selected threshold."
        )

    # -----------------------------------------------------
    # RANDOM FOREST IMPORTANCE
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown("### 4️⃣ Random Forest Feature Importance")

    try:

        from sklearn.ensemble import RandomForestClassifier

        X = df.drop(
            columns=[target_variable]
        )

        y = df[target_variable]

        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        rf.fit(X, y)

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": rf.feature_importances_
        })

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
        )

        st.bar_chart(
            importance_df.set_index(
                "Feature"
            )
        )

        st.dataframe(
            importance_df.round(4),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Feature importance could not be calculated: {e}"
        )


# =========================================================
# 6. MODEL COMPARISON
# =========================================================

elif page == "🤖 Model Comparison":

    st.markdown(
        '<div class="section-title">🤖 Model Development & Comparison</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Multiple classification algorithms are trained and evaluated
    using the same dataset. This allows a comparative study rather
    than selecting a model arbitrarily.
    """)

    try:

        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score
        )

        X = df.drop(
            columns=[target_variable]
        )

        y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        models = {

            "Logistic Regression":
                Pipeline([
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000
                        )
                    )
                ]),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42
                ),

            "KNN":
                Pipeline([
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
                ])
        }

        results = []

        trained_models = {}

        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(
                X_test
            )

            if hasattr(
                model,
                "predict_proba"
            ):

                y_prob = model.predict_proba(
                    X_test
                )[:, 1]

            else:

                y_prob = None

            results.append({

                "Model": name,

                "Accuracy": accuracy_score(
                    y_test,
                    y_pred
                ),

                "Precision": precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

                "Recall": recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

                "F1 Score": f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

                "ROC-AUC": (
                    roc_auc_score(
                        y_test,
                        y_prob
                    )
                    if y_prob is not None
                    else np.nan
                )
            })

            trained_models[name] = model

        results_df = pd.DataFrame(
            results
        )

        results_df = results_df.sort_values(
            "Accuracy",
            ascending=False
        )

        st.markdown("### 📊 Comparative Performance")

        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "ROC-AUC": "{:.4f}"
            }),
            use_container_width=True
        )

        st.markdown("### Model Accuracy")

        accuracy_chart = results_df[
            ["Model", "Accuracy"]
        ].set_index("Model")

        st.bar_chart(
            accuracy_chart
        )

        st.markdown("### Model F1 Score")

        f1_chart = results_df[
            ["Model", "F1 Score"]
        ].set_index("Model")

        st.bar_chart(
            f1_chart
        )

        # Best model based on accuracy
        best_model_name = results_df.iloc[
            0
        ]["Model"]

        best_accuracy = results_df.iloc[
            0
        ]["Accuracy"]

        st.success(
            f"🏆 Best performing model based on accuracy: "
            f"**{best_model_name}** "
            f"with accuracy of **{best_accuracy:.2%}**"
        )

        st.markdown("---")

        st.markdown("### 🌲 Random Forest with All Features")

        st.write("""
        Random Forest is also evaluated using the complete set of
        available predictive features.

        Tree-based ensemble models can often capture nonlinear
        relationships and interactions between variables, which can
        allow all available features to contribute to prediction.
        """)

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )

        rf.fit(
            X_train,
            y_train
        )

        rf_pred = rf.predict(
            X_test
        )

        rf_accuracy = accuracy_score(
            y_test,
            rf_pred
        )

        st.metric(
            "Random Forest Accuracy",
            f"{rf_accuracy:.2%}"
        )

        st.markdown("### Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": rf.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        st.bar_chart(
            importance_df.set_index(
                "Feature"
            )
        )

    except Exception as e:

        st.error(
            f"Model comparison failed: {e}"
        )


# =========================================================
# 7. DIABETES PREDICTION
# =========================================================

elif page == "🩺 Diabetes Prediction":

    st.markdown(
        '<div class="section-title">🩺 Diabetes Prediction</div>',
        unsafe_allow_html=True
    )

    st.write("""
    Enter the patient's clinical measurements below to generate
    a diabetes prediction using a Random Forest classification model.
    """)

    st.info(
        "⚠️ This application is an educational machine-learning "
        "prototype and should not be used as a medical diagnosis."
    )

    # -----------------------------------------------------
    # MODEL TRAINING
    # -----------------------------------------------------

    try:

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer

        X = df.drop(
            columns=[target_variable]
        )

        y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        imputer = SimpleImputer(
            strategy="median"
        )

        X_train_imp = imputer.fit_transform(
            X_train
        )

        X_test_imp = imputer.transform(
            X_test
        )

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )

        model.fit(
            X_train_imp,
            y_train
        )

        # -------------------------------------------------
        # INPUT FORM
        # -------------------------------------------------

        st.markdown("### Enter Patient Information")

        input_values = {}

        cols = st.columns(2)

        for i, feature in enumerate(X.columns):

            col = cols[i % 2]

            with col:

                # Integer-like columns
                if feature.lower() in [
                    "pregnancies",
                    "age"
                ]:

                    input_values[feature] = st.number_input(
                        feature,
                        min_value=0,
                        value=1,
                        step=1
                    )

                else:

                    default_value = float(
                        df[feature].median()
                    )

                    input_values[feature] = st.number_input(
                        feature,
                        value=default_value
                    )

        st.markdown("---")

        predict_button = st.button(
            "🔍 Predict Diabetes",
            type="primary",
            use_container_width=True
        )

        if predict_button:

            input_df = pd.DataFrame(
                [input_values]
            )

            input_array = imputer.transform(
                input_df
            )

            prediction = model.predict(
                input_array
            )[0]

            probability = model.predict_proba(
                input_array
            )[0][1]

            if prediction == 1:

                st.error(
                    "⚠️ Prediction: Higher likelihood of diabetes"
                )

            else:

                st.success(
                    "✅ Prediction: Lower likelihood of diabetes"
                )

            st.metric(
                "Estimated Diabetes Probability",
                f"{probability:.2%}"
            )

            st.progress(
                float(probability)
            )

            st.markdown("### Prediction Interpretation")

            if probability >= 0.5:

                st.write("""
                The model classified this input as **Outcome 1**
                based on the learned patterns in the training data.
                """)

            else:

                st.write("""
                The model classified this input as **Outcome 0**
                based on the learned patterns in the training data.
                """)

            st.caption(
                "This prediction is generated by a machine learning "
                "model and is not a medical diagnosis."
            )

    except Exception as e:

        st.error(
            f"Prediction could not be generated: {e}"
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748b;">
    <b>Diabetes Prediction | Predictive Analytics Project</b><br>
    Developed using Python, Pandas, NumPy, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
