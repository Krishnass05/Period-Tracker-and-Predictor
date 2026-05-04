import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import json
import os
from io import StringIO

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# File to save data locally (privacy-first)
DATA_FILE = "menstrual_cycle_dataset.csv"
ML_DATA_FILE = "expanded_fertility_dataset.csv"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return pd.DataFrame(columns=["start_date", "end_date", "cycle_length", "symptoms"])

        try:
            return pd.DataFrame(json.loads(content))
        except json.JSONDecodeError:
            try:
                df = pd.read_csv(StringIO(content), parse_dates=["start_date", "end_date"])
            except (pd.errors.EmptyDataError, ValueError):
                return pd.DataFrame(columns=["start_date", "end_date", "cycle_length", "symptoms"])

            if "symptoms" in df.columns:
                df["symptoms"] = df["symptoms"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x
                )
            return df

    return pd.DataFrame(columns=["start_date", "end_date", "cycle_length", "symptoms"])


def save_data(df):
    df.to_json(DATA_FILE, orient="records", date_format="iso")


def load_ml_data():
    if os.path.exists(ML_DATA_FILE):
        df = pd.read_csv(ML_DATA_FILE, na_values=["?", "NA", "N/A"])
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # Create the requested target columns if they can be derived from existing dataset fields.
        if "Next_Cycle_Length" not in df.columns and "LengthofCycle" in df.columns:
            df["Next_Cycle_Length"] = df["LengthofCycle"]

        if "Irregular" not in df.columns:
            if "CycleWithPeakorNot" in df.columns:
                # Interpret 'CycleWithPeakorNot' as irregular when there is no clear peak.
                df["Irregular"] = df["CycleWithPeakorNot"].fillna(0).astype(int).apply(lambda x: 1 if x == 0 else 0)
            elif "UnusualBleeding" in df.columns:
                df["Irregular"] = df["UnusualBleeding"].fillna(0).astype(int)

        return df
    return pd.DataFrame()


def _is_classifier_target(y):
    return y.dtype == object or y.nunique(dropna=True) <= 10


def prepare_ml_data(df, target, features):
    df = df[[target] + features].copy()
    df = df.dropna(subset=[target])
    if df.empty:
        return None, None, None, None

    X = df[features].copy()
    y = df[target].copy()

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    if numeric_features:
        X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
    if categorical_features:
        X[categorical_features] = X[categorical_features].fillna("missing")

    X = pd.get_dummies(X, dummy_na=False)

    if _is_classifier_target(y):
        y = y.astype(str)
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        return X, y, True, encoder

    return X, y.astype(float), False, None


def build_input_dataframe(df, features):
    row = {}
    for col in features:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            min_val = float(series.min(skipna=True)) if pd.notna(series.min(skipna=True)) else 0.0
            max_val = float(series.max(skipna=True)) if pd.notna(series.max(skipna=True)) else min_val + 100.0
            default = float(series.median(skipna=True)) if pd.notna(series.median(skipna=True)) else 0.0

            if pd.api.types.is_integer_dtype(series.dropna()):
                row[col] = st.number_input(col, value=int(default), min_value=int(min_val), max_value=int(max_val), step=1)
            else:
                row[col] = st.number_input(col, value=default, min_value=min_val, max_value=max_val, step=0.1, format="%.2f")
        else:
            options = series.dropna().astype(str).unique().tolist()
            if options:
                row[col] = st.selectbox(col, options)
            else:
                row[col] = st.text_input(col, value="")

    return pd.DataFrame([row])

# Load or initialize data
df = load_data()

st.set_page_config(page_title="🌸 My Cycle Tracker", layout="wide")

st.title("🌸 My Cycle Tracker")
st.markdown("Track your periods, answer a quick wellness question, and get personalized cycle insights.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 Start", "📊 Predictions & Insights", "📜 History", "🩺 Symptoms & More", "🧪 ML Predictor"])

with tab1:
    st.markdown(
        """
        <div class='page-background'>
            <div class='top-nav'>
                <span>← Back</span>
                <span>Skip</span>
            </div>
            <div class='question-card'>
                <h1>When do you feel best in your body?</h1>
                <p>This helps personalize your cycle insights and track how your body changes.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    options = ["Before my period", "During my period", "After my period", "I don't know"]
    choice = st.radio("When do you feel best in your body?", options, index=3)
    if st.button("Continue"):
        st.success(f"Great, we'll use '{choice}' to personalize your experience.")

    st.markdown("---")
    st.subheader("Log a New Period")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("First day of your period", datetime.today().date())
    with col2:
        end_date = st.date_input("Last day of your period (optional)", value=None)
    
    symptoms_list = ["Cramps", "Headache", "Mood swings", "Fatigue", "Bloating", "Breast tenderness", "Acne", "Nausea"]
    selected_symptoms = st.multiselect("Symptoms during this period", symptoms_list)
    
    if st.button("Save Period"):
        if not df.empty and start_date <= pd.to_datetime(df["start_date"]).max().date():
            st.error("This date is in the past or overlapping. Please check.")
        else:
            cycle_length = None
            if not df.empty:
                last_start = pd.to_datetime(df["start_date"].iloc[-1]).date()
                cycle_length = (start_date - last_start).days
            
            new_row = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat() if end_date else None,
                "cycle_length": cycle_length,
                "symptoms": selected_symptoms
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_data(df)
            st.success("Period logged successfully! 🎉")

with tab2:
    st.subheader("Predictions")
    if len(df) < 2:
        st.info("Log at least 2 periods to see accurate predictions.")
    else:
        # Calculate average cycle length (last 6 cycles or all)
        cycles = df["cycle_length"].dropna().tail(6)
        avg_cycle = int(cycles.mean()) if not cycles.empty else 28
        
        last_start = pd.to_datetime(df["start_date"].iloc[-1]).date()
        
        next_period = last_start + timedelta(days=avg_cycle)
        ovulation = next_period - timedelta(days=14)
        fertile_start = ovulation - timedelta(days=5)
        fertile_end = ovulation + timedelta(days=1)
        
        st.metric("Average Cycle Length", f"{avg_cycle} days")
        st.metric("Next Period Expected", next_period.strftime("%B %d, %Y"))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"**Ovulation** ≈ {ovulation.strftime('%B %d')}")
        with col2:
            st.info(f"**Fertile Window** {fertile_start.strftime('%b %d')} - {fertile_end.strftime('%b %d')}")
        with col3:
            st.warning("High chance of pregnancy in fertile window")
        
        # Simple chart
        df["start_date"] = pd.to_datetime(df["start_date"])
        fig = px.scatter(df, x="start_date", y="cycle_length", 
                         title="Your Cycle Length Trend",
                         labels={"start_date": "Period Start Date", "cycle_length": "Cycle Length (days)"})
        fig.add_hline(y=avg_cycle, line_dash="dash", annotation_text="Average")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Period History")
    if df.empty:
        st.info("No periods logged yet.")
    else:
        display_df = df.copy()
        display_df["start_date"] = pd.to_datetime(display_df["start_date"]).dt.strftime("%Y-%m-%d")
        display_df["end_date"] = pd.to_datetime(display_df["end_date"]).dt.strftime("%Y-%m-%d") if not display_df["end_date"].isna().all() else None
        st.dataframe(display_df[["start_date", "end_date", "cycle_length", "symptoms"]], use_container_width=True)

with tab4:
    st.subheader("Symptom Insights")
    if not df.empty and "symptoms" in df.columns:
        all_symptoms = [sym for sublist in df["symptoms"].dropna() for sym in sublist]
        if all_symptoms:
            symptom_df = pd.DataFrame(all_symptoms, columns=["symptom"])
            counts = symptom_df["symptom"].value_counts()
            fig = px.bar(counts, title="Most Common Symptoms")
            st.plotly_chart(fig, use_container_width=True)
    
    st.info("Tip: Track consistently for better predictions. Consider adding flow intensity, mood, or temperature next.")

with tab5:
    st.subheader("Random Forest Predictor")
    if not SKLEARN_AVAILABLE:
        st.error("The scikit-learn package is required for this tab. Install it with `pip install scikit-learn`.")
    else:
        ml_df = load_ml_data()
        if ml_df.empty:
            st.warning(f"ML dataset not found: {ML_DATA_FILE}")
        else:
            st.markdown("Use this tab to train a machine learning model on the fertility dataset and make predictions from user inputs.")
            target_options = [c for c in ["Next_Cycle_Length", "Irregular"] if c in ml_df.columns]
            if not target_options:
                st.error("The dataset does not contain the required target columns: Next_Cycle_Length and Irregular.")
                st.stop()
            target_column = st.selectbox("Choose target variable", target_options, index=0)

            all_features = [c for c in ml_df.columns if c != target_column and c not in ["ClientID", "Methoddate"]]
            default_features = [c for c in ["Age", "LengthofCycle", "LengthofMenses", "TotalDaysofFertility", "BMI"] if c in all_features]
            selected_features = st.multiselect("Select input features", all_features, default=default_features)

            if not selected_features:
                st.info("Select at least one input feature to train the model.")
            else:
                ml_train = ml_df[[target_column] + selected_features].copy()
                st.write("Training dataset shape:", ml_train.shape)
                X, y, is_classifier, target_encoder = prepare_ml_data(ml_df, target_column, selected_features)

                if X is None or X.empty:
                    st.error("Not enough data to train the model. Try a different target or reduce missing values.")
                else:
                    available_models = ["Random Forest Classifier", "Logistic Regression"] if is_classifier else ["Random Forest Regressor", "Linear Regression"]
                    selected_model = st.selectbox("Choose model", available_models)
                    test_size = st.slider("Test set size (%)", min_value=10, max_value=40, value=20)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

                    if selected_model == "Random Forest Classifier":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    elif selected_model == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000, random_state=42)
                    elif selected_model == "Random Forest Regressor":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    else:
                        model = LinearRegression()

                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    metric_name = "Accuracy" if is_classifier else "R²"
                    st.metric(metric_name, f"{score:.3f}")

                    with st.expander("Enter values for prediction"):
                        input_df = build_input_dataframe(ml_df, selected_features)

                    if st.button("Predict"):
                        input_encoded = pd.get_dummies(input_df, dummy_na=False)
                        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
                        prediction = model.predict(input_encoded)
                        if is_classifier and target_encoder is not None:
                            predicted_label = target_encoder.inverse_transform(prediction.astype(int))
                            st.success(f"Predicted target: {predicted_label[0]}")
                            if hasattr(model, "predict_proba"):
                                probs = model.predict_proba(input_encoded)[0]
                                st.write(pd.DataFrame({"class": target_encoder.inverse_transform(model.classes_), "probability": probs}))
                        else:
                            st.success(f"Predicted target: {prediction[0]:.3f}")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This is a privacy-first tracker. All data stays on your device.")
st.sidebar.markdown("**How predictions work:**\\n- Average of recent cycles\\n- Ovulation ~14 days before next period\\n- Fertile window: 5 days before + 1 day after ovulation")

st.sidebar.caption("Built with ❤️ in Python + Streamlit")

