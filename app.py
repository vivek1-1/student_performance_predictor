import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Load model and data
# -----------------------------


@st.cache_resource
def load_model():
    path = os.path.join("data", "student_performance_model.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    csv_path = os.path.join("data", "student_performance_data.csv")
    df = pd.read_csv(csv_path)
    return df



model = load_model()
df = load_data()


st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("🎓 Student Performance Prediction & Analytics System")

st.markdown(
    """
This app uses a **Random Forest model** trained on student data  
(Age, Study Time, Absences, Tutoring, Parental Support, etc.)  
to predict the **Grade Class** of a student and provides basic analytics.
"""
)

# -----------------------------
# 2. Sidebar: User input form
# -----------------------------
st.sidebar.header("Enter Student Details")

# NOTE: all of these must match the columns used when training X:
# ['Age','Gender','Ethnicity','ParentalEducation','StudyTimeWeekly',
#  'Absences','Tutoring','ParentalSupport','Extracurricular','Sports',
#  'Music','Volunteering','GPA']

age = st.sidebar.slider("Age (15–18)", 15, 18, 16)

gender_label = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender_map = {"Male": 0, "Female": 1}
gender = gender_map[gender_label]

ethnicity_label = st.sidebar.selectbox(
    "Ethnicity",
    ["Caucasian", "African American", "Asian", "Other"]
)
ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
ethnicity = ethnicity_map[ethnicity_label]

parent_edu_label = st.sidebar.selectbox(
    "Parental Education",
    ["None", "High School", "College", "Bachelor's", "Higher"]
)
parent_edu_map = {
    "None": 0,
    "High School": 1,
    "College": 2,
    "Bachelor's": 3,
    "Higher": 4,
}
parent_edu = parent_edu_map[parent_edu_label]

study_time = st.sidebar.slider("Study Time Weekly (hours)", 0.0, 20.0, 8.0, 0.5)

absences = st.sidebar.slider("Absences (per year)", 0, 50, 10)

tutoring_label = st.sidebar.selectbox("Tutoring", ["No", "Yes"])
binary_map = {"No": 0, "Yes": 1}
tutoring = binary_map[tutoring_label]

parent_support_label = st.sidebar.selectbox(
    "Parental Support", ["None", "Low", "Moderate", "High", "Very High"]
)
parent_support_map = {
    "None": 0,
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Very High": 4,
}
parent_support = parent_support_map[parent_support_label]

extra_label = st.sidebar.selectbox("Extracurricular Activities", ["No", "Yes"])
extra = binary_map[extra_label]

sports_label = st.sidebar.selectbox("Sports Participation", ["No", "Yes"])
sports = binary_map[sports_label]

music_label = st.sidebar.selectbox("Music Involvement", ["No", "Yes"])
music = binary_map[music_label]

vol_label = st.sidebar.selectbox("Volunteering", ["No", "Yes"])
volunteering = binary_map[vol_label]

gpa = st.sidebar.slider("Current GPA (2.0 – 4.0)", 2.0, 4.0, 3.0, 0.01)

# Build single-row input dataframe in the SAME column order as training X
input_data = pd.DataFrame(
    {
        "Age": [age],
        "Gender": [gender],
        "Ethnicity": [ethnicity],
        "ParentalEducation": [parent_edu],
        "StudyTimeWeekly": [study_time],
        "Absences": [absences],
        "Tutoring": [tutoring],
        "ParentalSupport": [parent_support],
        "Extracurricular": [extra],
        "Sports": [sports],
        "Music": [music],
        "Volunteering": [volunteering],
        "GPA": [gpa],
    }
)

st.subheader("📥 Input Features Preview")
st.write(input_data)

# -----------------------------
# 3. Prediction
# -----------------------------
st.subheader("🎯 Prediction")

if st.button("Predict Grade Class"):
    pred_class = model.predict(input_data)[0]
    # probability of each class
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
    else:
        proba = None

    st.success(f"Predicted Grade Class: **{pred_class}**")

    if proba is not None:
        proba_df = pd.DataFrame(
            [proba],
            columns=[f"Class {c}" for c in np.unique(df["GradeClass"])]
        )
        st.write("Prediction probabilities:")
        st.dataframe(proba_df)

    st.caption(
        "Note: GradeClass is the label from the dataset (e.g., 1=Low, 2=Average, 3=Good, 4=Excellent – "
        "use your report to define exact meaning)."
    )

# -----------------------------
# 4. Analytics Section
# -----------------------------
st.markdown("---")
st.header("📊 Dataset Analytics")

tab1, tab2, tab3 = st.tabs(["GradeClass Distribution", "GPA vs GradeClass", "Correlation Heatmap"])

with tab1:
    st.subheader("GradeClass Distribution")
    grade_counts = df["GradeClass"].value_counts().sort_index()
    st.bar_chart(grade_counts)

with tab2:
    st.subheader("GPA by GradeClass")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="GradeClass", y="GPA", data=df, ax=ax)
    ax.set_title("GPA Distribution for Each GradeClass")
    st.pyplot(fig)

with tab3:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.markdown("---")
st.caption("Built with Streamlit, RandomForest, and your Student Performance dataset.")
