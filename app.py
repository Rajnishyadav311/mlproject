import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Page settings
st.set_page_config(
    page_title="EduTrack ML",
    page_icon="📚",
    layout="wide"
)

# App title
st.title("📚 EduTrack ML")
st.subheader("Student Performance Analyzer")
st.write("Analyze student performance and predict final marks using Machine Learning.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("student_performance.csv")

try:
    df = load_data()

    # Required columns
    required_columns = [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Final_Marks"
    ]

    if not all(col in df.columns for col in required_columns):
        st.error("CSV file mein required columns missing hain.")
        st.stop()

    # Dataset overview
    st.header("📊 Dataset Overview")
    st.write("Total students:", len(df))
    st.dataframe(df.head(10), use_container_width=True)

    # Features and target
    X = df[[
        "Study_Hours",
        "Attendance",
        "Previous_Marks"
    ]]
    y = df["Final_Marks"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.header("🤖 Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error", f"{mae:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")

    # Visualizations
    st.header("📈 Data Visualization")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(df["Final_Marks"], bins=15, edgecolor="black")
        ax.set_title("Distribution of Final Marks")
        ax.set_xlabel("Final Marks")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x="Study_Hours",
            y="Final_Marks",
            ax=ax
        )
        ax.set_title("Study Hours vs Final Marks")
        st.pyplot(fig)
        plt.close(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(
        df[required_columns].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

    # Prediction section
    st.header("🎯 Predict Student Marks")
    st.write("Student ki details enter karke estimated marks dekhein.")

    study_hours = st.slider(
        "Daily Study Hours",
        min_value=0.0,
        max_value=12.0,
        value=4.0,
        step=0.5
    )

    attendance = st.slider(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=75
    )

    previous_marks = st.slider(
        "Previous Marks",
        min_value=0,
        max_value=100,
        value=65
    )

    if st.button("Predict Final Marks"):
        input_data = pd.DataFrame([{
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Previous_Marks": previous_marks
        }])

        prediction = model.predict(input_data)[0]
        prediction = np.clip(prediction, 0, 100)

        st.success(
            f"Estimated Final Marks: {prediction:.2f} / 100"
        )

        st.info(
            "This is an educational estimate based on the sample dataset, "
            "not a guaranteed result."
        )

    # Actual vs predicted
    st.header("📉 Actual vs Predicted Marks")

    result_df = pd.DataFrame({
        "Actual Marks": y_test.values,
        "Predicted Marks": y_pred
    })

    st.dataframe(result_df.head(10), use_container_width=True)

    st.caption(
        "Disclaimer: This demo uses the supplied dataset. "
        "Predictions are estimates and should not be used "
        "for official academic decisions."
    )

except FileNotFoundError:
    st.error(
        "student_performance.csv file nahi mili. "
        "Ensure karein ki CSV aur app.py same repository folder mein hain."
    )

except Exception as e:
    st.error(f"Error: {e}")
