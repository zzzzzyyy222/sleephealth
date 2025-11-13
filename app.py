import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# Page config
st.set_page_config(page_title="Sleep Health Dashboard", layout="wide")

# Load data safely
@st.cache_data
def load_data(path="Sleep_health_and_lifestyle_dataset.csv"):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except FileNotFoundError:
        st.error(" Dataset file not found!")
        st.stop()

df = load_data()

# Sidebar
st.sidebar.title("Sleep Health Dashboard")
page = st.sidebar.radio("Navigate to:", ["About", "Introduction", "EDA", "Prediction"])

with st.sidebar.expander("\U0001F9E9 Filters", expanded=False):
    st.markdown("Use these filters to customize the dataset view across the dashboard.")
    
    sel_gender = st.multiselect("Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
    sel_occupation = st.multiselect("Occupation", options=df["Occupation"].unique(), default=df["Occupation"].unique())
    
    min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    
    sel_quality = st.multiselect("Quality of Sleep", options=df["Quality of Sleep"].unique(), default=df["Quality of Sleep"].unique())


def filter_df(df):
    return df[
        (df["Gender"].isin(sel_gender)) &
        (df["Occupation"].isin(sel_occupation)) &
        (df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) &
        (df["Quality of Sleep"].isin(sel_quality))
    ]

filtered = filter_df(df)

# About Section
if page == "About":
    st.title("\U0001F4D8 Welcome to the Sleep Health & Lifestyle Dashboard!")

    st.markdown("""
This interactive dashboard provides insights into sleep patterns, lifestyle habits, and health indicators using a comprehensive dataset of individuals across various occupations and age groups.

### \U0001F9FE Dataset Overview
- **Number of Records:** 374  
- **Key Features:** Gender, Age, Occupation, Sleep Duration, Stress Level, Physical Activity, BMI, Sleep Disorder  
- **Purpose:** To analyze and visualize sleep health trends and predict sleep disorders


### \U0001F4E5 Download Dataset
[Download Sleep Health Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

### \U0001F4CA Dashboard Features
- **EDA:** Explore sleep metrics by gender, age, occupation, and more  
- **Prediction:** Use machine learning to predict sleep disorders  
- **Filters:** Customize views by gender, age range, occupation, and sleep quality

### \U0001F468 Created By
- Chong Zhe, Lam Pooi Syuen, Hashika  

### \U0001F4CC How to Use
Use the sidebar to navigate between sections and apply filters.  
Visualizations and predictions will update based on your selections.
""")

elif page == "Introduction":
    st.title("\U0001FA84 Introduction")

    image_path = "woman-sleeping-in-bed-at-night-time-royalty-free-image-1580223174.jpg"
    st.image(image_path, caption="", use_container_width=True)

    st.title("\U0001F4CA Understanding Sleep Health Trends")
    st.markdown("""
Sleep is a vital component of overall health, influencing physical, mental, and emotional well-being.  
This dashboard explores sleep patterns, lifestyle habits, and health indicators using a dataset of individuals across various occupations and age groups.
""")

    st.markdown("### \U0001F4F9 Why Do We Need To Sleep?")
    st.video("https://www.youtube.com/watch?v=dqONk48l5vY")

    st.subheader("\U0001F6CC What Is Sleep Health?")
    st.markdown("""
Sleep health refers to the quality, duration, and consistency of sleep that supports optimal functioning.  
Poor sleep can lead to stress, reduced productivity, and long-term health issues such as heart disease, obesity, and depression.
""")

    st.subheader("\U0001F4C8 Trends in Sleep & Lifestyle")
    st.markdown("""
- \u23F0 Shortened sleep duration among high-stress occupations  
- \U0001F9E0 Strong correlation between stress levels and sleep quality  
- \U0001F6B6 Lower physical activity linked to sleep disorders  
- \U0001F4C9 Sleep disorders more common in individuals with high BMI and low activity  
""")

    st.markdown("### \U0001F9E0 Did You Know?")
    st.info("""
- People who sleep less than 6 hours a night are **13% more likely** to die earlier than those sleeping 7–9 hours.  
- \U0001F319 Blue light from screens suppresses melatonin, disrupting your body’s sleep rhythm.  
- \U0001F4A1 Short naps (20–30 minutes) can boost alertness and performance without affecting nighttime sleep.
""")

    # Question
    st.markdown("### \u2753 Quick Question")
    user_answer = st.radio(
        "How many hours of sleep do you usually get per night?",
        ["Less than 5", "5–7", "7–9", "More than 9"]
    )

    if user_answer:
        if user_answer == "Less than 5":
            st.warning("\u26A0\uFE0F That’s considered sleep deprivation — you might experience fatigue or poor focus.")
        elif user_answer == "5–7":
            st.info("\U0001F914 That’s average, but try aiming for 7–9 hours for optimal rest.")
        elif user_answer == "7–9":
            st.success("\u2705 Great! That’s the recommended sleep duration for most adults.")
        else:
            st.warning("\U0001F634 Oversleeping can also affect mood and metabolism — balance is key!")

    #Key Features of This Dashboard
    st.subheader("\U0001F539 Key Features of This Dashboard")
    st.markdown("""
- Interactive filters and visual analytics  
- Correlation and trend exploration  
- Machine learning prediction for sleep disorders  
- Data-driven health insights  
""")


elif page == "EDA":
    st.title("\U0001F4CA Exploratory Data Analysis")
    image_path = "photo-1666875753105-c63a6f3bdc86.jpeg"
    st.image(image_path, caption="", use_container_width=True)

    st.title("\U0001F4CA Understanding Sleep Health Trends")

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Sleep Duration", f"{filtered['Sleep Duration'].mean():.2f}")
    col2.metric("Median Sleep Duration", f"{filtered['Sleep Duration'].median():.2f}")
    col3.metric("Avg Stress Level", f"{filtered['Stress Level'].mean():.2f}")
    col4.metric("Avg Daily Steps", f"{filtered['Daily Steps'].mean():.0f}")

    #1. Lifestyle Balance by Occupation
    st.subheader("\U0001F52C Lifestyle Balance by Occupation (Radar Chart)")

    radar_data = (
    filtered.groupby("Occupation")[["Sleep Duration", "Stress Level", "Physical Activity Level", "Quality of Sleep"]]
    .mean()
    .reset_index()
)

    # Normalize values for fair visual comparison (so all metrics use same scale)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(radar_data.drop("Occupation", axis=1))
    radar_scaled = pd.DataFrame(scaled_values, columns=["Sleep Duration", "Stress Level", "Physical Activity Level", "Quality of Sleep"])
    radar_scaled["Occupation"] = radar_data["Occupation"]

    # Convert into long format for Plotly
    radar_long = radar_scaled.melt(id_vars="Occupation", var_name="Metric", value_name="Value")

    fig_radar = px.line_polar(
    radar_long,
    r="Value",
    theta="Metric",
    color="Occupation",
    line_close=True,
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Lifestyle Balance Across Occupations"
   )
    fig_radar.update_traces(fill='toself', opacity=0.6)

    st.plotly_chart(fig_radar, use_container_width=True)


    st.markdown("""
**Understanding Lifestyle Balance Across Occupations**  
This radar (spider) chart visualizes how different **occupations** perform across four key lifestyle indicators:  
*Sleep Duration*,  *Stress Level*, *Physical Activity*, and *Quality of Sleep*.

Each colored shape represents an occupation’s average profile, helping identify balanced or unhealthy patterns:
- Occupations with **larger overall shapes** indicate **healthier lifestyles**, longer sleep, lower stress, higher activity, and better sleep quality.  
- Jobs in **corporate or healthcare fields** may show **higher stress and lower sleep**, forming narrower radar shapes.  
- Creative and academic roles often appear **more balanced**, maintaining moderate stress and better sleep quality.

This visualization offers a holistic snapshot of how **work-life dynamics** affect sleep health and well-being.
""")

    # 2. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = filtered.select_dtypes(include=np.number).columns.tolist()
    cols_to_remove = ["Person ID", "Person_ID", "ID", "id"]
    numeric_cols = [col for col in numeric_cols if col not in cols_to_remove]

    corr = filtered[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="Spectral", center=0, ax=ax, linewidths=0.5, fmt=".2f")
    st.pyplot(fig)
    st.markdown("""
    **Relationships Between Key Variables**  
    This correlation map visually summarizes how variables interact with each other.  
    A notable observation is the **negative correlation between Stress Level and Sleep Duration**, confirming that individuals under higher stress tend to sleep less.  
    Meanwhile, **Physical Activity Level** shows a mild positive relationship with **Quality of Sleep**, suggesting that even moderate exercise contributes to better rest.  
    These relationships help identify which variables might serve as strong predictors in modeling sleep disorders.
    """)

    # 3. Sleep Disorder Distribution
    st.subheader("Sleep Disorder Distribution")
    disorder_counts = filtered["Sleep Disorder"].value_counts().reset_index()
    disorder_counts.columns = ["Sleep Disorder", "Count"]
    fig4 = px.pie(disorder_counts, names="Sleep Disorder", values="Count", 
                  color="Sleep Disorder", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("""
    **Distribution of Sleep Disorders Among Respondents**  
    The pie chart shows that **Insomnia** is the most common sleep disorder, followed by **Sleep Apnea**.  
    This pattern may reflect modern lifestyle factors such as stress, screen exposure, and irregular work hours.  
    Understanding disorder distribution is crucial for **targeted awareness and intervention programs**, as it indicates where preventive education or treatment accessibility should be prioritized.
    """)

    # 4. Distribution of Sleep Quality (Histogram)
    st.subheader("Distribution of Sleep Quality")
    fig6 = px.histogram(
        filtered, 
        x="Quality of Sleep", 
        nbins=10, 
        color="Gender",  # Differentiate by gender for color contrast
        barmode="overlay",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("""
    **Overall Sleep Quality Levels**  
    This histogram shows that most participants report **moderate sleep quality scores** (around 5–7).  
    Very few participants achieved extremely high quality, indicating that **optimal rest is relatively uncommon** in this dataset.  
    Differences between genders appear minimal, although male participants show a slightly wider variation in reported quality.  
    The findings highlight the need for interventions targeting **consistent sleep routines and stress management** to improve sleep quality.
    """)
    
    # 5. Sleep Duration by Occupation
    st.subheader("Sleep Duration by Occupation")
    fig2 = px.box(filtered, x="Occupation", y="Sleep Duration", color="Occupation", 
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    **Occupational Influence on Sleep Duration**  
    This boxplot highlights how work type affects sleep.  
    Occupations with high mental or physical demands — such as corporate, healthcare, or technical jobs — show shorter and more varied sleep durations.  
    On the other hand, those in creative or academic roles demonstrate more consistent and slightly longer sleep periods.  
    The spread within each occupation also suggests **individual lifestyle differences**, indicating that factors like workload, shift schedules, and stress coping methods heavily influence rest patterns.
    """)
    
    # 6. Sleep Disorder by Age (Stacked Percent Bar)
    st.subheader("Sleep Disorders by Age Group")
    fig7 = px.histogram(filtered, x="Age", color="Sleep Disorder",
                        barnorm='percent', color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown("""
    **Age-Based Prevalence of Sleep Disorders**  
    This chart indicates that sleep disorders are more frequent among **middle-aged and older adults**, where lifestyle stress and health factors accumulate.  
    Younger participants (below 30) tend to show fewer disorders, possibly due to more flexible schedules and faster recovery rates.  
    The pattern suggests that **preventive sleep education and stress awareness** should target working-age adults.
    """)

    # 7. Physical Activity vs Sleep Quality (Box Plot)
    st.subheader("Physical Activity Level and Sleep Quality")
    fig8 = px.box(filtered, x="Physical Activity Level", y="Quality of Sleep",
                  color="Physical Activity Level", color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown("""
    **Effect of Physical Activity on Sleep Quality**  
    This visualization reveals a clear relationship: participants with **higher physical activity** levels generally enjoy **better sleep quality**.  
    The trend highlights the physiological benefits of exercise — such as improved mood regulation, reduced anxiety, and enhanced deep sleep cycles.  
    Encouraging regular exercise routines may therefore be a **practical and cost-effective method** to enhance sleep health across all demographics.
    """)

    # 8. Sleep Disorder by BMI Category (Donut Chart)
    st.subheader("Sleep Disorder by BMI Category")
    bmi_disorder = filtered.groupby("BMI Category")["Sleep Disorder"].count().reset_index()
    fig9 = px.pie(bmi_disorder, names="BMI Category", values="Sleep Disorder",
                  hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig9, use_container_width=True)
    st.markdown("""
    **Sleep Disorders and Body Mass Index**  
    The donut chart demonstrates that individuals in the **overweight or obese BMI ranges** show slightly higher occurrences of sleep disorders.  
    This aligns with existing research linking **body composition to sleep apnea and poor sleep efficiency**.  
    Weight management and balanced diet programs could therefore play a significant role in improving sleep health outcomes.
    """)

    # 9. Daily Steps vs Stress Level
    st.subheader("Daily Steps vs Stress Level")
    filtered["Step Category"] = pd.cut(
        filtered["Daily Steps"],
        bins=[0, 3000, 7000, 10000, 15000],
        labels=["Sedentary", "Low", "Moderate", "High"]
    )
    fig10 = px.box(
        filtered,
        x="Step Category",
        y="Stress Level",
        color_discrete_sequence=["#636EFA"],
        title="Stress by Daily Steps Category"
    )
    st.plotly_chart(fig10, use_container_width=True)
    st.markdown("""
    **Impact of Daily Physical Activity on Stress Levels**  
    This boxplot illustrates that participants with **higher daily step counts** tend to have **lower stress levels**.  
    The reduction is particularly visible beyond the "Moderate" activity category, suggesting that even modest movement helps mental well-being.  
    Regular walking and physical activity can therefore act as a **preventive measure** against chronic stress and related sleep issues.
    """)

    # 10. Gender vs Sleep Disorder
    st.subheader("Treemap of Sleep Disorders by Gender")

    treemap_data = (
    filtered.groupby(["Gender", "Sleep Disorder"])
    .size()
    .reset_index(name="Count")
    )

    fig_treemap = px.treemap(
    treemap_data,
    path=["Gender", "Sleep Disorder"],
    values="Count",
    color="Count",
    color_continuous_scale="Tealgrn",
    )

    st.plotly_chart(fig_treemap, use_container_width=True)

    st.markdown("""
**Visualizing Sleep Disorders Across Gender Groups**  
This treemap provides a **hierarchical view** of how sleep disorders are distributed between genders.  
Each block represents a gender, subdivided by the type of sleep disorder (e.g., *Insomnia*, *Sleep Apnea*, *None*).  
The **size** of each section corresponds to the **number of participants** within that category.  
From this visualization, it’s clear that:
- **Insomnia** is more prevalent among **female participants**, possibly linked to stress or multitasking demands.  
- **Sleep Apnea** appears slightly more frequent among **male participants**, consistent with known medical trends.  
- Individuals reporting **no disorder** form the largest segment, suggesting lifestyle and stress management play vital roles in prevention.  

Overall, the treemap effectively illustrates **gender-based differences** while preserving an easy-to-understand hierarchy of sleep health categories.
""")

elif page == "Prediction":
    st.title("\U0001F52E Sleep Disorder Prediction")

    # Prepare dataset
    df_pred = df.copy()
    df_pred["Sleep Disorder"] = df_pred["Sleep Disorder"].fillna("None")
    df_pred = df_pred.drop(columns=["Person ID"])  # drop identifier

    # Split blood pressure into systolic/diastolic if needed
    if "Blood Pressure" in df_pred.columns:
        bp_split = df_pred["Blood Pressure"].str.split("/", expand=True).astype(float)
        df_pred["Systolic"] = bp_split[0]
        df_pred["Diastolic"] = bp_split[1]
        df_pred = df_pred.drop(columns=["Blood Pressure"])

    # Encode categorical features using pandas
    categorical_cols = ["Gender", "Occupation", "BMI Category"]
    df_encoded = pd.get_dummies(df_pred, columns=categorical_cols, drop_first=True)

    # Build features and target
    features = [col for col in df_encoded.columns if col != "Sleep Disorder"]
    X_full = df_encoded[features].values

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df_encoded["Sleep Disorder"])

    # Outlier Detection (Z-score) on numeric features only
    from scipy.stats import zscore
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    z_scores = np.abs(zscore(df_encoded[numeric_cols]))
    mask = (z_scores < 3).all(axis=1)
    X_full = X_full[mask]
    y = y[mask]

    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X_full)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Class Balancing
    st.subheader("\U0001F4CA Class Balancing (SMOTE option)")
    balance = st.checkbox("Apply SMOTE Oversampling", value=True, key="smote_checkbox")

    if balance:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        st.info("SMOTE applied!")  
    else:
        st.info("Oversampling not applied.")  

    # Model selection
    st.subheader("Choose Model") 
    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "SVM", "Decision Tree", "XGBoost"],
        key="model_selection"
    )

    # Initialize Model
    if model_choice == "Random Forest":
        model = RandomForestClassifier(class_weight='balanced' if not balance else None)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, class_weight='balanced' if not balance else None)
    elif model_choice == "SVM":
        model = SVC(kernel='rbf', probability=True, class_weight='balanced' if not balance else None)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(class_weight='balanced' if not balance else None)
    elif model_choice == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    from sklearn.metrics import accuracy_score, classification_report
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)

    st.subheader("\U+1F4CA " + f"{model_choice} Evaluation Metrics")  
    st.markdown(f"- **Accuracy:** `{accuracy_score(y_test, y_pred):.2f}`")
    st.markdown(f"- **Classes:** `{', '.join(le.classes_)}`")

    with st.expander("\U0001F4D8 Classification Report", expanded=False):
        st.dataframe(report_df)

    # --- Feature Importance (calculated but not displayed) ---
    importance = None
    if model_choice in ["Random Forest", "Decision Tree", "XGBoost"]:
        importance = dict(zip(features, model.feature_importances_))
    elif model_choice == "Logistic Regression":
        importance = dict(zip(features, abs(model.coef_[0])))
    elif model_choice == "SVM":
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importance = dict(zip(features, perm.importances_mean))
    # (kept in memory for reporting, not shown in dashboard)

    # --- Prediction Input ---
    st.subheader("\U0001F9E0 Predict Sleep Disorder")

    # Numeric inputs with units
    age = st.slider("Age (years)", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].mean()))
    sleep = st.slider("Sleep Duration (hours per day)", 4.0, 10.0, 7.0)
    quality = st.slider("Quality of Sleep (scale 1–10)", 1, 10, 7)
    stress = st.slider("Stress Level (scale 1–10)", 1, 10, 5)
    activity = st.slider("Physical Activity Level (minutes per day)", 0, 300, 30)
    systolic = st.slider("Systolic Blood Pressure (mmHg)", 90, 180, 120)
    diastolic = st.slider("Diastolic Blood Pressure (mmHg)", 60, 120, 80)
    heart_rate = st.slider("Resting Heart Rate (bpm)", 50, 120, 70)
    steps = st.slider("Daily Steps (count)", 1000, 20000, 8000)

    # Categorical inputs
    gender = st.selectbox("Gender", df["Gender"].unique())
    occupation = st.selectbox("Occupation", df["Occupation"].unique())
    bmi = st.selectbox("BMI Category", df["BMI Category"].unique())

    # Build input dataframe
    input_dict = {
        "Age": age,
        "Sleep Duration": sleep,
        "Quality of Sleep": quality,
        "Stress Level": stress,
        "Physical Activity Level": activity,
        "Systolic": systolic,
        "Diastolic": diastolic,
        "Heart Rate": heart_rate,
        "Daily Steps": steps,
        "Gender": gender,
        "Occupation": occupation,
        "BMI Category": bmi
    }
    input_df = pd.DataFrame([input_dict])

    # Predict only when button is clicked
    if st.button("\u2705 Predict Sleep Disorder"):
        # Apply same preprocessing (get_dummies for categoricals)
        input_encoded = pd.get_dummies(input_df, columns=["Gender","Occupation","BMI Category"], drop_first=True)

        # Align with training features
        for col in features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[features]

        # Scale
        input_scaled = scaler.transform(input_encoded)

        # Predict
        prediction_encoded = model.predict(input_scaled)[0]
        prediction = le.inverse_transform([prediction_encoded])[0]
        if prediction == "None":
            prediction = "Normal Sleep"

        # Advice mapping
        advice_map = {
            "Normal Sleep": "Your sleep pattern looks healthy. Keep maintaining good habits like regular exercise and consistent bedtimes.",
            "Insomnia": "You may be experiencing insomnia. Try to improve sleep hygiene, reduce screen time before bed, keep a consistent schedule, and consider relaxation techniques. Focus on lifestyle changes like losing weight, exercising, and avoiding alcohol, sedatives, and smoking.",
            "Sleep Apnea": " This often relates to breathing interruptions during sleep. Focus on lifestyle changes like losing weight, exercising, and avoiding alcohol, sedatives, and smoking. It is also helpful to change your sleeping position to your side and to address nasal congestion. It’s best to consult a healthcare professional for proper evaluation."
        }

        # Display result and advice
        st.subheader("\U0001F50E Prediction Result") 
        st.success(f"Predicted Sleep Disorder: {prediction}")

        st.markdown(f"\U0001F4A1 **Recommendation:** {advice_map.get(prediction, 'No advice available for this outcome.')}")

        st.subheader("\U0001F4CB Prediction Summary") 
        st.table(input_df.assign(Predicted_Disorder=prediction))



