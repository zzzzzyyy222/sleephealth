import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import plotly.express as px
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

    st.markdown("""
    Explore how lifestyle and demographic factors influence sleep health, stress, and activity levels. 
    This section provides insights into sleep patterns, disorders, and the impact of physical activity across different participant groups.
    """)

    # -----------------------------
    # SECTION 1: SLEEP PATTERNS
    # -----------------------------
    st.header("1️. Sleep Patterns & Disorders")

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Sleep Duration (hrs)", f"{filtered['Sleep Duration'].mean():.2f}")
    col2.metric("Median Sleep Duration (hrs)", f"{filtered['Sleep Duration'].median():.2f}")
    col3.metric("Avg Stress Level", f"{filtered['Stress Level'].mean():.2f}")
    col4.metric("Avg Daily Steps", f"{filtered['Daily Steps'].mean():.0f}")

    # --- Sleep Duration by Occupation ---
    st.subheader("\U0001F4CA Sleep Duration by Occupation")
    fig_occ = px.box(
        filtered,
        x="Occupation",
        y="Sleep Duration",
        color="Occupation",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Sleep Duration by Occupation"
    )
    fig_occ.update_layout(yaxis_title="Sleep Duration (hrs)")
    st.plotly_chart(fig_occ, use_container_width=True)
    st.markdown("""
    Sleep patterns can vary across different occupations due to varying demands and work schedules. Some occupations may show wider variations in sleep duration, reflecting differences in personal habits or responsibilities. Observing these patterns helps understand how work-life balance influences rest. The visualization provides a general overview of how occupation relates to sleep without focusing on specific values.
    """)

    # --- Distribution of Sleep Quality ---
    st.subheader("\U0001F4A4 Distribution of Sleep Quality")
    sleep_quality_counts = filtered.groupby(["Quality of Sleep", "Gender"]).size().reset_index(name="Count")
    fig_quality = px.bar(
        sleep_quality_counts,
        x="Quality of Sleep",
        y="Count",
        color="Gender",
        barmode="group",
        title="Distribution of Sleep Quality by Gender",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_quality.update_layout(xaxis_title="Quality of Sleep (1-10)", yaxis_title="Count")
    st.plotly_chart(fig_quality, use_container_width=True)
    st.markdown("""
    Sleep quality varies across individuals and can be influenced by factors like stress, lifestyle, and daily habits. The distribution shows general trends in how participants report their rest without focusing on precise counts. Differences between groups may indicate patterns worth further exploration. Overall, this visualization helps identify broad trends in sleep quality across the population.
    """)

    # --- Stress vs Sleep Quality ---
    st.subheader("\U0001F4A2 Stress Level vs Average Sleep Quality")
    avg_stress_sleep = filtered.groupby("Stress Level")["Quality of Sleep"].mean().reset_index()
    fig_stress = px.line(
        avg_stress_sleep,
        x="Stress Level",
        y="Quality of Sleep",
        markers=True,
        color_discrete_sequence=["#FF6F61"],
        title="Stress Level vs Average Sleep Quality"
    )
    fig_stress.update_layout(yaxis_title="Average Sleep Quality (1-10)")
    st.plotly_chart(fig_stress, use_container_width=True)
    st.markdown("""
    There is a general relationship between stress and sleep quality, where higher stress tends to relate to lower quality of sleep. This trend highlights the importance of mental and emotional well-being for restful sleep. Observing patterns across participants helps identify how stress may impact sleep in different contexts. The chart provides a high-level view of this relationship without specifying numerical values.
    """)

    # -----------------------------
    # SECTION 2: LIFESTYLE & ACTIVITY
    # -----------------------------
    st.header("2️. Lifestyle & Physical Activity")

    # Physical Activity vs Sleep Quality (Scatter Plot)
    st.subheader("\U0001F3C3 Physical Activity Level vs Sleep Quality")
    fig_activity_sleep = px.scatter(
        filtered,
        x="Physical Activity Level",
        y="Quality of Sleep",
        color="Gender",
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Physical Activity Level vs Quality of Sleep"
    )
    fig_activity_sleep.update_layout(
        yaxis_title="Quality of Sleep (1-10)",
        xaxis_title="Physical Activity Level (minutes)"
    )
    st.plotly_chart(fig_activity_sleep, use_container_width=True)
    st.markdown("""
    Participants with higher physical activity levels generally report better sleep quality. This suggests a positive relationship between exercise and restful sleep. The scatter plot allows us to observe overall patterns and trends rather than focusing on specific values. These insights highlight the potential benefits of regular activity on sleep health in a general context.
    """)

    # -----------------------------
    # SECTION 3: CORRELATION
    # -----------------------------
    st.header("3️. Correlation Analysis")
    numeric_cols = filtered.select_dtypes(include=np.number).columns.tolist()
    cols_to_remove = ["Person ID", "Person_ID", "ID", "id"]
    numeric_cols = [col for col in numeric_cols if col not in cols_to_remove]
    corr = filtered[numeric_cols].corr()

    st.subheader("\U0001F4C8 Correlation Heatmap")
    fig_heat, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="Spectral",
        center=0,
        linewidths=0.5,
        fmt=".2f",
        ax=ax,
        cbar_kws={"shrink": 0.75}
    )
    ax.set_title("Correlation Matrix")
    st.pyplot(fig_heat)
    st.markdown("""
    The correlation heatmap shows how numerical variables relate to each other. Positive or negative trends indicate general patterns in the data without relying on exact numbers. Observing correlations helps identify which factors are associated with better or worse sleep. This visualization provides a high-level overview of potential relationships for further exploration.
    """)

    # -----------------------------
    # SECTION 4: SLEEP DISORDER ANALYSIS
    # -----------------------------
    st.header("4️. Sleep Disorder Analysis")

    # Sleep Disorder Distribution (Pie)
    st.subheader("\U0001FA7A Sleep Disorder Distribution")
    disorder_counts = filtered["Sleep Disorder"].value_counts().reset_index()
    disorder_counts.columns = ["Sleep Disorder", "Count"]
    fig_disorder = px.pie(
        disorder_counts,
        names="Sleep Disorder",
        values="Count",
        color="Sleep Disorder",
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="Sleep Disorder Distribution"
    )
    fig_disorder.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_disorder, use_container_width=True)
    st.markdown("""
    The distribution of sleep disorders shows general prevalence patterns among participants. Some disorders appear more frequently, while others are less common. This overview helps identify areas for awareness or intervention without emphasizing specific counts. Understanding overall trends allows for a broad view of sleep health across the population.
    """)

    # Sleep Disorder by Age
    st.subheader("\U0001F476\U0001F9D3 Sleep Disorders by Age Group")
    fig_age = px.histogram(
        filtered,
        x="Age",
        color="Sleep Disorder",
        nbins=20,
        barnorm="percent",
        color_discrete_sequence=px.colors.qualitative.Set1,
        title="Sleep Disorders by Age Group"
    )
    st.plotly_chart(fig_age, use_container_width=True)
    st.markdown("""
    Sleep disorders may vary with age, showing general patterns of prevalence across different age groups. Observing these trends helps understand how life stage and lifestyle might influence sleep health. The visualization provides an overall perspective rather than exact numbers. This can inform broad strategies for preventive measures and health promotion.
    """)

    # Sleep Disorder by BMI Category
    st.subheader("\U00002696 Sleep Disorder by BMI Category")
    bmi_counts = filtered.groupby("BMI Category")["Sleep Disorder"].count().reset_index()
    bmi_counts.columns = ["BMI Category", "Count"]
    fig_bmi_donut = px.pie(
        bmi_counts,
        names="BMI Category",
        values="Count",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Sleep Disorder by BMI Category"
    )
    fig_bmi_donut.update_traces(textinfo="label+value+percent", textposition="outside",
                                hovertemplate="%{label}: %{value} (%{percent})")
    fig_bmi_donut.update_layout(margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_bmi_donut, use_container_width=True)
    st.markdown("""
    Sleep disorder prevalence can differ across BMI categories, showing general trends rather than specific values. Patterns suggest potential associations between body composition and sleep health outcomes. Visualizing these relationships helps highlight populations that might benefit from lifestyle interventions. The donut chart provides an overview of how BMI relates to sleep disorders in the dataset.
    """)
elif page == "Prediction":
    st.title("\U0001F52E Sleep Disorder Prediction")

    # Prepare dataset
    df_pred = df.copy()
    df_pred["Sleep Disorder"] = df_pred["Sleep Disorder"].fillna("None")
    df_pred = df_pred.drop(columns=["Person ID"])

    # Split blood pressure
    if "Blood Pressure" in df_pred.columns:
        bp_split = df_pred["Blood Pressure"].str.split("/", expand=True).astype(float)
        df_pred["Systolic"] = bp_split[0]
        df_pred["Diastolic"] = bp_split[1]
        df_pred = df_pred.drop(columns=["Blood Pressure"])

    # Encode categorical features
    categorical_cols = ["Gender", "Occupation", "BMI Category"]
    df_encoded = pd.get_dummies(df_pred, columns=categorical_cols, drop_first=True)

    # Features and target
    features = [col for col in df_encoded.columns if col != "Sleep Disorder"]
    X_full = df_encoded[features].values

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df_encoded["Sleep Disorder"])

    # Outlier removal
    from scipy.stats import zscore
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    z_scores = np.abs(zscore(df_encoded[numeric_cols]))
    mask = (z_scores < 3).all(axis=1)
    X_full = X_full[mask]
    y = y[mask]

    # Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X_full)

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE option
    st.subheader("\U0001F4CA Class Balancing (SMOTE option)")
    balance = st.checkbox("Apply SMOTE Oversampling", value=True)

    from collections import Counter
    y_before_counts = Counter(y_train)

    if balance:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        st.info("SMOTE applied!")
    else:
        st.info("Oversampling not applied.")

    # SMOTE visualization
    if balance:
        y_after_counts = Counter(y_train)
        df_balance = pd.DataFrame({
            "Class": list(le.classes_) * 2,
            "Count": [y_before_counts.get(i, 0) for i in range(len(le.classes_))] +
                     [y_after_counts.get(i, 0) for i in range(len(le.classes_))],
            "Stage": ["Before SMOTE"] * len(le.classes_) + ["After SMOTE"] * len(le.classes_)
        })

        import plotly.express as px
        fig_balance = px.bar(
            df_balance,
            x="Class",
            y="Count",
            color="Stage",
            barmode="group",
            title="Class Distribution Before and After SMOTE",
            color_discrete_map={"Before SMOTE": "#62C3A5", "After SMOTE": "#F78364"}
        )
        fig_balance.update_layout(xaxis_title="Class", yaxis_title="Count", title_x=0.3)
        st.plotly_chart(fig_balance, use_container_width=True)
    else:
        st.info("SMOTE not applied — no balancing visualization to show.")

    # Model selection
    st.subheader("Choose Model")
    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "SVM", "Decision Tree", "XGBoost"])

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

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

    # Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)

    st.subheader(f"{model_choice} Evaluation Metrics")
    st.markdown(f"- **Accuracy:** `{accuracy_score(y_test, y_pred):.2f}`")
    st.markdown(f"- **Classes:** `{', '.join(le.classes_)}`")

    with st.expander("\U0001F4D8 Classification Report", expanded=False):
        st.dataframe(report_df)

    # Feature importance
    importance = None
    if model_choice in ["Random Forest", "Decision Tree", "XGBoost"]:
        importance = dict(zip(features, model.feature_importances_))
    elif model_choice == "Logistic Regression":
        importance = dict(zip(features, abs(model.coef_[0])))
    elif model_choice == "SVM":
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importance = dict(zip(features, perm.importances_mean))

    # Top features
    st.subheader("\U0001F9E0 Predict Sleep Disorder")
    top_n = 10
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_feature_names = [f[0] for f in top_features]

    st.markdown(f"Showing top **{top_n} important features** based on {model_choice} importance ranking.")

    # Input fields
    input_dict = {}
    used_categorical = set()

    for feature in top_feature_names:
        if any(cat in feature for cat in ["Gender", "Occupation", "BMI Category"]):
            base_feature = "Gender" if "Gender" in feature else \
                           "Occupation" if "Occupation" in feature else "BMI Category"
            if base_feature not in used_categorical:
                options = df[base_feature].unique()
                input_dict[base_feature] = st.selectbox(f"{base_feature}", options)
                used_categorical.add(base_feature)
        else:
            col_min, col_max = float(df_pred[feature].min()), float(df_pred[feature].max())
            default = float(df_pred[feature].mean())
            input_dict[feature] = st.slider(f"{feature}", col_min, col_max, default)

    input_df = pd.DataFrame([input_dict])

    # Prediction
    if st.button("\u2705 Predict Sleep Disorder"):
        input_encoded = pd.get_dummies(input_df, columns=["Gender", "Occupation", "BMI Category"], drop_first=True)
        for col in features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[features]

        input_scaled = scaler.transform(input_encoded)
        prediction_encoded = model.predict(input_scaled)[0]
        prediction = le.inverse_transform([prediction_encoded])[0]
        if prediction == "None":
            prediction = "Normal Sleep"

        advice_map = {
            "Normal Sleep": "Your sleep pattern looks healthy. Keep maintaining good habits like regular exercise and consistent bedtimes.",
            "Insomnia": "You may be experiencing insomnia. Try to improve sleep hygiene, reduce screen time before bed, and manage stress levels.",
            "Sleep Apnea": "This often relates to breathing interruptions during sleep. Lifestyle changes such as weight management and sleeping on your side can help."
        }

        st.subheader("\U0001F50E Prediction Result")
        st.success(f"Predicted Sleep Disorder: {prediction}")
        st.markdown(f"\U0001F4A1 **Recommendation:** {advice_map.get(prediction, 'No advice available for this outcome.')}")
        st.subheader("\U0001F4CB Prediction Summary")


        st.subheader("\U0001F50E Prediction Result")
        st.success(f"Predicted Sleep Disorder: {prediction}")
        st.markdown(f"\U0001F4A1 **Recommendation:** {advice_map.get(prediction, 'No advice available for this outcome.')}")
        st.subheader("\U0001F4CB Prediction Summary")
