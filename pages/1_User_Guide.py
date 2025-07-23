# pages/1_Overview.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="User Guide", layout="wide")
st.title("Page 1: User Guide")

with st.expander("📘 Explanation on Dataset", expanded=False):

    st.markdown("#### Key Terminology")
    st.markdown("""
    - **Variables**: A general term referring to the columns (features or targets) in a dataset.
    - **Independent Variables** (*Features* or *Attributes*): These are input variables used to train ML/DL models.
    - **Dependent Variables** (*Targets* or *Objectives*): These are output variables predicted by ML/DL models.
    """)

    st.markdown("#### Preview of Dataset")
    if "shared_df" in st.session_state:
        df = st.session_state["shared_df"]
    else:
        df = pd.read_excel("pages/ACD_Database_v2.xlsx", sheet_name="Data_Summary_PST")

    # Load and show full dataset
    st.dataframe(df)

    # Show numeric summary
    st.markdown("**Summary Statistics for Numeric Variables**")
    numeric_summary = df.select_dtypes(include=['number']).describe().T
    st.dataframe(numeric_summary)

    # Show categorical summary
    st.markdown("**Summary for Categorical Variables**")
    df = df.drop(columns=['Blend number'], errors='ignore')
    cat_summary = pd.DataFrame({
        'Unique Values': df.select_dtypes(include='object').nunique(),
        'Most Frequent (Mode)': df.select_dtypes(include='object').mode().iloc[0],
        'Frequency of Mode': df.select_dtypes(include='object').apply(lambda x: x.value_counts().iloc[0])
    })
    st.dataframe(cat_summary)

    st.markdown("#### Variables Used in Modeling")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Independent Variables** (Inputs to Models):")
        st.markdown("""
        - Primary Surfactant Name  
        - Secondary Surfactant Name  
        - Tertiary Surfactant Name  
        - Primary Surfactant Level (%)  
        - Secondary Surfactant Level (%)  
        - Tertiary Surfactant Level (%)  
        """)

    with col2:
        st.markdown("**Dependent Variables** (Targets for Prediction):")
        st.markdown("""
        - Clarity  
        - Colour  
        - Physical State  
        - Initial pH  
        - Appearance Absorbance Value  
        - Height Foam (mm)  
        - CMC  
        - Mildness  
        """)

    st.markdown("#### ⚠️ Notes on Other Variables")
    st.markdown("""
    Additional columns in the dataset — such as **surfactant weights**, **surfactant ratios**, **total surfactant ratios**, or **sodium benzoate (g)** — are either:
    - Constant values, or  
    - Calculated from the independent variables.
    
    These are **not used** in Gap Data Generation or ML model development to ensure model simplicity and avoid redundancy.
    """)


with st.expander("📊 UMAP Exploration", expanded=False):

    st.markdown("#### Layout and Navigation")

    st.markdown("""
    - To begin, you first need to upload the data in "Upload Data" page.
    - On the **right side**, you will see:
        - Three UMAP plots: **Feature Space**, **Performance Space**, and **Combined Space**.
        - A **Custom Section** where you can define your own UMAP projections using selected variables.
        - A **Download Filtered Data** section to export your explored subset.
    - On the **left side**, you'll find a set of **filter options** to interactively explore and refine the dataset.
    """)

    st.markdown("#### What is UMAP and Why is it Used?")

    st.markdown("""
    All visualizations in this section are powered by a technique called **UMAP**, short for **Uniform Manifold Approximation and Projection**.

    - UMAP is a **non-linear dimensionality reduction method**.
    - It helps to project high-dimensional data (e.g., with dozens of columns) into a **2D space**, while preserving both **local** and **global structures**.
    - In simpler terms, it **clusters similar data points together** based on their relationships in the original dataset.
    - This makes UMAP especially suitable for **datasets with complex or non-linear patterns**.

    In this dashboard:
    - **Feature Space Plot**: uses **independent variables** only.
    - **Performance Space Plot**: uses **dependent variables** only.
    - **Combined Space Plot**: uses **both** independent and dependent variables.
    """)

    st.markdown("#### Exploring Patterns and Clusters")

    st.markdown("""
    UMAP plots allow you to explore the dataset visually by highlighting meaningful clusters and relationships:

    - You can **colour the data points** using the dropdown menu labeled **"Colour by"**.
    - For instance, in the **Performance Space Plot**, selecting `Initial pH` as the color will reveal clusters of formulations with **high**, **low**, or **moderate** initial pH levels.
    - This helps you see how different performance results or ingredient choices affect the way the data groups together.

    You can also apply **filters** (on the left panel) to:
    - Narrow down the data based on specific **formulation components**, or
    - Isolate samples with desired **performance outcomes**.

    This interactive exploration enables you to quickly generate insights on the behavior and formulation patterns hidden in the data.
    """)


with st.expander("💰 Blend Price Calaculation", expanded=False):
    
    st.markdown("#### Price Calculation Overview") 
    st.markdown("""
    This section is in UMAP Exploration page and helps you estimate the **total minimum and maximum cost** of a surfactant blend based on:

    - The **blend weight (g)**
    - The **active % and level %** of primary (and optionally secondary and tertiary) surfactants
    - The known **price range per kg** of each surfactant
    """)

    st.markdown("#### What you can do?")
    st.markdown("""
    - Enter the **active percentage** and **level percentage** of the **primary surfactant**
    - Choose whether to include a **secondary** and **tertiary** surfactant
    - If selected, provide their **active %** and **level %** as well
    """)

    st.markdown("#### What happens after clicking “Calculate Price”?")
    st.markdown("""
    - It computes the **minimum and maximum estimated price** of the entire blend
    - Based on each surfactant’s individual weight and their respective **price range per kg** (converted from g to kg)
    """)



with st.expander("🧬 Gap Data Generation", expanded=False):

    st.markdown("#### Purpose of CTGAN and How It Works")

    st.markdown("""
    The gap data generation process uses a method called **CTGAN**, which stands for **Conditional Tabular Generative Adversarial Network**.

    - CTGAN is specially designed for **tabular data**, and it handles both **categorical** and **continuous** features.
    - It uses two neural networks:
        - A **Generator** that creates synthetic samples.
        - A **Discriminator** that tries to distinguish between real and fake samples.
    - These networks are trained together so that the generator learns to produce **realistic, high-quality synthetic data**.
    - CTGAN is capable of learning **complex relationships between variables**, which helps it generate hypothetical data that fills in underrepresented areas in the original dataset.
    """)

    st.markdown("#### How to Generate Gap Data")

    st.markdown("""
    - The synthetic (gap) data generation feature is located on the **UMAP Exploration** page.
    - At the **top of the page**, you'll see a button labeled **Synthetic Data Generation**.
    - Simply click this button to begin the process. It takes approximately **5 minutes** to complete.
    - Once done, the new gap data points will appear **in red** across all UMAP plots — **if you select `Type` in the "Colour by" dropdown box**.
    
    This feature helps uncover new potential formulations by generating data that fills in gaps not covered by the existing dataset.
    """)


with st.expander("📈 Performance Prediction", expanded=False):

    st.markdown("#### Layout Overview")
    st.markdown("""
    The Performance Prediction section is divided into **three parts**:
    
    1. **Train Model** – Train the models.
    2. **Make Prediction** – Enter a desired formulation to receive predicted performance.
    3. **Upload Model and Make Prediction** – Use pre-trained models to skip retraining time.
    """)

    st.markdown("#### Train Model")
    st.markdown("""
    - To begin, you first need to upload the data in "Upload Data" page.
    - Model training takes approximately **5 minutes**.
    - After training, you can:
        - Download the trained model, which will be given in zip folder, for future use.
        - Start making predictions using the trained model.
    """)

    st.markdown("#### Make Prediction")
    st.markdown("""
    - Provide your desired formulation inputs in the interface.
    - Click the **Predict** button to see expected performance values.
    - These results are **estimates** based on the model and **not guaranteed** to match lab outcomes.
    - It's recommended to validate important formulations in the **lab** before making decisions.
    """)

    st.markdown("#### Upload Model and Predict")
    st.markdown("""
    - If you've already trained models and want to avoid retraining, you can **upload the model files**.
    - First unzip the folder and upload the files based on their names indicated in the section.
    - After upload, you can immediately begin predictions using those models.
    """)

    st.markdown("#### Purpose of Predicting Performance")
    st.markdown("""
    This section helps users make informed decisions **before conducting lab experiments**.
    
    - If you're unsure how a formulation will perform, predicting it with an ML model gives you a solid expectation.
    - This can **save time, cost, and effort** by focusing only on promising candidates in the lab.
    """)

    st.markdown("#### Explanation of Model Performance")
    st.markdown("""
    The dashboard uses two machine learning models:
    - **Gradient Boosting Machine (GBM)**
    - **Random Forest (RF)**

    To improve prediction clarity, some target variables have been **simplified**:

    - **Clarity**:
        - Original: Clear, Slightly Turbid, Turbid
        - Simplified: Clear (merged with Slightly Turbid) and Turbid

    - **Colour**:
        - Original: Colourless, Grey, White, Yellow
        - Simplified: Colourless and Coloured (merged Grey, White, Yellow)

    - **CMC**:
        - Originally a numeric variable between 0.01 and 0.15
        - Converted into categorical: values **> 0.03** vs **≤ 0.03**

    These simplifications comes with trade off between improving models' performance and the informantion that the models can give.
    """)

    st.markdown("#### Model Performance Metrics")
    st.markdown("""
    - **Clarity (categorical)**: 0.76 (± 0.07)  
    - **Colour (categorical)**: 0.78 (± 0.05)  
    - **Physical State (categorical)**: 0.88 (± 0.20)  
    - **CMC (categorical)**: 0.85 (± 0.01)  
    - **Initial pH (numeric)**: 0.95 (± 0.01)  
    - **Appearance Absorbance Value (numeric)**: 0.84 (± 0.09)  
    - **Height Foam (numeric)**: 0.91 (± 0.03)  
    - **Mildness (numeric)**: 0.73 (± 0.08)
    """)

    st.markdown("#### How to Interpret These Scores")

    st.markdown("**For numeric targets (R²):**")
    st.markdown("""
    - A value between **0.7 and 0.9** indicates that the model explains a **large portion of the variability**.
    - This makes the predictions reliable for exploration and decision-making.
    """)

    st.markdown("**For categorical targets (Balanced Accuracy):**")
    st.markdown("""
    - A value between **0.65 and 0.8** suggests the model **handles class imbalance well**.
    - It means the model is **not biased toward dominant classes** and performs reasonably across all categories.
    """)


with st.expander("🧪 Formulation Suggestion", expanded=False):

    st.markdown("#### Layout Overview")

    st.markdown("""
    This section contains **three parts**:
    
    1. **Train GP Model** – Train the Gaussian Process (GP) model.  
    2. **BRO: Specified Targets** – Suggest formulations by defining exact target values and their importance.  
    3. **BRO: Maximize / Minimize / Close-to** – Suggest formulations by defining objectives like maximizing or minimizing for each numeric target, soft-weight for each class in a categorical target, and their overall importance.
    """)

    st.markdown("#### Train GP Model")

    st.markdown("""
    - To begin, you first need to upload the data in "Upload Data" page.
    - Training takes around **2 minutes**.
    - This model is used internally to power both BRO methods and
    - Download the model option is not available for this model.
    """)

    st.markdown("#### BRO: Specified Targets")

    st.markdown("""
    - In this section, you can define **specific performance values** that you want your formulation to achieve.
    - You can also assign **weights (importance levels)** to each target.
    - The system will generate **5 suggested formulations** that aim to match your specified targets as closely as possible.
    """)

    st.markdown("#### BRO: Maximize / Minimize / Close-to")

    st.markdown("""
    - Here, you can define your **goal for each numeric target**: whether to **maximize**, **minimize**, or **match a reference value**.
    - For **categorical targets**, you can assign **soft weights** for each class
    - You can also set the **importance level** for each target.
    - After configuration, the system will return **5 recommended formulations**.
    """)

    st.markdown("#### Why Use Formulation Suggestion?")

    st.markdown("""
    This tool helps users **identify promising formulations** based on performance goals **before conducting lab experiments**.

    - It reduces the number of lab trials required.
    - Saves **time**, **resources**, and **costs** by narrowing the focus to data-driven suggestions.
    """)

    st.markdown("#### Evaluating Model Output")

    st.markdown("""
    So far, model performance is judged based on the **diversity of the suggested formulations**:

    - **Diverse suggestions** indicate that the model is exploring multiple possibilities:
        - Increases robustness of results,
        - Reduces risk of missing better options,
        - Gives flexibility for constraints like ingredient cost or availability.
    - **Similar suggestions** indicate exploitation of a narrow region:
        - Can result in local optima,
        - May miss better solutions elsewhere,
        - Limits flexibility for experimental design.
    
    This balance between **exploration** (trying new areas) and **exploitation** (focusing on known good areas) is key to effective optimization.
    The current BRO model implemented in this dashboard gives acceptable diversity in the suggestions but there will be room for improvement.
    """)

    st.markdown("#### Iterative Optimization Process")

    st.markdown("""
    - Both BRO methods return only **5 formulation suggestions**.
    - Some of these may be **similar**, so users don’t need to test all 5 in the lab.
    - After testing a few promising ones, users should **add those lab results to the dataset**, upload the updated file in "Upload Data" page, and **retrain the GP model**.
    - Then, repeat the BRO process with the same goals to receive **improved suggestions**.

    ⚠️ *This is an iterative cycle.* You repeat the process until you arrive at a formulation that meets your desired performance.
    """)
