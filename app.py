import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import joblib
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, 
    LabelEncoder, MinMaxScaler, RobustScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)

# All sklearn models
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Set page config
st.set_page_config(
    page_title="AutoML with Streamlit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox, .stMultiselect, .stSlider {
        background-color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
    .model-results {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .download-section {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'all_models_results' not in st.session_state:
    st.session_state.all_models_results = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

# Sidebar navigation
st.sidebar.title("AutoML Navigation")
page = st.sidebar.radio("Go to", 
    ["📤 Data Upload", "🔧 Data Preprocessing", "🤖 Model Training", "📊 Results"])

# Data Upload Page
if page == "📤 Data Upload":
    st.title("📤 Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.df_processed = df.copy()
            
            # Data preview
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head())
            
            # Data information
            st.subheader("ℹ️ Data Information")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
            # Descriptive statistics
            st.subheader("📊 Descriptive Statistics")
            st.write(df.describe())
            
            # Missing values
            st.subheader("❓ Missing Values")
            missing = df.isnull().sum()
            st.write(missing[missing > 0])
            
            st.success("✅ Data loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# Data Preprocessing Page
elif page == "🔧 Data Preprocessing" and st.session_state.df is not None:
    st.title("🔧 Data Preprocessing")
    df = st.session_state.df_processed
    
    # Display current data
    st.subheader("📋 Current Data")
    st.dataframe(df, height=300)
    
    # Column removal
    st.subheader("🗑️ Column Removal")
    cols_to_drop = st.multiselect("Select columns to remove", df.columns)
    if st.button("Remove Selected Columns"):
        df = df.drop(columns=cols_to_drop)
        st.session_state.df_processed = df
        st.rerun()
    
    # Select target variable
    st.subheader("🎯 Select Target Variable")
    target = st.selectbox("Choose the target variable", df.columns)
    st.session_state.target = target
    
    # Determine problem type
    st.subheader("🔮 Select Problem Type")
    if pd.api.types.is_numeric_dtype(df[target]):
        problem_type = st.radio("Is this a regression or classification problem?",
                              ["Regression", "Classification"])
    else:
        problem_type = "Classification"
        st.info("ℹ️ Target variable is non-numeric, defaulting to Classification")
    
    st.session_state.problem_type = problem_type
    
    # Feature selection
    st.subheader("📌 Feature Selection")
    features = [col for col in df.columns if col != target]
    selected_features = st.multiselect("Select features to include", features, default=features)
    st.session_state.selected_features = selected_features
    
    # Handle missing values
    st.subheader("🔍 Missing Value Handling")
    numeric_cols = df[selected_features].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df[selected_features].select_dtypes(exclude=np.number).columns.tolist()
    
    st.write("🔢 Numeric columns:", numeric_cols)
    st.write("🔤 Categorical columns:", categorical_cols)
    
    # Numeric preprocessing options
    st.markdown("**🔢 Numeric Columns**")
    num_strategy = st.selectbox("Numeric missing values strategy", 
                               ["mean", "median", "most_frequent", "constant"])
    num_scaling = st.selectbox("Numeric scaling method",
                             ["Standard Scaler", "MinMax Scaler", "Robust Scaler", "None"])
    
    # Categorical preprocessing options
    st.markdown("**🔤 Categorical Columns**")
    cat_strategy = st.selectbox("Categorical missing values strategy",
                              ["most_frequent", "constant"])
    cat_encoding = st.selectbox("Categorical encoding method",
                              ["One-Hot Encoding", "Ordinal Encoding", "Label Encoding"])
    
    # Process button
    if st.button("🚀 Process Data"):
        with st.spinner("🔄 Processing data..."):
            # Numeric transformers
            if num_scaling == "Standard Scaler":
                scaler = StandardScaler()
            elif num_scaling == "MinMax Scaler":
                scaler = MinMaxScaler()
            elif num_scaling == "Robust Scaler":
                scaler = RobustScaler()
            else:
                scaler = None
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=num_strategy))])
            
            if scaler is not None:
                numeric_transformer.steps.append(('scaler', scaler))
            
            # Categorical transformers
            if cat_encoding == "One-Hot Encoding":
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=cat_strategy, fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            elif cat_encoding == "Ordinal Encoding":
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=cat_strategy, fill_value='missing')),
                    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
            else:  # Label Encoding
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=cat_strategy, fill_value='missing')),
                    ('label', LabelEncoder())])
            
            # Preprocessing pipeline
            transformers = []
            if numeric_cols:
                transformers.append(('num', numeric_transformer, numeric_cols))
            if categorical_cols:
                transformers.append(('cat', categorical_transformer, categorical_cols))
            
            preprocessor = ColumnTransformer(transformers=transformers)
            
            st.session_state.preprocessor = preprocessor
            st.success("✅ Data processing complete!")

# Model Training Page
elif page == "🤖 Model Training" and st.session_state.df_processed is not None and st.session_state.target is not None:
    st.title("🤖 Model Training")
    
    df = st.session_state.df_processed
    target = st.session_state.target
    problem_type = st.session_state.problem_type
    
    # Display current data
    st.subheader("📋 Processed Data")
    st.dataframe(df, height=300)
    
    # Train-test split
    st.subheader("✂️ Train-Test Split")
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    with col2:
        random_state = st.number_input("Random state", value=42)
    
    X = df[st.session_state.selected_features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    # Model selection
    st.subheader("🤖 Model Selection")
    
    # Option to try all models
    try_all = st.checkbox("Try all models (comprehensive comparison)")
    
    if not try_all:
        # Single model selection
        if problem_type == "Classification":
            model_options = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Support Vector Machine": SVC(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Extra Trees": ExtraTreesClassifier(),
                "Bagging": BaggingClassifier(),
                "Naive Bayes": GaussianNB(),
                "Stochastic Gradient Descent": SGDClassifier(),
                "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                "Neural Network": MLPClassifier(),
                "Gaussian Process": GaussianProcessClassifier()
            }
            
            model_choice = st.selectbox("Select a classification model", list(model_options.keys()))
            model = model_options[model_choice]
            
        else:  # Regression
            model_options = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "ElasticNet": ElasticNet(),
                "Support Vector Machine": SVR(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Extra Trees": ExtraTreesRegressor(),
                "Bagging": BaggingRegressor(),
                "Stochastic Gradient Descent": SGDRegressor(),
                "Neural Network": MLPRegressor(),
                "Gaussian Process": GaussianProcessRegressor()
            }
            
            model_choice = st.selectbox("Select a regression model", list(model_options.keys()))
            model = model_options[model_choice]
        
        # Create and fit pipeline
        if st.session_state.preprocessor is not None:
            pipeline = Pipeline(steps=[
                ('preprocessor', st.session_state.preprocessor),
                ('model', model)])
            
            if st.button("🚀 Train Selected Model"):
                with st.spinner(f"🧠 Training {model_choice}..."):
                    try:
                        start_time = time.time()
                        pipeline.fit(X_train, y_train)
                        training_time = time.time() - start_time
                        
                        # Make predictions
                        y_pred = pipeline.predict(X_test)
                        
                        # Store results
                        st.session_state.model = pipeline
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.predictions = y_pred
                        st.session_state.current_model_name = model_choice
                        st.session_state.training_time = training_time
                        
                        # Store model for download
                        st.session_state.trained_models[model_choice] = pipeline
                        
                        st.success(f"✅ {model_choice} trained successfully in {training_time:.2f} seconds!")
                    except Exception as e:
                        st.error(f"❌ Error training model: {e}")
    
    else:  # Try all models
        if st.button("🚀 Train All Models"):
            if st.session_state.preprocessor is not None:
                all_models = []
                results = []
                
                if problem_type == "Classification":
                    models = [
                        ("Random Forest", RandomForestClassifier(random_state=random_state)),
                        ("Gradient Boosting", GradientBoostingClassifier(random_state=random_state)),
                        ("Logistic Regression", LogisticRegression(random_state=random_state)),
                        ("SVM", SVC(random_state=random_state)),
                        ("K-Nearest Neighbors", KNeighborsClassifier()),
                        ("Decision Tree", DecisionTreeClassifier(random_state=random_state)),
                        ("AdaBoost", AdaBoostClassifier(random_state=random_state)),
                        ("Naive Bayes", GaussianNB()),
                        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
                        ("Neural Network", MLPClassifier(random_state=random_state))
                    ]
                else:  # Regression
                    models = [
                        ("Random Forest", RandomForestRegressor(random_state=random_state)),
                        ("Gradient Boosting", GradientBoostingRegressor(random_state=random_state)),
                        ("Linear Regression", LinearRegression()),
                        ("Ridge Regression", Ridge(random_state=random_state)),
                        ("Lasso Regression", Lasso(random_state=random_state)),
                        ("SVM", SVR()),
                        ("K-Nearest Neighbors", KNeighborsRegressor()),
                        ("Decision Tree", DecisionTreeRegressor(random_state=random_state)),
                        ("AdaBoost", AdaBoostRegressor(random_state=random_state)),
                        ("Neural Network", MLPRegressor(random_state=random_state))
                    ]
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (name, model) in enumerate(models):
                    try:
                        status_text.text(f"🧠 Training {name}...")
                        pipeline = Pipeline(steps=[
                            ('preprocessor', st.session_state.preprocessor),
                            ('model', model)])
                        
                        start_time = time.time()
                        pipeline.fit(X_train, y_train)
                        training_time = time.time() - start_time
                        
                        y_pred = pipeline.predict(X_test)
                        
                        if problem_type == "Classification":
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            results.append({
                                "Model": name,
                                "Accuracy": f"{accuracy:.4f}",
                                "Precision": f"{precision:.4f}",
                                "Recall": f"{recall:.4f}",
                                "F1 Score": f"{f1:.4f}",
                                "Training Time (s)": f"{training_time:.2f}"
                            })
                        else:  # Regression
                            mae = mean_absolute_error(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y_test, y_pred)
                            results.append({
                                "Model": name,
                                "MAE": f"{mae:.4f}",
                                "MSE": f"{mse:.4f}",
                                "RMSE": f"{rmse:.4f}",
                                "R2": f"{r2:.4f}",
                                "Training Time (s)": f"{training_time:.2f}"
                            })
                        
                        # Store model for download
                        st.session_state.trained_models[name] = pipeline
                        
                        # Store the best model
                        if i == 0:
                            best_model = pipeline
                            best_model_name = name
                            best_score = accuracy if problem_type == "Classification" else r2
                        else:
                            current_score = accuracy if problem_type == "Classification" else r2
                            if (problem_type == "Classification" and current_score > best_score) or \
                               (problem_type == "Regression" and current_score > best_score):
                                best_model = pipeline
                                best_model_name = name
                                best_score = current_score
                        
                    except Exception as e:
                        results.append({
                            "Model": name,
                            "Error": str(e)
                        })
                    
                    progress_bar.progress((i + 1) / len(models))
                
                # Store all results
                st.session_state.all_models_results = pd.DataFrame(results)
                st.session_state.model = best_model
                st.session_state.current_model_name = best_model_name
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.predictions = best_model.predict(X_test)
                
                status_text.text("✅ Training complete!")
                st.success(f"🏆 Best model: {best_model_name}")
                
                # Show results table
                st.subheader("📊 All Models Performance")
                st.dataframe(st.session_state.all_models_results)
            else:
                st.warning("⚠️ Please process the data first on the Data Preprocessing page")

# Results Page
elif page == "📊 Results" and st.session_state.model is not None:
    st.title("📊 Model Results")
    
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = st.session_state.predictions
    problem_type = st.session_state.problem_type
    
    # Display metrics
    st.subheader("📈 Model Performance")
    
    if hasattr(st.session_state, 'current_model_name'):
        st.markdown(f"### 🏆 Current Model: **{st.session_state.current_model_name}**")
    
    if hasattr(st.session_state, 'training_time'):
        st.markdown(f"⏱️ **Training Time:** {st.session_state.training_time:.2f} seconds")
    
    if problem_type == "Classification":
        st.markdown("""
        | Metric | Value |
        |---|---|
        | Accuracy | {:.4f} |
        | Precision (weighted) | {:.4f} |
        | Recall (weighted) | {:.4f} |
        | F1 Score (weighted) | {:.4f} |
        """.format(
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average='weighted'),
            recall_score(y_test, y_pred, average='weighted'),
            f1_score(y_test, y_pred, average='weighted')
        ))
        
        # Classification report
        st.subheader("📝 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(report).transpose())
        
        # Confusion matrix
        st.subheader("🔄 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                           columns=[f"Predicted {i}" for i in np.unique(y_test)],
                           index=[f"Actual {i}" for i in np.unique(y_test)])
        st.write(cm_df)
        
    else:  # Regression
        st.markdown("""
        | Metric | Value |
        |---|---|
        | Mean Absolute Error | {:.4f} |
        | Mean Squared Error | {:.4f} |
        | Root Mean Squared Error | {:.4f} |
        | R-squared | {:.4f} |
        """.format(
            mean_absolute_error(y_test, y_pred),
            mean_squared_error(y_test, y_pred),
            np.sqrt(mean_squared_error(y_test, y_pred)),
            r2_score(y_test, y_pred)
        ))
        
        # Plot actual vs predicted
        st.subheader("📈 Actual vs Predicted")
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.line_chart(results.head(50))
    
    # Feature importance for tree-based models
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        st.subheader("🔍 Feature Importance")
        try:
            # Get feature names after preprocessing
            preprocessor = model.named_steps['preprocessor']
            feature_names = []
            
            # Numeric features
            if 'num' in preprocessor.named_transformers_:
                feature_names.extend(preprocessor.named_transformers_['num'].feature_names_in_)
            
            # Categorical features (after one-hot encoding)
            if 'cat' in preprocessor.named_transformers_:
                ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = preprocessor.named_transformers_['cat'].feature_names_in_
                ohe_names = ohe.get_feature_names_out(cat_features)
                feature_names.extend(ohe_names)
            
            # Get importances
            importances = model.named_steps['model'].feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            st.bar_chart(importance_df.set_index('Feature').head(10))
        except Exception as e:
            st.warning(f"⚠️ Could not display feature importance: {e}")
    
    # Download Section
    st.markdown("---")
    st.subheader("💾 Download Options")
    
    # Download preprocessed data
    if st.session_state.df_processed is not None:
        csv = st.session_state.df_processed.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Preprocessed Data (CSV)",
            data=csv,
            file_name='preprocessed_data.csv',
            mime='text/csv'
        )
    
    # Download models
    if len(st.session_state.trained_models) > 0:
        if len(st.session_state.trained_models) == 1:
            # Single model download
            model_name, model_obj = next(iter(st.session_state.trained_models.items()))
            model_bytes = io.BytesIO()
            joblib.dump(model_obj, model_bytes)
            model_bytes.seek(0)
            
            st.download_button(
                label=f"📥 Download {model_name} Model",
                data=model_bytes,
                file_name=f'{model_name.lower().replace(" ", "_")}_model.joblib',
                mime='application/octet-stream'
            )
        else:
            # Multiple models - create zip
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                for name, model_obj in st.session_state.trained_models.items():
                    model_bytes = io.BytesIO()
                    joblib.dump(model_obj, model_bytes)
                    model_bytes.seek(0)
                    zip_file.writestr(f'{name.lower().replace(" ", "_")}_model.joblib', model_bytes.getvalue())
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📦 Download All Models (ZIP)",
                data=zip_buffer,
                file_name='trained_models.zip',
                mime='application/zip'
            )

# Handle cases where required data isn't available
else:
    if page != "📤 Data Upload":
        st.warning("⚠️ Please upload data first on the Data Upload page")
    st.title("🤖 Codeless Machine Learning")
    st.markdown("""
    Welcome to the **Codeless Machine Learning** platform! This tool allows you to:
    
    1. **📤 Upload** your CSV file
    2. **🔧 Preprocess** your data (handle missing values, scale features, etc.)
    3. **🤖 Train** machine learning models (both classification and regression)
    4. **📊 Evaluate** model performance
    
    ### Getting Started:
    1. Go to the **Data Upload** page
    2. Upload your CSV file
    3. Follow the steps in the navigation sidebar
    
    No coding required! 🎉
    """)