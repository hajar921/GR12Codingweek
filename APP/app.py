import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt
import shap  # Import SHAP library

# Set page configuration
st.set_page_config(
    page_title="Obesity Risk Prediction Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_PATH = "Training Model\obesity_model.joblib"

# Function to load saved models and preprocessing objects
@st.cache_resource
def load_saved_model():
    # Load all saved objects
    model_data = joblib.load(MODEL_PATH)
    return model_data

# Helper function to display feature descriptions
def get_feature_description(feature):
    descriptions = {
        'Age': 'Patient age in years',
        'Gender': 'Patient gender (Male/Female)',
        'family_history_with_overweight': 'Family history of overweight (Yes/No)',
        'FAVC': 'Frequent consumption of high caloric food (Yes/No)',
        'FCVC': 'Frequency of consumption of vegetables (1-3 scale)',
        'NCP': 'Number of main meals per day (1-4)',
        'CAEC': 'Consumption of food between meals (Never/Sometimes/Frequently/Always)',
        'SMOKE': 'Smoking status (Yes/No)',
        'CH2O': 'Daily water consumption in liters (1-3)',
        'SCC': 'Calorie consumption monitoring (Yes/No)',
        'FAF': 'Physical activity frequency per week (0-3 scale)',
        'TUE': 'Time using technology devices per day (0-2 scale)',
        'CALC': 'Consumption of alcohol (Never/Sometimes/Frequently/Always)',
        'MTRANS': 'Transportation used (Automobile/Bike/Motorbike/Public Transportation/Walking)'
    }
    return descriptions.get(feature, "No description available")

# Function to encode user input
def encode_user_input(user_input, data_dict):
    # Prepare dataframe for prediction
    input_df = pd.DataFrame({}, index=[0])

    # Add numerical features and standardize
    for feature in data_dict["numerical_features"]:
        input_df[feature] = user_input[feature]

    input_df[data_dict["numerical_features"]] = data_dict["scaler"].transform(input_df[data_dict["numerical_features"]])

    # Add categorical features and encode
    for feature in data_dict["categorical_features"]:
        # Get encoded value
        encoder = data_dict["label_encoders"][feature]
        # Some encoders might not have seen certain values during training
        try:
            encoded_value = encoder.transform([user_input[feature]])[0]
        except ValueError:
            # Default to the first class if value is not recognized
            encoded_value = 0
        input_df[feature] = encoded_value

    return input_df

# Function to generate prediction
def generate_prediction(user_input_df, data_dict):
    # Make prediction
    prediction = data_dict["best_model"].predict(user_input_df)[0]
    predicted_class = data_dict["le_target"].inverse_transform([prediction])[0]

    # Generate probabilities
    probabilities = data_dict["best_model"].predict_proba(user_input_df)[0]
    prob_df = pd.DataFrame({
        'Obesity Level': data_dict["obesity_categories"],
        'Probability': probabilities
    })

    return predicted_class, prob_df

# Function to perform SHAP analysis
# Function to perform SHAP analysis
# Function to perform SHAP analysis
# Function to perform SHAP analysis
# Function to perform SHAP analysis
# Function to perform SHAP analysis
# Function to perform SHAP analysis
# Function to perform SHAP analysis
# Function to perform SHAP analysis
def perform_shap_analysis(best_model, input_df):
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(best_model)

        # Feature names
        feature_names = list(input_df.columns)

        # Calculate SHAP values
        shap_values = explainer.shap_values(input_df)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # Multi-class model
            prediction = best_model.predict(input_df)[0]
            # Ensure prediction is a valid index
            class_index = min(int(prediction), len(shap_values) - 1)
            class_index = max(0, class_index)  # Ensure it's not negative
            shap_values_for_instance = shap_values[class_index][0]
        else:
            # Binary classification or regression
            shap_values_for_instance = shap_values[0]
            class_index = 0

        # Ensure we have a flat array
        shap_values_for_instance = np.array(shap_values_for_instance).flatten()

        # Handle dimension mismatch
        if len(shap_values_for_instance) != len(feature_names):
            # Warning message removed as requested

            # If there are more SHAP values than features (one-hot encoding)
            if len(shap_values_for_instance) > len(feature_names):
                values_per_feature = len(shap_values_for_instance) // len(feature_names)
                if values_per_feature * len(feature_names) == len(shap_values_for_instance):
                    aggregated_values = []
                    for i in range(len(feature_names)):
                        start_idx = i * values_per_feature
                        end_idx = start_idx + values_per_feature
                        aggregated_values.append(np.sum(shap_values_for_instance[start_idx:end_idx]))
                    shap_values_for_instance = np.array(aggregated_values)
                else:
                    # Use top features by magnitude
                    top_indices = np.argsort(np.abs(shap_values_for_instance))[::-1][:len(feature_names)]
                    shap_values_for_instance = shap_values_for_instance[top_indices]
            else:
                # More feature names than SHAP values
                feature_names = feature_names[:len(shap_values_for_instance)]

        # Get base value
        base_value = explainer.expected_value
        if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
            try:
                base_value = base_value[class_index]
            except:
                base_value = base_value[0]

        # Create tabs for different visualizations
        shap_tabs = st.tabs(["Feature Importance", "Waterfall Plot", "Decision Plot"])

        # Feature Importance tab
        with shap_tabs[0]:
            st.markdown("#### Feature Importance")
            st.markdown("Shows which features are most important for this prediction.")

            # Create a simple bar chart
            plt.figure(figsize=(10, 6))
            importances = np.abs(shap_values_for_instance)
            indices = np.argsort(importances)[::-1]
            plt.barh(range(len(indices)), importances[indices], color='skyblue')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Absolute SHAP Value (Feature Importance)')
            plt.title('Feature Importance')
            plt.tight_layout()
            st.pyplot(plt)
            plt.clf()

            # Display top 5 features in text
            st.markdown("### Top influencing features:")
            top_features = sorted(zip(feature_names, shap_values_for_instance),
                                  key=lambda x: abs(x[1]), reverse=True)[:5]
            for feature, value in top_features:
                direction = "increases" if value > 0 else "decreases"
                st.markdown(f"- **{feature}**: {direction} risk by {abs(value):.2f}")

        # Waterfall Plot tab
        with shap_tabs[1]:
            st.markdown("#### Waterfall Plot")
            st.markdown(
                "Visualizes how each feature contributes to push the model output from the base value to the final prediction.")

            # Direct waterfall implementation
            plt.figure(figsize=(10, 6))
            # Sort indices by magnitude
            indices = np.argsort(np.abs(shap_values_for_instance))[::-1][:10]  # Top 10 features

            # Plot waterfall
            cumulative = np.zeros(len(indices) + 1)
            cumulative[1:] = np.cumsum(shap_values_for_instance[indices])

            # Add base value
            plt.barh(0, base_value, color='gray')
            plt.text(base_value, 0, f'Base: {base_value:.2f}', ha='left', va='center')

            # Add each feature's contribution
            for i, idx in enumerate(indices):
                plt.barh(i + 1, shap_values_for_instance[idx],
                         left=cumulative[i] + base_value,
                         color='red' if shap_values_for_instance[idx] > 0 else 'blue')
                plt.text(cumulative[i + 1] + base_value, i + 1,
                         f'{feature_names[idx]}: {shap_values_for_instance[idx]:.2f}',
                         ha='left' if shap_values_for_instance[idx] > 0 else 'right',
                         va='center')

            # Final prediction
            plt.barh(len(indices) + 1, 0, left=cumulative[-1] + base_value, color='gray')
            plt.text(cumulative[-1] + base_value, len(indices) + 1,
                     f'Final: {cumulative[-1] + base_value:.2f}',
                     ha='left', va='center')

            plt.yticks(range(len(indices) + 2),
                       ['Base Value'] + [feature_names[i] for i in indices] + ['Final Prediction'])
            plt.title('Waterfall Plot')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(plt)
            plt.clf()

        # Decision Plot tab
        with shap_tabs[2]:
            st.markdown("#### Decision Plot")
            st.markdown("Shows the path from the base value to the final prediction.")

            # Create an alternative decision plot
            plt.figure(figsize=(10, 6))
            # Sort values by importance
            sorted_idx = np.argsort(np.abs(shap_values_for_instance))[::-1][:10]  # Top 10

            # Calculate cumulative values
            sorted_values = shap_values_for_instance[sorted_idx]
            cum_values = np.cumsum(sorted_values)
            cum_values = np.insert(cum_values, 0, 0)  # Start from 0

            # Plot
            for i in range(1, len(cum_values)):
                plt.plot([i - 1, i], [base_value + cum_values[i - 1], base_value + cum_values[i]],
                         marker='o', color='blue' if sorted_values[i - 1] > 0 else 'red')

            plt.axhline(y=base_value, color='gray', linestyle='--', label='Base Value')
            plt.text(0, base_value, f'Base: {base_value:.2f}', va='bottom', ha='left')

            # Final value
            plt.text(len(sorted_idx), base_value + cum_values[-1],
                     f'Final: {base_value + cum_values[-1]:.2f}', va='bottom', ha='right')

            plt.xticks(range(len(sorted_idx)),
                       [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
            plt.title('Decision Plot: Path to Prediction')
            plt.xlabel('Features (ordered by importance)')
            plt.ylabel('Prediction Value')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(plt)
            plt.clf()

        # Add explanation of SHAP values
        with st.expander("What are SHAP values?"):
            st.write("""
            SHAP (SHapley Additive exPlanations) values explain how much each feature contributes to the prediction:
            - Higher absolute values mean the feature has a stronger impact on the prediction
            - Positive values push the prediction higher, negative values push it lower
            - These values help identify which patient characteristics most influenced the obesity risk assessment
            """)

    except Exception as e:
        st.error(f"Error in SHAP analysis: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

def main():
    # Title and introduction
    st.title("Obesity Risk Prediction Tool for Physicians")

    # Check if model exists and load it
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please run the training script first.")
        st.info("Run 'python train_model.py' to train and save the model before using this application.")
        return

    # Load saved model and data
    try:
        data_dict = load_saved_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the model has been trained and saved correctly.")
        return

    # Create columns for layout
    left_column, right_column = st.columns([1, 2])

    # Information about the model
    with st.sidebar:
        st.subheader("Model Information")
        st.write(f"Best Model: {data_dict['best_model_name']}")
        st.write(f"Accuracy: {data_dict['model_performance'][data_dict['best_model_name']]:.2%}")

        with st.expander("View All Models Performance"):
            for name, accuracy in data_dict["model_performance"].items():
                st.write(f"{name}: {accuracy:.2%}")

        with st.expander("About This Tool"):
            st.markdown("""
            This tool helps physicians predict a patient's obesity risk level based on lifestyle factors.

            Features used:
            - Demographic information
            - Eating habits
            - Physical activity
            - Transportation methods

            How to use:
            1. Enter patient information in the left panel
            2. Click "Predict" to see results
            3. Review the prediction

            Note: This tool is for informational purposes and should be used as part of a comprehensive clinical assessment.
            """)

    # User input form
    with left_column:
        st.header("Patient Lifestyle Data")

        # Create tabs for different categories of inputs
        demographic_tab, diet_tab, activity_tab, habits_tab = st.tabs(
            ["Demographics", "Diet", "Activity", "Other Habits"])

        with demographic_tab:
            age = st.number_input("Age", min_value=10, max_value=90, value=30, help=get_feature_description("Age"))
            gender = st.radio("Gender", options=["Male", "Female"], horizontal=True,
                              help=get_feature_description("Gender"))
            family_history = st.radio("Family History of Overweight", options=["Yes", "No"], horizontal=True,
                                      help=get_feature_description("family_history_with_overweight"))

        with diet_tab:
            favc = st.radio("Frequent Consumption of High Caloric Food", options=["Yes", "No"], horizontal=True,
                            help=get_feature_description("FAVC"))
            fcvc = st.slider("Frequency of Vegetable Consumption", min_value=1.0, max_value=3.0, value=2.0, step=0.1,
                             help=get_feature_description("FCVC"))
            ncp = st.slider("Number of Main Meals Per Day", min_value=1.0, max_value=4.0, value=3.0, step=0.1,
                            help=get_feature_description("NCP"))
            caec = st.select_slider("Consumption of Food Between Meals",
                                    options=["Never", "Sometimes", "Frequently", "Always"], value="Sometimes",
                                    help=get_feature_description("CAEC"))
            ch2o = st.slider("Daily Water Consumption (Liters)", min_value=1.0, max_value=3.0, value=2.0, step=0.1,
                             help=get_feature_description("CH2O"))
            calc = st.select_slider("Alcohol Consumption", options=["Never", "Sometimes", "Frequently", "Always"],
                                    value="Sometimes", help=get_feature_description("CALC"))
            scc = st.radio("Monitors Calorie Consumption", options=["Yes", "No"], horizontal=True,
                           help=get_feature_description("SCC"))

        with activity_tab:
            faf = st.slider("Physical Activity Frequency (Weekly)", min_value=0.0, max_value=3.0, value=1.0, step=0.1,
                            help=get_feature_description("FAF"))
            mtrans = st.select_slider("Transportation Used Most Frequently",
                                      options=["Automobile", "Bike", "Motorbike", "Public Transportation", "Walking"],
                                      value="Public Transportation",
                                      help=get_feature_description("MTRANS"))

        with habits_tab:
            tue = st.slider("Time Using Technology Devices (Daily Hours)", min_value=0.0, max_value=2.0, value=1.0,
                            step=0.1, help=get_feature_description("TUE"))
            smoke = st.radio("Smoking Status", options=["Yes", "No"], horizontal=True,
                             help=get_feature_description("SMOKE"))

        # Compile all inputs
        user_input = {
            "Age": age,
            "Gender": gender,
            "family_history_with_overweight": family_history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans
        }

        # Prediction button
        if st.button("Predict Obesity Risk", type="primary"):
            # Encode input for prediction
            input_df = encode_user_input(user_input, data_dict)

            # Make prediction
            predicted_class, prob_df = generate_prediction(input_df, data_dict)

            # Display results
            with right_column:
                # Clear any previous content
                right_column.empty()

                # Create header for results
                st.header("Prediction Results")

                # Create columns for prediction and risk level indicator
                pred_col, risk_col = st.columns([3, 1])

                # Map obesity levels to risk categories for visual representation
                risk_mapping = {
                    "Insufficient_Weight": {"level": "Low", "color": "blue"},
                    "Normal_Weight": {"level": "Normal", "color": "green"},
                    "Overweight_Level_I": {"level": "Moderate", "color": "yellow"},
                    "Overweight_Level_II": {"level": "Moderate", "color": "orange"},
                    "Obesity_Type_I": {"level": "High", "color": "red"},
                    "Obesity_Type_II": {"level": "Very High", "color": "darkred"},
                    "Obesity_Type_III": {"level": "Extremely High", "color": "purple"}
                }

                with pred_col:
                    st.subheader("Predicted Obesity Level")
                    st.markdown(
                        f"<h3 style='color: {risk_mapping.get(predicted_class, {}).get('color', 'black')};'>{predicted_class.replace('_', ' ')}</h3>",
                        unsafe_allow_html=True)

                with risk_col:
                    risk_level = risk_mapping.get(predicted_class, {}).get("level", "Unknown")
                    risk_color = risk_mapping.get(predicted_class, {}).get("color", "gray")
                    st.subheader("Risk")
                    st.markdown(f"<h3 style='color: {risk_color};'>{risk_level}</h3>", unsafe_allow_html=True)

                # Display probability distribution
                st.subheader("Probability Distribution")

                # Sort probabilities for better visualization
                prob_df = prob_df.sort_values(by='Probability', ascending=False)

                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.barh(
                    [cat.replace('_', ' ') for cat in prob_df['Obesity Level']],
                    prob_df['Probability'],
                    color=[risk_mapping.get(cat, {}).get('color', 'gray') for cat in prob_df['Obesity Level']]
                )
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.set_title('Obesity Level Probability Distribution')

                # Add percentage labels
                for bar in bars:
                    width = bar.get_width()
                    label_x_pos = width if width > 0.05 else width + 0.02
                    ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.1%}',
                            va='center', ha='left' if width <= 0.05 else 'right',
                            color='black' if width <= 0.05 else 'white')

                st.pyplot(fig)

                # Recommendations section
                st.subheader("Lifestyle Recommendations")

                # Generate simple recommendations based on the prediction
                if predicted_class in ["Normal_Weight", "Insufficient_Weight"]:
                    st.write(
                        "Patient is at a healthy weight or below. Focus on maintaining healthy habits and ensure adequate nutrition.")
                else:
                    st.write("Consider discussing the following lifestyle modifications with the patient:")
                    st.write("- Increase physical activity frequency")
                    st.write("- Increase daily water consumption")
                    st.write("- Increase vegetable consumption")
                    st.write("- Consider adjusting meal frequency")
                    st.write("- Reduce consumption of high-calorie foods")
                    st.write("- Reduce sedentary screen time")
                    st.write("- Reduce snacking between meals")

                st.markdown("---")
                st.caption(
                    "This prediction is based on machine learning analysis of lifestyle factors. It should be used as part of a comprehensive clinical assessment.")

                # Perform SHAP analysis
                st.subheader("SHAP Analysis")
                feature_names = list(user_input.keys())  # Use feature names from user input
                perform_shap_analysis(data_dict["best_model"], input_df)


if __name__ == "__main__":
    main()