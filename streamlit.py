import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('exams.csv')

# Perform data preprocessing (similar to the previous steps)

# Define the features (X) and target variable (y)
#X = df.drop('math score', axis=1)
X = df[['parental level of education','reading score','writing score']]
y = df['math score']

# Perform one-hot encoding on the features
#X_encoded = pd.get_dummies(X, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])
X_encoded = pd.get_dummies(X, columns=['parental level of education'])

# Create an instance of the Linear Regression model
model = LinearRegression()

# Train the model on the entire dataset
model.fit(X_encoded, y)

# Streamlit app title and description
st.title("Student Performance Predictor")
st.markdown("<p style='font-size:24px; color:yellow;'>Welcome to the Student Performance Predictor app!</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px; color:yellow;'>This app uses machine learning techniques to predict mathematics scores based on selected features. You can choose to either make predictions or visualize the data.</p>", unsafe_allow_html=True)

# Additional Streamlit app code if needed
# ...

# Sidebar inputs for prediction
st.sidebar.header("Input Features")

#gender = st.sidebar.selectbox("Gender", df['gender'].unique())
#race = st.sidebar.selectbox("Race/Ethnicity", df['race/ethnicity'].unique())
parent_education = st.sidebar.selectbox("Parental Level of Education", df['parental level of education'].unique())
#lunch = st.sidebar.selectbox("Lunch", df['lunch'].unique())
#test_prep = st.sidebar.selectbox("Test Preparation Course", df['test preparation course'].unique())
reading_score = st.sidebar.slider("Reading Score", min_value=0, max_value=100, step=1, value=50)
writing_score = st.sidebar.slider("Writing Score", min_value=0, max_value=100, step=1, value=50)

# Process the prediction
if st.sidebar.button('Predict'):
    if not (reading_score and writing_score and parent_education):
        st.sidebar.warning("Please enter all features.")
    else:
        # Create a DataFrame from the input data
        input_data = {
            #'gender': [gender],
            #'race/ethnicity': [race],
            'parental level of education': [parent_education],
            #'lunch': [lunch],
            #'test preparation course': [test_prep],
            'reading score': [reading_score],
            'writing score': [writing_score]
        }
        input_df = pd.DataFrame(input_data)

        # Perform one-hot encoding on the input DataFrame
        #input_df_encoded = pd.get_dummies(input_df, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])
        input_df_encoded = pd.get_dummies(input_df, columns=['parental level of education'])
        # Ensure input DataFrame has the same columns as the training data
        input_df_encoded = input_df_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        # Make predictions using the pre-trained model
        predicted_math_score = model.predict(input_df_encoded)

        # Calculate the accuracy of the model's predictions (R-squared)
        y_pred_train = model.predict(X_encoded)
        accuracy = r2_score(y, y_pred_train)

        # Display the predicted math score
        st.subheader("Predicted Math Score")
        st.write(f"{predicted_math_score[0]:.2f}%")

        # Display the accuracy of the prediction
        st.subheader("Model Accuracy")
        st.write(f"{accuracy*100:.2f}%")

# Sidebar inputs for visualization
st.sidebar.header("Visualizations")

# Create a form for visualization
with st.sidebar.form(key='visualization_form'):
    selected_visualization = st.sidebar.selectbox("Select Visualization", ["Distribution of Math Scores", "Math Score vs. Writing Score"])

    # Visualization button
    visualization_button = st.form_submit_button('Visualize')

# Display visualizations based on user selection
if visualization_button:
    if selected_visualization == "Distribution of Math Scores":
        # Display histogram of math scores
        st.subheader("Distribution of Math Scores")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df, x='math score', kde=True)
        plt.xlabel("Math Score")
        st.pyplot(fig)

    elif selected_visualization == "Math Score vs. Writing Score":
        # Display scatter plot of math score vs. writing score with gender coloring
        st.subheader("Math Score vs. Writing Score with Gender Coloring")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x='math score', y='writing score', hue='gender', palette=['blue', 'orange'], ax=ax)
        plt.xlabel("Math Score")
        plt.ylabel("Writing Score")
        st.pyplot(fig)

