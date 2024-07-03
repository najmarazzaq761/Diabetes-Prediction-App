import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"]=[10,5]
warnings.simplefilter(action="ignore",category=FutureWarning)

st.title("Diabetes Prediction App")
st.write("Dataset is:")
df=pd.read_csv("diabetes.csv")
st.write(df.head())
st.write("Number iof rows:",df.shape[0])
st.write("Number iof columns:",df.shape[1])
st.header("Data Visualization")
st.write("### 1.Distribution plots")

st.write("#### 1.1 histogram plot")
fig, ax = plt.subplots()
df["Age"].plot(kind="hist", ax=ax)
st.pyplot(fig)

st.write("#### 1.2 kde plot")
fig, ax = plt.subplots()
sns.kdeplot(data=df, x="Age", ax=ax)
st.pyplot(fig)

st.write("### 2.relationl plots")
st.write("#### 2.1 Line Plot outcome vs glucose")
x_axis="Outcome"
y_axis="Glucose"
st.line_chart(df[[x_axis, y_axis]])

st.write("####  2.2 Scatter Plot")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="Outcome", y="Glucose", hue="Age", ax=ax)       
st.pyplot(fig)

st.write("### 3.Categorical Plots")
st.write("#### 3.1 Bar plots")
x_axis="Outcome"
# y_axis="Glucose"
st.bar_chart(df[x_axis].value_counts())


st.write("#### 3.2 strip plots")
fig, ax = plt.subplots()
sns.stripplot(x="Outcome", y="Age", data=df, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.stripplot(x="Outcome", y="BloodPressure", data=df, ax=ax)
st.pyplot(fig)

st.write("#### 3.3 Swarm plot")
fig, ax = plt.subplots(figsize=(8, 7))
sns.swarmplot(x="Outcome", y="Age", data=df, ax=ax)
st.pyplot(fig)

st.write("#### 3.4 Box plot")
fig, ax = plt.subplots()   
sns.boxplot(data=df, x="Outcome", y="Glucose", ax=ax)
st.pyplot(fig)

st.write("#### 3.5 violin plots")
fig, ax = plt.subplots()   
sns.violinplot(data=df, x="Outcome", y="Glucose", ax=ax)
st.pyplot(fig)


fig, ax = plt.subplots()   
sns.violinplot(data=df, x="Outcome", y="Glucose",hue="Pregnancies", ax=ax)
st.pyplot(fig)

st.write("#### 3.6 count plot")
fig, ax = plt.subplots()
sns.countplot(x="Outcome", data=df, ax=ax) 
st.pyplot(fig)

st.write("#### 3.7 point plot")
fig, ax = plt.subplots(figsize=(6, 6))
sns.pointplot(x="Outcome", y="Glucose", data=df, ax=ax)
st.pyplot(fig)



st.write("#### 4 Regression plots : Regression plots show relation : strong or weak")
st.write("#### 4.1 lmplots")
sns.lmplot(x="Age", y="Pregnancies", data=df, aspect=1.5)
st.pyplot(plt.gcf())

# lmplot for SkinThickness vs. Insulin
sns.lmplot(x="SkinThickness", y="Insulin", data=df, aspect=1.5)
st.pyplot(plt.gcf())

st.write("#### 4.2 Regplot")
fig, ax = plt.subplots()
sns.regplot(x="Glucose", y="Outcome", data=df, ax=ax)
st.pyplot(fig)
st.write("### 5 Matrix plots")
st.write("#### heatmap")

# Filter the DataFrame to include only numeric columns
numeric_df = df.select_dtypes(include=[float, int])
# Calculate the correlation matrix
correlation = numeric_df.corr()
# Plot the heatmap
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(correlation, cmap="BrBG", annot=True, ax=ax)
# Display the plot in the Streamlit app
st.pyplot(fig)

# Split data to be used in the models
# Create matrix of features
x=df.drop('Outcome',axis=1)
y=df["Outcome"]
st.write("training data",x.shape)
st.write("target variavle",y.head(5))

# Feature Scaling
from sklearn import preprocessing
pre_process = preprocessing.StandardScaler()
x_transform = pre_process.fit_transform(x)

# Verify transformation
x_transform_df = pd.DataFrame(x_transform, columns=x.columns)
st.write("Input Features After Transformation")
st.write(x_transform_df.head())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_transform_df, y, test_size = .10, random_state = 101)
st.write("training input data",x_train.shape)
st.write("testing input data",x_test.shape)
st.write("training output data",y_train.shape)
st.write("testing outpupt data",y_test.shape)


model_options = ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier", "Gradient Boosting Classifier", "Naive Bayes", "SVC"]
selected_model = st.selectbox("Select a model", model_options)

if selected_model == "Logistic Regression":
        model = LogisticRegression()
elif selected_model == "Random Forest Classifier":
        model = RandomForestClassifier()
elif selected_model == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
elif selected_model == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
elif selected_model == "Naive Bayes":
        model = GaussianNB()
elif selected_model == "SVC":
        model = SVC()

# Train the selected model
model.fit(x_train, y_train)
predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
st.write("### Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")

from sklearn.metrics import precision_score,recall_score,confusion_matrix
precision=precision_score(y_test,predictions)
recall=recall_score(y_test,predictions)
st.write(f'precision:{precision:.2f}')
st.write(f'recall:{recall:.2f}')
st.write("---"*30)
confusion=confusion_matrix(y_test,predictions)
st.write(confusion)
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(confusion,annot=True,fmt="d", ax=ax)
# Display the plot in the Streamlit app
st.pyplot(fig)

