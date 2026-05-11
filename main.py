import pandas as pd
import numpy as np

# What describes a student's performance? --> Study hours, attendance, assignment marks, quiz marks 

np.random.seed(42)
n = 1000  # 1000 students 

# Create dataset "fake students"
df = pd.DataFrame({
    "study_hours": np.random.normal(8, 3, n).clip(0),
    "attendance": np.random.normal(80, 10, n).clip(0, 100),
    "assignment_average": np.random.normal(75, 15, n).clip(0, 100),
    "quiz_average": np.random.normal(70, 15, n).clip(0, 100),
})

# Create target variable
df["final_grade"] = (
    df["study_hours"] * 2 +
    df["attendance"] * 0.3 +
    df["assignment_average"] * 0.3 +
    df["quiz_average"] * 0.2 +
    np.random.normal(0, 2, n) # Account for random factors 
    # Set how much each factor affects final grade 
    # np.random.normal(mean, standard_deviation, count) 
)

from sklearn.model_selection import train_test_split

X = df.drop("final_grade", axis=1)  # Input for training --> drop final_grade so model can't see the answer
y = df["final_grade"]  # Output (what we are trying to predict)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression  # Create the model 

model = LinearRegression()
model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error, r2_score

predictions = model.predict(X_test)

# Measure Error 
print("Error:",  mean_absolute_error(y_test, predictions))
print("R2 score: ", r2_score(y_test, predictions))

print(model.coef_)

# Make graphs 

import matplotlib.pyplot as plt 

plt.scatter(y_test, predictions)
plt.xlabel("Actual grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades")
plt.show()


# Predict student grades
class_data = pd.read_csv("students(Sheet1).csv")
predicted_grades = model.predict(class_data)
print(predicted_grades)