import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Use a relative path to the CSV file in the same project folder
file_path = 'adult_income.csv'

# Check if the file exists
if not os.path.isfile(file_path):
    print(f"File not found: {file_path}")
else:
    # Load the data
    column_names = ['age', 'education', 'occupation', 'relationship', 'income']
    data = pd.read_csv(file_path, header=None, names=column_names, na_values=' ?', skipinitialspace=True)

    # Drop rows with missing values
    data = data.dropna()                                                                                           

    # Encode categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Separate features and target variable
    X = data.drop('income', axis=1)
    y = data['income']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Visualize the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X.columns, class_names=[str(c) for c in clf.classes_], filled=True, rounded=True)
    plt.show()
