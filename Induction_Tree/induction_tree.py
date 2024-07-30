import os
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

def calculate_entropy(df, target_column):
    """
    Calculates entropy (I(p,n)) for a given dataset and target column.
    
    Parameters:
    - df: DataFrame containing the dataset.
    - target_column: Name of the target column (e.g., 'income').
    
    Returns:
    - entropy: Calculated entropy value.
    """
    p = len(df[df[target_column] == '<=50K'])
    n = len(df[df[target_column] == '>50K'])
    total = p + n
    
    if p == 0 or n == 0:
        entropy = 0
    else:
        entropy = - (p/total) * math.log2(p/total) - (n/total) * math.log2(n/total)
    
    return entropy

def calculate_individual_entropy(df, attribute, target_column):
    """
    Calculates individual entropy (I(attribute)) for a given attribute in the dataset.
    
    Parameters:
    - df: DataFrame containing the dataset.
    - attribute: Name of the attribute column for which to calculate entropy.
    - target_column: Name of the target column (e.g., 'income').
    
    Returns:
    - individual_entropy: Calculated individual entropy value.
    """
    attribute_values = df[attribute].unique()
    individual_entropy = 0
    
    for value in attribute_values:
        subset = df[df[attribute] == value]
        subset_size = len(subset)
        
        p = len(subset[subset[target_column] == '<=50K'])
        n = len(subset[subset[target_column] == '>50K'])
        total = p + n
        
        if p == 0 or n == 0:
            entropy = 0
        else:
            entropy = - (p/total) * math.log2(p/total) - (n/total) * math.log2(n/total)
        
        individual_entropy += (subset_size / len(df)) * entropy
    
    return individual_entropy

def calculate_weighted_entropy(df, attribute, target_column):
    """
    Calculates the weighted entropy (E(attribute)) for a given attribute in the dataset.
    
    Parameters:
    - df: DataFrame containing the dataset.
    - attribute: Name of the attribute column for which to calculate entropy.
    - target_column: Name of the target column (e.g., 'income').
    
    Returns:
    - weighted_entropy: Calculated weighted entropy value.
    """
    attribute_values = df[attribute].unique()
    weighted_entropy = 0
    
    for value in attribute_values:
        subset = df[df[attribute] == value]
        subset_size = len(subset)
        
        p = len(subset[subset[target_column] == '<=50K'])
        n = len(subset[subset[target_column] == '>50K'])
        total = p + n
        
        if p == 0 or n == 0:
            entropy = 0
        else:
            entropy = - (p/total) * math.log2(p/total) - (n/total) * math.log2(n/total)
        
        weighted_entropy += (subset_size / len(df)) * entropy
    
    return weighted_entropy

if __name__ == "__main__":
    # Load your CSV file into a DataFrame
    filename = 'adult_income.csv'
    df = pd.read_csv(filename)
    
    # Specify the target column for which you want to build the decision tree
    target_column = 'income'
    
    # Specify the columns (attributes) to consider for splitting
    columns = ['age', 'education', 'occupation', 'relationship']
    
    # Calculate entropy for the target column
    entropy = calculate_entropy(df, target_column)
    print(f"Entropy (I(p,n)) = {entropy:.2f}")
    print()
    
    # Calculate and print individual entropies (I(attribute))
    for attribute in columns:
        individual_entropy = calculate_individual_entropy(df, attribute, target_column)
        print(f"{attribute}:")
        print(f"I(pi, ni) = {individual_entropy:.2f}")
        
        # Calculate and print weighted entropy (E(attribute)) and information gain
        weighted_entropy = calculate_weighted_entropy(df, attribute, target_column)
        gain = entropy - weighted_entropy
        print(f"E({attribute}) = {weighted_entropy:.2f}")
        print(f"Gain({attribute}) = {gain:.2f}")
        
        print()
    
    # Prepare data for classification
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    X = df.drop('income', axis=1)
    y = df['income']
    
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
