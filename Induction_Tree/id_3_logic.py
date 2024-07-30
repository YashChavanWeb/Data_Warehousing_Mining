import pandas as pd
import math

def calculate_entropy(df, target_column):
    """
    Calculates entropy (I(p,n)) for a given dataset and target column.
    
    Parameters:
    - df: DataFrame containing the dataset.
    - target_column: Name of the target column (e.g., 'income').
    
    Returns:
    - entropy: Calculated entropy value.
    """
    # Count number of instances where target column is in each class
    p = len(df[df[target_column] == '<=50K'])
    n = len(df[df[target_column] == '>50K'])
    
    # Total number of instances
    total = p + n
    
    # Calculate entropy I(p,n)
    if p == 0 or n == 0:
        entropy = 0  # If either p or n is 0, entropy is 0 (perfectly classified)
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
    # Calculate unique values of the attribute
    attribute_values = df[attribute].unique()
    
    individual_entropy = 0
    
    # Calculate entropy for each value of the attribute
    for value in attribute_values:
        subset = df[df[attribute] == value]
        subset_size = len(subset)
        
        # Calculate p and n for this subset
        p = len(subset[subset[target_column] == '<=50K'])
        n = len(subset[subset[target_column] == '>50K'])
        
        # Total instances in this subset
        total = p + n
        
        # Calculate entropy for this subset
        if p == 0 or n == 0:
            entropy = 0
        else:
            entropy = - (p/total) * math.log2(p/total) - (n/total) * math.log2(n/total)
        
        # Weighted sum
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
    # Calculate unique values of the attribute
    attribute_values = df[attribute].unique()
    
    weighted_entropy = 0
    
    # Calculate entropy for each value of the attribute
    for value in attribute_values:
        subset = df[df[attribute] == value]
        subset_size = len(subset)
        
        # Calculate p and n for this subset
        p = len(subset[subset[target_column] == '<=50K'])
        n = len(subset[subset[target_column] == '>50K'])
        
        # Total instances in this subset
        total = p + n
        
        # Calculate entropy for this subset
        if p == 0 or n == 0:
            entropy = 0
        else:
            entropy = - (p/total) * math.log2(p/total) - (n/total) * math.log2(n/total)
        
        # Weighted sum
        weighted_entropy += (subset_size / len(df)) * entropy
    
    return weighted_entropy

if __name__ == "__main__":
    # Load your CSV file into a DataFrame
    filename = 'adult_income.csv'
    df = pd.read_csv(filename)
    
    # Specify the target column for which you want to build the decision tree
    target_column = 'income'
    
    # Specify the columns (attributes) to consider for splitting
    # Here, we select 4 columns apart from the target column 'income'
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
