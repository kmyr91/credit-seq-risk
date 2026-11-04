# 2. Let's redo Step 5 without giving you the solution
# Open:

# credit-seq-risk/src/data/audit.py

# Your assignment is to write a function named audit_statements.
# Function requirements
# Input: A pandas DataFrame containing two columns:
# - customer_ID: identifies each customer.
# - S_2: date of a monthly statement.
# For example:
# customer_ID	S_2
# A	2026-01-01
# A	2026-02-01
# A	2026-03-01
# B	2026-01-01
# B	2026-02-01

# Output: A dictionary with these six values:
# Key	Expected result
# n_rows	5
# n_customers	2
# duplicate_customer_dates	0
# min_statements	2
# max_statements	3
# median_statements	2.5

# What I want you to implement
# 1. Import pandas.
# 2. Define a function accepting a DataFrame.
# 3. Check that both required columns exist. Raise ValueError otherwise.
# 4. Calculate the total number of rows.
# 5. Calculate the number of unique customers.
# 6. Count duplicated combinations of customer ID and statement date.
# 7. Calculate how many statements each customer has.
# 8. Find the minimum, maximum, and median of those counts.
# 9. Return the six results as a dictionary.

import pandas as pd


def audit_statements(df):
    # Check that both required columns exist.
    required_columns = {'customer_id', 'statement_date'}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            'The DataFrame must have the columns '
            '"customer_id" and "statement_date".'
        )

    # Calculate the total number of rows.
    n_rows = df.shape[0]

    # Calculate the number of unique customers.
    n_unique_customers = df['customer_id'].nunique()

    # Count duplicated combinations of customer ID and statement date.
    n_duplicated_combinations = df.duplicated(
        subset=['customer_id', 'statement_date']
    ).sum()

    # Calculate how many statements each customer has.
    n_statements_per_customer = df['customer_id'].value_counts()

    # Find the minimum, maximum, and median of those counts.
    min_statements = n_statements_per_customer.min()
    max_statements = n_statements_per_customer.max()
    median_statements = n_statements_per_customer.median()

    return {
        'n_rows': n_rows,
        'n_unique_customers': n_unique_customers,
        'n_duplicated_combinations': n_duplicated_combinations,
        'min_statements': min_statements,
        'max_statements': max_statements,
        'median_statements': median_statements
    }
