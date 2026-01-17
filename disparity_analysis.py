"""
Health Disparity Analysis
Analyzes demographic and geographic health disparities across US states using CDC data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath):
    """
    Load health data and prepare for disparity analysis
    
    Parameters:
    filepath (str): Path to CSV file
    
    Returns:
    pd.DataFrame: Processed dataframe
    """
    df = pd.read_csv(filepath, encoding='latin-1')
    
    # Remove aggregate rows
    df = df[df['State'] != 'ALL'].copy()
    
    # Convert numeric columns
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    
    return df


def extract_demographic_info(measure_name):
    """
    Extract demographic category from measure name
    
    Parameters:
    measure_name (str): Full measure name
    
    Returns:
    dict: Base measure and demographic category
    """
    if ' - ' not in measure_name:
        return {'base_measure': measure_name, 'demographic': 'Overall'}
    
    parts = measure_name.split(' - ', 1)
    return {'base_measure': parts[0], 'demographic': parts[1]}


def categorize_demographic(demographic):
    """
    Categorize demographic into type (race, income, geography, etc.)
    
    Parameters:
    demographic (str): Demographic label
    
    Returns:
    str: Category type
    """
    race_terms = ['White', 'Black', 'Hispanic', 'Asian', 'American Indian', 
                  'Alaska Native', 'Hawaiian', 'Pacific Islander', 'Multiracial']
    
    if any(term in demographic for term in race_terms):
        return 'Race/Ethnicity'
    elif '$' in demographic or 'Less Than' in demographic:
        return 'Income'
    elif demographic in ['Metro', 'Nonmetro']:
        return 'Geography'
    elif demographic in ['Male', 'Female']:
        return 'Gender'
    elif 'Ages' in demographic or 'Age' in demographic:
        return 'Age'
    elif demographic in ['College Grad', 'High School', 'Less Than High School', 
                        'Some Post-High School', 'GED']:
        return 'Education'
    elif demographic in ['LGBQ+', 'Straight']:
        return 'Sexual Orientation'
    elif 'Difficulty' in demographic or 'Without a Disability' in demographic:
        return 'Disability'
    elif demographic in ['Served', 'Not Served']:
        return 'Military Service'
    else:
        return 'Other'


def calculate_disparity_metrics(df, measure, demographic_type):
    """
    Calculate disparity metrics for a given measure across demographic groups
    
    Parameters:
    df (pd.DataFrame): Health data
    measure (str): Base health measure
    demographic_type (str): Type of demographic split
    
    Returns:
    pd.DataFrame: Disparity metrics by state
    """
    # Filter to relevant measures
    measure_data = df[df['Measure'].str.startswith(measure)].copy()
    
    # Extract demographic info
    measure_data['demographic_info'] = measure_data['Measure'].apply(extract_demographic_info)
    measure_data['base_measure'] = measure_data['demographic_info'].apply(lambda x: x['base_measure'])
    measure_data['demographic'] = measure_data['demographic_info'].apply(lambda x: x['demographic'])
    measure_data['demo_category'] = measure_data['demographic'].apply(categorize_demographic)
    
    # Filter to specific demographic type
    demo_data = measure_data[measure_data['demo_category'] == demographic_type]
    
    if len(demo_data) == 0:
        return pd.DataFrame()
    
    # Calculate disparity within each state
    disparity_results = []
    
    for state in demo_data['State'].unique():
        state_data = demo_data[demo_data['State'] == state]
        
        if len(state_data) < 2:
            continue
        
        values = state_data['Value'].dropna()
        
        if len(values) < 2:
            continue
        
        disparity_results.append({
            'State': state,
            'Measure': measure,
            'Demographic_Type': demographic_type,
            'Min_Value': values.min(),
            'Max_Value': values.max(),
            'Disparity_Gap': values.max() - values.min(),
            'Disparity_Ratio': values.max() / values.min() if values.min() > 0 else np.nan,
            'Num_Groups': len(values)
        })
    
    return pd.DataFrame(disparity_results)


def calculate_metro_nonmetro_gap(df):
    """
    Calculate gap between metro and nonmetro areas for each state
    
    Parameters:
    df (pd.DataFrame): Health data
    
    Returns:
    pd.DataFrame: Metro-nonmetro gaps by state
    """
    # Get all measures with metro/nonmetro splits
    metro_measures = df[df['Measure'].str.contains(' - Metro', na=False)]
    
    results = []
    
    for state in df['State'].unique():
        if state == 'ALL':
            continue
        
        state_metro = metro_measures[metro_measures['State'] == state]
        
        for measure in state_metro['Measure'].str.replace(' - Metro', '').unique():
            metro_val = df[(df['State'] == state) & (df['Measure'] == f'{measure} - Metro')]['Value']
            nonmetro_val = df[(df['State'] == state) & (df['Measure'] == f'{measure} - Nonmetro')]['Value']
            
            if len(metro_val) > 0 and len(nonmetro_val) > 0:
                metro_v = metro_val.iloc[0]
                nonmetro_v = nonmetro_val.iloc[0]
                
                if pd.notna(metro_v) and pd.notna(nonmetro_v):
                    results.append({
                        'State': state,
                        'Measure': measure,
                        'Metro_Value': metro_v,
                        'Nonmetro_Value': nonmetro_v,
                        'Gap': abs(metro_v - nonmetro_v)
                    })
    
    return pd.DataFrame(results)


def calculate_racial_disparity(df):
    """
    Calculate racial health disparities by state
    
    Parameters:
    df (pd.DataFrame): Health data
    
    Returns:
    pd.DataFrame: Racial disparity metrics
    """
    race_categories = ['White', 'Black', 'Hispanic', 'Asian', 'American Indian/Alaska Native']
    
    results = []
    
    # Focus on key health outcomes
    key_measures = ['Premature Death', 'Obesity', 'Diabetes', 'High Blood Pressure', 
                   'Depression', 'Uninsured']
    
    for measure in key_measures:
        measure_data = df[df['Measure'].str.startswith(measure)].copy()
        
        for state in df['State'].unique():
            if state == 'ALL':
                continue
            
            state_data = measure_data[measure_data['State'] == state]
            
            # Get values for each race
            race_values = {}
            for race in race_categories:
                race_measure = f'{measure} - {race}'
                race_data = state_data[state_data['Measure'] == race_measure]['Value']
                
                if len(race_data) > 0 and pd.notna(race_data.iloc[0]):
                    race_values[race] = race_data.iloc[0]
            
            if len(race_values) >= 2:
                values = list(race_values.values())
                results.append({
                    'State': state,
                    'Measure': measure,
                    'Min_Value': min(values),
                    'Max_Value': max(values),
                    'Racial_Gap': max(values) - min(values),
                    'Num_Races': len(race_values)
                })
    
    return pd.DataFrame(results)


def calculate_income_disparity(df):
    """
    Calculate health disparities across income levels
    
    Parameters:
    df (pd.DataFrame): Health data
    
    Returns:
    pd.DataFrame: Income-based health disparities
    """
    income_categories = ['Less Than $25,000', '$25,000-$49,999', '$50,000-$74,999',
                        '$75,000-$99,999', '$100,000-$149,999', '$150,000 or More']
    
    results = []
    
    key_measures = ['Obesity', 'Diabetes', 'Smoking', 'Depression', 'High Blood Pressure',
                   'Avoided Care Due to Cost']
    
    for measure in key_measures:
        for state in df['State'].unique():
            if state == 'ALL':
                continue
            
            income_values = {}
            
            for income in income_categories:
                income_measure = f'{measure} - {income}'
                income_data = df[(df['State'] == state) & (df['Measure'] == income_measure)]['Value']
                
                if len(income_data) > 0 and pd.notna(income_data.iloc[0]):
                    income_values[income] = income_data.iloc[0]
            
            if len(income_values) >= 2:
                values = list(income_values.values())
                results.append({
                    'State': state,
                    'Measure': measure,
                    'Lowest_Income_Value': income_values.get('Less Than $25,000', np.nan),
                    'Highest_Income_Value': income_values.get('$150,000 or More', np.nan),
                    'Income_Gap': max(values) - min(values),
                    'Num_Income_Levels': len(income_values)
                })
    
    return pd.DataFrame(results)


def create_composite_disparity_index(metro_gaps, racial_gaps, income_gaps):
    """
    Create composite disparity index combining multiple dimensions
    
    Parameters:
    metro_gaps (pd.DataFrame): Geographic disparities
    racial_gaps (pd.DataFrame): Racial disparities  
    income_gaps (pd.DataFrame): Income disparities
    
    Returns:
    pd.DataFrame: Composite disparity scores by state
    """
    # Calculate average gaps by state
    metro_avg = metro_gaps.groupby('State')['Gap'].mean().reset_index()
    metro_avg.columns = ['State', 'Avg_Metro_Gap']
    
    racial_avg = racial_gaps.groupby('State')['Racial_Gap'].mean().reset_index()
    racial_avg.columns = ['State', 'Avg_Racial_Gap']
    
    income_avg = income_gaps.groupby('State')['Income_Gap'].mean().reset_index()
    income_avg.columns = ['State', 'Avg_Income_Gap']
    
    # Merge all together
    composite = metro_avg.merge(racial_avg, on='State', how='outer')
    composite = composite.merge(income_avg, on='State', how='outer')
    
    # Standardize each component
    scaler = StandardScaler()
    
    for col in ['Avg_Metro_Gap', 'Avg_Racial_Gap', 'Avg_Income_Gap']:
        if col in composite.columns:
            valid_data = composite[col].dropna()
            if len(valid_data) > 0:
                composite[f'{col}_Standardized'] = scaler.fit_transform(
                    composite[[col]].fillna(composite[col].mean())
                )
    
    # Calculate composite index (average of standardized components)
    std_cols = [c for c in composite.columns if '_Standardized' in c]
    composite['Composite_Disparity_Index'] = composite[std_cols].mean(axis=1)
    
    # Rank states
    composite['Disparity_Rank'] = composite['Composite_Disparity_Index'].rank(ascending=False)
    
    return composite.sort_values('Composite_Disparity_Index', ascending=False)


if __name__ == '__main__':
    # Load data
    print("Loading data...")
    df = load_and_prepare_data('us_health_2025.csv')
    
    print("\nCalculating disparity metrics...")
    
    # Calculate different disparity types
    metro_gaps = calculate_metro_nonmetro_gap(df)
    print(f"Metro-nonmetro gaps calculated: {len(metro_gaps)} records")
    
    racial_gaps = calculate_racial_disparity(df)
    print(f"Racial disparities calculated: {len(racial_gaps)} records")
    
    income_gaps = calculate_income_disparity(df)
    print(f"Income disparities calculated: {len(income_gaps)} records")
    
    # Create composite index
    print("\nCreating composite disparity index...")
    composite = create_composite_disparity_index(metro_gaps, racial_gaps, income_gaps)
    
    print("\nTop 10 states by health disparity:")
    print(composite[['State', 'Composite_Disparity_Index', 'Disparity_Rank']].head(10))
    
    # Save results
    metro_gaps.to_csv('metro_nonmetro_gaps.csv', index=False)
    racial_gaps.to_csv('racial_disparities.csv', index=False)
    income_gaps.to_csv('income_disparities.csv', index=False)
    composite.to_csv('composite_disparity_index.csv', index=False)
    
    print("\nResults saved to CSV files")
