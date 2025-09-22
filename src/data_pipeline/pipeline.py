import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def calculate_fantasy_points(row):
    """
    Calculates a single team's fantasy points based on a simplified Dream11 T20 system.
    This function is for demonstration and needs to be tailored to actual player data.
    """
    points = 0
    # Batting Points (simplified for team-level data)
    runs = row['runs']
    points += runs
    # Fours & Sixes bonuses (assuming average distribution)
    points += row['boundaries'] * 1 # Assuming a mix of fours/sixes for simplicity
    
    # Bowling Points
    wickets = row['wickets']
    points += wickets * 25
    
    # Maiden Over bonus (simplified)
    # Assuming one maiden over for every 10 overs bowled
    overs = int(row['overs'])
    points += (overs // 10) * 8
    
    # Wicket haul bonuses
    if wickets >= 5:
        points += 16
    elif wickets == 4:
        points += 8
    elif wickets == 3:
        points += 4
        
    return points

def run_pipeline(input_path, output_path):
    """
    Orchestrates the full data preprocessing and feature engineering pipeline.
    
    Args:
        input_path (str): The file path to the raw data.
        output_path (str): The file path to save the processed data.
    """
    print("Starting data preprocessing pipeline...")
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(input_path)
        print(f"Data loaded from {input_path}")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {input_path}. Please place it there.")
        return

    # --- 2. Feature Engineering ---
    # Create the target variable 'fantasy_points'
    print("Calculating team-level fantasy points...")
    # This assumes team-level stats, which is a big simplification
    # In a real system, you would do this for individual players.
    df['home_team_points'] = df.apply(lambda row: calculate_fantasy_points({
        'runs': row['home_runs'], 
        'wickets': row['home_wickets'], 
        'boundaries': row['home_boundaries'], 
        'overs': row['home_overs']
    }), axis=1)

    df['away_team_points'] = df.apply(lambda row: calculate_fantasy_points({
        'runs': row['away_runs'], 
        'wickets': row['away_wickets'], 
        'boundaries': row['away_boundaries'], 
        'overs': row['away_overs']
    }), axis=1)

    # Now let's create a single 'fantasy_points' column for prediction
    # This is still a simplification; a real system needs player-level data
    df['fantasy_points'] = df['home_team_points'] + df['away_team_points']

    # New feature: a simple ratio of home team score vs away team score
    df['score_ratio'] = df['home_runs'] / (df['away_runs'] + 1e-6) # Add small number to avoid division by zero

    # New feature: home team advantage
    df['home_advantage'] = np.where(df['home_team'] == df['winner'], 1, 0)

    # --- 3. Handle Missing Values ---
    print("Handling missing values...")
    # For simplicity, we'll fill missing numeric values with the mean
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # --- 4. Categorical Encoding ---
    print("Performing one-hot encoding for categorical features...")
    categorical_cols = ['home_team', 'away_team', 'venue_name', 'description']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Drop non-numeric and unnecessary columns
    drop_cols = ['id', 'short_name', 'start_date', 'end_date', 'winner', 
                 'result', 'toss_won', 'description', 'home_key_batsman', 'home_key_bowler',
                 'home_playx1', 'away_playx1', 'away_key_batsman', 'away_key_bowler',
                 'match_days', 'umpire1', 'umpire2', 'tv_umpire', 'referee', 'reserve_umpire',
                 'home_captain', 'away_captain', 'pom', 'highlights', 'super_over', 'points']
    df = df.drop(columns=drop_cols, errors='ignore')

    # Convert remaining non-numeric columns to numeric (if any)
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # --- 5. Save Processed Data ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print("Data pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline('data/raw/ipl_cleaned.csv', 'data/processed/processed_data.csv')
