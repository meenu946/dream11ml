import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
import pulp
from src.optimization.optimizer import TeamOptimizer
from sklearn.preprocessing import StandardScaler

# --- MLOps Setup ---
# Use absolute paths to ensure the files are found regardless of the working directory
MODEL_PATH = r"C:\Users\Admin\Desktop\dream11ml\notebooks\src\models\best_model.pkl"
PROCESSED_DATA_PATH = r"C:\Users\Admin\Desktop\dream11ml\data\processed\processed_data.csv"

# A Pydantic model to define the structure of a player's data in the API request
class Player(BaseModel):
    player_name: str
    team: str
    role: str
    credit: float

# A Pydantic model for the API response (the optimized team)
class OptimizedTeamResponse(BaseModel):
    selected_players: List[dict]
    total_credits: float
    total_points: float

app = FastAPI(
    title="Dream11 Fantasy Cricket Predictor",
    description="An ML system to predict player points and optimize team selection."
)

# --- Load the ML model and data schema on startup ---
try:
    # Use joblib to load the model artifact directly
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Load a sample of the processed data to get the exact feature names and order
    # This is the most crucial part for ensuring the backend matches the training data
    processed_df_sample = pd.read_csv(PROCESSED_DATA_PATH)
    processed_df_features = processed_df_sample.drop(columns=['fantasy_points'], errors='ignore')
    
    # Get the list of all feature columns the model was trained on
    MODEL_FEATURES = processed_df_features.columns.tolist()

    # Create a pre-fitted scaler object from the processed data
    # In a real pipeline, you would save and load this scaler as well.
    scaler = StandardScaler()
    scaler.fit(processed_df_features.select_dtypes(include=np.number))

except FileNotFoundError:
    print(f"Error: Required files not found. Check paths: {MODEL_PATH} and {PROCESSED_DATA_PATH}")
    model = None
except Exception as e:
    print(f"Error during startup: {e}")
    model = None

# --- Mock Player Data for Demo ---
# In a real app, this would come from a database or a live source.
mock_players_data = {
    'player_name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'],
    'team': ['Team1'] * 10 + ['Team2'] * 10,
    'role': ['Batsman', 'Batsman', 'Bowler', 'Bowler', 'Wicket-keeper', 'All-rounder', 'Batsman', 'Bowler', 'All-rounder', 'Batsman'] * 2,
    'credit': [9.5, 9.0, 8.5, 8.0, 10.0, 10.5, 9.0, 8.5, 9.0, 9.5] * 2,
    'runs': [50, 60, 5, 2, 35, 45, 70, 1, 25, 55, 65, 35, 15, 2, 40, 50, 60, 10, 30, 40],
    'wickets': [0, 0, 3, 2, 0, 1, 0, 4, 1, 0, 0, 0, 1, 3, 0, 1, 0, 2, 0, 1],
    'boundaries': [5, 6, 0, 0, 3, 4, 8, 0, 2, 5, 7, 3, 1, 0, 4, 5, 6, 0, 3, 4],
    'overs': [0, 0, 4, 4, 0, 3, 0, 4, 2, 0, 0, 0, 1, 3, 0, 1, 0, 2, 0, 1]
}
mock_players_df = pd.DataFrame(mock_players_data)


@app.post("/predict_and_optimize/")
def predict_and_optimize_team(players: List[Player]):
    """
    Takes a list of players, predicts their fantasy points, and returns an
    optimized 11-player team based on Dream11 constraints.
    """
    if model is None:
        return {"error": "Model is not loaded. Please check the model file path."}

    # 1. Prepare player data for prediction
    players_df = mock_players_df.copy()

    # --- Feature Engineering (must match the pipeline) ---
    players_df['score_ratio'] = players_df['runs'] / (players_df['wickets'] + 1e-6)
    players_df['home_advantage'] = 0

    # --- One-hot Encoding (must match the pipeline) ---
    all_teams_in_training = ['Team1', 'Team2']
    encoded_features = pd.get_dummies(players_df['team'], prefix='team')
    
    for team in all_teams_in_training:
        team_col_name = f'team_{team}'
        if team_col_name not in encoded_features.columns:
            encoded_features[team_col_name] = 0
    
    players_df = pd.concat([players_df, encoded_features], axis=1)

    # --- Scale numerical features (must use the pre-fitted scaler) ---
    numerical_features_to_scale = players_df.select_dtypes(include=np.number).columns.tolist()
    players_df[numerical_features_to_scale] = scaler.transform(players_df[numerical_features_to_scale])
    
    # --- Select and reorder columns to match the model's training schema ---
    final_features_df = players_df[MODEL_FEATURES]

    # 2. Predict fantasy points
    players_df['predicted_points'] = model.predict(final_features_df)

    # 3. Use the optimization engine to select the best team
    optimizer = TeamOptimizer(players_df)
    optimized_team_df = optimizer.optimize_team()
    
    # Check if a feasible solution was found
    if pulp.LpStatus[optimizer.prob.status] != 'Optimal':
        return {"error": "Could not find a feasible team. Please adjust player selection or constraints."}

    # 4. Format the response
    selected_players = optimized_team_df.to_dict('records')

    total_credits = optimized_team_df['credit'].sum()
    total_points = optimized_team_df['predicted_points'].sum()

    return OptimizedTeamResponse(
        selected_players=selected_players,
        total_credits=total_credits,
        total_points=total_points
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
