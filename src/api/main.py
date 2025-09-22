import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from ..optimization.optimizer import TeamOptimizer
import os

# Define the model path relative to the project root
MODEL_PATH = 'mlruns/latest_model/random_forest_model.pkl'

# A Pydantic model to define the structure of a player's data in the API request
class Player(BaseModel):
    player_name: str
    team: str
    role: str
    credit: float
    # The 'predicted_points' will be populated by our model, not from the request
    predicted_points: Optional[float] = None

# A Pydantic model for the API response (the optimized team)
class OptimizedTeamResponse(BaseModel):
    selected_players: List[Player]
    total_credits: float
    total_points: float

app = FastAPI(
    title="Dream11 Fantasy Cricket Predictor",
    description="An ML system to predict player points and optimize team selection."
)

# Load the trained model on startup
# This saves time and resources by loading the model only once
try:
    with open(MODEL_PATH, 'rb') as f:
        model = joblib.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None

@app.post("/predict_and_optimize/", response_model=OptimizedTeamResponse)
def predict_and_optimize_team(players: List[Player]):
    """
    Takes a list of players, predicts their fantasy points, and returns an
    optimized 11-player team based on Dream11 constraints.
    """
    if model is None:
        return {"error": "Model is not loaded. Check model path."}

    # 1. Prepare player data for prediction
    players_df = pd.DataFrame([player.dict() for player in players])
    
    # 2. Predict fantasy points using the loaded model
    # We must match the features used in training
    # For this example, let's assume the request includes these features.
    # In a real app, you would have to fetch/generate these features.
    # For now, we'll use placeholder data to demonstrate the flow.
    players_df['runs'] = players_df['player_name'].apply(lambda x: len(x))
    players_df['wickets'] = players_df['team'].apply(lambda x: len(x))
    players_df['fours'] = players_df['role'].apply(lambda x: len(x))
    players_df['sixes'] = players_df['credit'].apply(lambda x: x)
    players_df['catches'] = players_df['player_name'].apply(lambda x: len(x))
    
    # Run prediction
    predicted_points = model.predict(players_df[['runs', 'wickets', 'fours', 'sixes', 'catches']])
    players_df['predicted_points'] = predicted_points

    # 3. Use the optimization engine to select the best team
    optimizer = TeamOptimizer(players_df)
    optimized_team_df = optimizer.optimize_team()
    
    # 4. Format the response
    selected_players = [
        Player(
            player_name=row['player_name'],
            team=row['team'],
            role=row['role'],
            credit=row['credit'],
            predicted_points=row['predicted_points']
        )
        for index, row in optimized_team_df.iterrows()
    ]

    total_credits = optimized_team_df['credit'].sum()
    total_points = optimized_team_df['predicted_points'].sum()

    return OptimizedTeamResponse(
        selected_players=selected_players,
        total_credits=total_credits,
        total_points=total_points
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
