import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(
    page_title="Dream11 Fantasy Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title('🏏 Dream11 Fantasy Cricket Predictor')
st.markdown("---")
st.subheader("Powered by an MLOps Pipeline")
st.markdown("""
This application uses a machine learning model to predict player performance and an
optimization engine to recommend the best possible fantasy cricket team under Dream11 constraints.
""")

# --- User Input Form ---
st.header("Player Data Input")
st.markdown("Enter player details for two teams to get an optimized team recommendation.")

# Let's provide a pre-filled example for demonstration purposes
example_players = [
    {"player_name": "Rohit Sharma", "team": "MI", "role": "Batsman", "credit": 9.5},
    {"player_name": "Jasprit Bumrah", "team": "MI", "role": "Bowler", "credit": 9.0},
    {"player_name": "Hardik Pandya", "team": "MI", "role": "All-rounder", "credit": 10.0},
    {"player_name": "Suryakumar Yadav", "team": "MI", "role": "Batsman", "credit": 9.0},
    {"player_name": "Ishan Kishan", "team": "MI", "role": "Wicket-keeper", "credit": 8.5},
    {"player_name": "Virat Kohli", "team": "RCB", "role": "Batsman", "credit": 10.5},
    {"player_name": "Glenn Maxwell", "team": "RCB", "role": "All-rounder", "credit": 9.0},
    {"player_name": "Faf du Plessis", "team": "RCB", "role": "Batsman", "credit": 9.5},
    {"player_name": "Mohammed Siraj", "team": "RCB", "role": "Bowler", "credit": 8.5},
    {"player_name": "Dinesh Karthik", "team": "RCB", "role": "Wicket-keeper", "credit": 8.0},
    {"player_name": "Devdutt Padikkal", "team": "RCB", "role": "Batsman", "credit": 8.5},
    {"player_name": "Adam Zampa", "team": "RCB", "role": "Bowler", "credit": 7.5},
    {"player_name": "Kyle Jamieson", "team": "RCB", "role": "Bowler", "credit": 8.0},
    {"player_name": "Yuzvendra Chahal", "team": "RCB", "role": "Bowler", "credit": 8.5},
    {"player_name": "Kieron Pollard", "team": "MI", "role": "All-rounder", "credit": 9.5},
    {"player_name": "Trent Boult", "team": "MI", "role": "Bowler", "credit": 8.5},
    {"player_name": "Krunal Pandya", "team": "MI", "role": "All-rounder", "credit": 8.0},
    {"player_name": "Quinton de Kock", "team": "MI", "role": "Wicket-keeper", "credit": 9.0},
]
# Create a DataFrame from the example players
players_df = pd.DataFrame(example_players)
edited_df = st.data_editor(players_df, num_rows="dynamic", use_container_width=True)

# --- Button to Trigger Prediction ---
if st.button('Generate Optimized Team', help="Click to send data to the backend API and get the best team."):
    if len(edited_df) < 11:
        st.error("Please enter at least 11 players to generate a team.")
    else:
        with st.spinner('Generating team recommendation...'):
            try:
                # Convert the edited DataFrame to a list of dicts for the API call
                player_data = edited_df.to_dict('records')
                
                # Make the POST request to the FastAPI backend
                response = requests.post(
                    "http://localhost:8000/predict_and_optimize/", # Adjust URL if deploying
                    data=json.dumps(player_data),
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("🎉 Team successfully generated!")
                    
                    st.header("Recommended Dream11 Team")
                    st.dataframe(pd.DataFrame(result['selected_players']), use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("Summary")
                    st.write(f"**Total Predicted Points:** {result['total_points']:.2f}")
                    st.write(f"**Total Credits Used:** {result['total_credits']:.2f}")
                    
                else:
                    st.error(f"Error from backend: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Connection error. Is the FastAPI backend running?")
            except json.JSONDecodeError:
                st.error("Invalid JSON response from the backend.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
