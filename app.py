import streamlit as st
import pandas as pd
import requests
import json

# Define the FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/predict_and_optimize/"

# --- Page Configuration ---
st.set_page_config(
    page_title="Dream11 Fantasy Cricket Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- App Title and Description ---
st.title("Dream11 Fantasy Cricket Predictor")
st.markdown("Optimize your Dream11 team with the power of Machine Learning and Integer Linear Programming.")

# --- Sidebar for Player Selection ---
st.sidebar.header("Player Selection")
st.sidebar.markdown("Add players to your team based on their names, teams, roles, and credits.")

# Initialize player list in session state
if 'players' not in st.session_state:
    st.session_state.players = []

player_name = st.sidebar.text_input("Player Name")
player_team = st.sidebar.selectbox("Team", ["Team1", "Team2"])
player_role = st.sidebar.selectbox("Role", ["Batsman", "Bowler", "All-rounder", "Wicket-keeper"])
player_credit = st.sidebar.slider("Credits", 8.0, 12.0, 9.0, 0.5)

if st.sidebar.button("Add Player"):
    if player_name:
        st.session_state.players.append({
            "player_name": player_name,
            "team": player_team,
            "role": player_role,
            "credit": player_credit,
        })
    else:
        st.sidebar.warning("Please enter a player name.")

if st.sidebar.button("Clear Team"):
    st.session_state.players = []

# --- Main Page Content ---

if st.button("Generate Dream11 Team"):
    if not st.session_state.players:
        st.warning("Please add at least one player to generate a team.")
    else:
        st.info("Generating your optimized team...")
        
        # Prepare data for the API call
        player_data = st.session_state.players
        
        # Make the API call
        try:
            response = requests.post(
                API_URL, 
                data=json.dumps(player_data), 
                headers={'Content-Type': 'application/json'}
            )
            
            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                
                # Check for an error message in the response body
                if "error" in result:
                    st.error(f"An error occurred: {result['error']}")
                else:
                    st.success("🎉 Team successfully generated!")
                    
                    # Display the optimized team
                    st.header("Recommended Dream11 Team")
                    selected_players_df = pd.DataFrame(result["selected_players"])
                    
                    # Reorder and format columns for better display
                    display_cols = ['player_name', 'team', 'role', 'credit', 'predicted_points']
                    selected_players_df = selected_players_df[display_cols].rename(columns={
                        'player_name': 'Player',
                        'team': 'Team',
                        'role': 'Role',
                        'credit': 'Credits',
                        'predicted_points': 'Predicted Points'
                    })
                    
                    st.dataframe(selected_players_df, use_container_width=True)
                    
                    # Display total credits and points
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Total Credits Used", value=f"₹{result['total_credits']:.2f}")
                    with col2:
                        st.metric(label="Total Predicted Points", value=f"{result['total_points']:.2f}")
                        
            else:
                st.error(f"An unexpected error occurred from the backend: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API. Please make sure the FastAPI server is running.")
            st.code("uvicorn src.api.main:app --reload")

# Display current player list
if st.session_state.players:
    st.subheader("Your Current Player Pool")
    players_df = pd.DataFrame(st.session_state.players)
    st.dataframe(players_df, use_container_width=True)