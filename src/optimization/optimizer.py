import pulp
import pandas as pd
from .constraints import Dream11Constraints

class TeamOptimizer:
    """
    Optimizes a fantasy cricket team selection using Integer Linear Programming.
    """
    def __init__(self, players_df):
        """
        Args:
            players_df (pd.DataFrame): DataFrame containing player data
                                       with columns: 'player_name', 'team',
                                       'role', 'credit', 'predicted_points'.
        """
        self.players_df = players_df
        self.constraints = Dream11Constraints()

    def optimize_team(self):
        """
        Solves the ILP problem to find the optimal team lineup.

        Returns:
            pd.DataFrame: DataFrame of the selected 11 players.
        """
        print("Starting team optimization using Integer Linear Programming...")
        
        # 1. Create the ILP problem
        prob = pulp.LpProblem("Dream11_Team_Optimization", pulp.LpMaximize)

        # 2. Define Decision Variables
        # A binary variable for each player: 1 if selected, 0 otherwise
        player_vars = pulp.LpVariable.dicts(
            "player", self.players_df.index, cat='Binary')

        # 3. Objective Function: Maximize total predicted points
        prob += pulp.lpSum(
            [player_vars[i] * self.players_df.loc[i, 'predicted_points'] 
             for i in self.players_df.index]
        ), "Total_Fantasy_Points"

        # 4. Define Constraints

        # Constraint 1: Total team size must be exactly 11 players
        prob += pulp.lpSum(
            [player_vars[i] for i in self.players_df.index]
        ) == self.constraints.team_size, "Team_Size_Constraint"

        # Constraint 2: Total credits must not exceed the limit
        prob += pulp.lpSum(
            [player_vars[i] * self.players_df.loc[i, 'credit'] 
             for i in self.players_df.index]
        ) <= self.constraints.max_credits, "Max_Credits_Constraint"

        # Constraint 3: Player role distribution
        for role in ['Wicket-keeper', 'Batsman', 'All-rounder', 'Bowler']:
            prob += pulp.lpSum(
                [player_vars[i] for i in self.players_df.index 
                 if self.players_df.loc[i, 'role'] == role]
            ) >= getattr(self.constraints, f'min_{role.lower().replace("-", "")}'), f"Min_{role}_Constraint"
            
            prob += pulp.lpSum(
                [player_vars[i] for i in self.players_df.index 
                 if self.players_df.loc[i, 'role'] == role]
            ) <= getattr(self.constraints, f'max_{role.lower().replace("-", "")}'), f"Max_{role}_Constraint"

        # Constraint 4: Maximum players from a single team
        for team in self.players_df['team'].unique():
            prob += pulp.lpSum(
                [player_vars[i] for i in self.players_df.index 
                 if self.players_df.loc[i, 'team'] == team]
            ) <= self.constraints.max_players_per_team, f"Max_Players_From_{team}_Constraint"
        
        # 5. Solve the problem
        prob.solve()
        
        # 6. Extract the optimal team
        selected_players = self.players_df.loc[
            [i for i in self.players_df.index if player_vars[i].varValue == 1]
        ]
        
        return selected_players
