class Dream11Constraints:
    """
    Defines the rules and constraints for building a Dream11 fantasy cricket team.
    """
    def __init__(self):
        # Maximum total credits for the team
        self.max_credits = 100
        
        # Total number of players in the team
        self.team_size = 11
        
        # Player role constraints (min and max players for each role)
        self.min_wk = 1
        self.max_wk = 4
        self.min_bat = 3
        self.max_bat = 6
        self.min_ar = 1
        self.max_ar = 4
        self.min_bowl = 3
        self.max_bowl = 6
        
        # Player-per-team constraint
        self.max_players_per_team = 7
