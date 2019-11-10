class Coordinator:
    """Central worker that maintains payoff matrix and assigns new matches."""

    def __init__(self, league):
        self.league = league

    def send_outcome(self, home_player, away_player, outcome):
        self.league.update(home_player, away_player, outcome)
        if home_player.ready_to_checkpoint():
            self.league.add_player(home_player.checkpoint())
    