import os
import pickle


"""
    This function formats the saving of rmleague and saves it accordingly
"""
def Save(rmleague, save_dir, league_file):
    """
        This function saves the whole rmleague
        Args:
            rmleague: the rmleague class object
            save_dir: the directory to save everything to
            league_file: the path of the file that holds the most updated rmleague
    """
    def save_rmleague(rmleague, lf):
        with open(lf, 'wb') as f:
            pickle.dump(rmleague, f)
    # Save the most updated rmleague
    league_file_tmp = league_file + "_tmp"
    save_rmleague(rmleague, league_file_tmp)
    if os.path.exists(league_file):
        os.remove(league_file)
    os.rename(league_file_tmp, league_file)
    # Save copies of rmleague periodically
    if rmleague.nupdates % 500 == 0:
        rec_league_file = os.path.join(save_dir, 'record', 'rmleague', 'rmleague' + str(rmleague.nupdates))
        save_rmleague(rmleague, rec_league_file)
    # Save the most updated main agent
    def save_player(rmleague,  p_file, player_type, idx):
        main_agent_player = rmleague.get_player(player_type, idx)
        with open(p_file, 'wb') as f:
            pickle.dump(main_agent_player, f)
    main_player_file = os.path.join(save_dir, 'main_player')
    main_player_file_tmp = main_player_file + "_tmp"
    save_player(rmleague, main_player_file_tmp, 'main_agent', 0)
    if os.path.exists(main_player_file):
        os.remove(main_player_file)
    os.rename(main_player_file_tmp, main_player_file)
    # Save copies of rmleague players periodically
    if rmleague.nupdates % 120 == 0:
        for player_type, players in rmleague._learning_agents.items():
            for pidx, player in enumerate(players):
                player_file = os.path.join(save_dir, 'record', 'rmleague_players', player_type + str(pidx) + ":" + str(player.get_agent().get_steps()))
                #save_player(rmleague,
                #            player_file,
                #            player_type,
                #            pidx)
    # Save payoff players periodically
    def save_payoff_player(p_file, player):
        with open(p_file, 'wb') as f:
            pickle.dump(player, f)
    if rmleague.nupdates % 48 == 0:
        pass
        #for payoff_player in rmleague.get_payoff_players():
        #    player_file = os.path.join(save_dir, 'record', 'payoff_players', payoff_player.name)
        #    save_payoff_player(player_file, payoff_player)
