import os
import sys
import tensorflow as tf
from mpi4py import MPI
from datetime import datetime
import pickle

from rmleague.league import League
from environment.envs.simple import make_env
from environment.envhandler import EnvHandler


def main():
    """
    Trains the AlphaStar league.
    This main function acts as the coordinator for rmleague.
    """
    league_file = os.path.join(os.getcwd(), 'data', 'league')
    rmleague = load_rmleague(league_file)

    n_agents = 3
    actor_procs = 1
    learner_procs = 1

    sub_comms = []
    actor_lower_bounds = []
    actor_upper_bounds = []

    # Start agent and learner processes
    for idx in range(n_agents):
        actor_lower_bound = 0
        actor_upper_bound = actor_procs
        learner_bound = actor_procs + learner_procs - 1

        actor_lower_bounds.append(actor_lower_bound)
        actor_upper_bounds.append(actor_upper_bound)

        sub_comm  = MPI.COMM_SELF.Spawn_multiple([sys.executable, sys.executable],
                                                 args=[
                                                        ['actor.py', str(idx), str(0.95), str(0.99), str(learner_bound)],
                                                        ['learner.py', str(idx), str(actor_lower_bound), str(actor_upper_bound)]
                                                      ],
                                                 maxprocs=[actor_procs, learner_procs]
                                                )
        sub_comms.append(sub_comm)

        agent = rmleague.get_player_agent(idx)
        sub_comm.send(agent, dest=learner_bound)
    
    # Start evaluator process (No eval during development so commented out)
    '''
    eval_comm  = MPI.COMM_SELF.Spawn(sys.executable,
                                     args=['evaluator.py', str(0)],
                                     maxprocs=1
                                    )
    eval_intv_steps = 100
    '''

    # Setup summary writer
    dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(os.getcwd(), "data", "logs", dt_string)
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Start coordinator loop
    opponent_coordinator = dict()
    while True:
        # Send opponent to Actor
        for idx in range(n_agents):
            player = rmleague.get_player(idx)
            opponent, match_bool = player.get_match()
            opponent_coordinator[idx] = opponent
            for ai in range(actor_lower_bounds[idx], actor_upper_bounds[idx]):
                sub_comms[idx].send(opponent.get_agent(), dest=ai)
        # Receive trajectory outcome from Actor
        for idx in range(n_agents):
            player = rmleague.get_player(idx)
            opponent = opponent_coordinator[idx]
            for ai in range(actor_lower_bounds[idx], actor_upper_bounds[idx]):
                traj_outcome = sub_comms[idx].recv()
                rmleague.update(player, opponent, traj_outcome['outcome'])
                if player.ready_to_checkpoint():
                    rmleague.add_player(player.checkpoint())
        # Receive new agent and ppo summaries from Learner
        for idx in range(n_agents):
            # Update agent of player
            player = rmleague.get_player(idx)
            results = sub_comms[idx].recv()
            player.set_agent(results['agent'])
            player.incre_updates()
            # Report and log each Player's PPO details
            tf.summary.scalar(player.get_name()+':approxkl', results['approxkl'], step=player.get_updates())
            tf.summary.scalar(player.get_name()+':loss', results['loss'], step=player.get_updates())
            tf.summary.scalar(player.get_name()+':entropy', results['entropy'], step=player.get_updates())
            tf.summary.scalar(player.get_name()+':return', results['return'], step=player.get_updates())
            tf.summary.scalar(player.get_name()+':total_steps', player.get_agent().get_steps(), step=player.get_updates())
        # Save progress
        save_rmleague(rmleague, league_file)
        save_main_player(rmleague, os.path.join(os.getcwd(), 'data', 'main_player'))
        # Run eval if main agent has enough steps
        if rmleague.get_player_agent(0).get_steps() >= rmleague.get_eval_intv_steps():
            # Run eval process
            main_player_agent = rmleague.get_player_agent(0)
            eval_comm.send(main_player_agent, dest=0)
            # Report and log rmleague
            tf.summary.scalar('no. of rmleague players', len(rmleague._payoff._players), step=rmleague.get_eval_intv_steps())
            for i, p1 in enumerate(rmleague.get_payoff_players()):
                for j, p2 in enumerate(rmleague.get_payoff_players()):
                    if i != j:
                        winrate = rmleague.get_winrate(p1, p2)
                        tf.summary.scalar(p1.get_name()+' vs '+p2.get_name(), winrate, step=rmleague.get_eval_intv_steps())
            rmleague.incre_eval_intv_steps()
        # Report end of while loop
        print("Coordinator looped")
        sys.stdout.flush()


def save_rmleague(rmleague, league_file):
    with open(league_file, 'wb') as f:
        pickle.dump(rmleague, f)


def load_rmleague(league_file):
    if os.path.isfile(league_file):
        with open(league_file, 'rb') as f:
            return pickle.load(f)
    else:
        rmleague = League()
        rmleague.set_eval_intv_steps(100000000) # Basically never during development
        return rmleague


def save_main_player(rmleague,  mp_file):
    with open(mp_file, 'wb') as f:
        pickle.dump(rmleague.get_player(0), f)


if __name__ == '__main__':
  main()
  