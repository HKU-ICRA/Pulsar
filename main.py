import os
import sys
from mpi4py import MPI
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
    
    # Start evaluator process
    eval_comm  = MPI.COMM_SELF.Spawn(sys.executable,
                                     args=['evaluator.py', str(0)],
                                     maxprocs=1
                                    )
    eval_intv_steps = 100
    
    # Start coordinator loop
    opponent_coordinator = dict()
    while True:
        for idx in range(n_agents):
            player = rmleague.get_player(idx)
            opponent, tf = player.get_match()
            opponent_coordinator[idx] = opponent
            for ai in range(actor_lower_bounds[idx], actor_upper_bounds[idx]):
                sub_comms[idx].send(opponent.get_agent(), dest=ai)
        for idx in range(n_agents):
            player = rmleague.get_player(idx)
            opponent = opponent_coordinator[idx]
            for ai in range(actor_lower_bounds[idx], actor_upper_bounds[idx]):
                traj_outcome = sub_comms[idx].recv()
                rmleague.update(player, opponent, traj_outcome['outcome'])
                if player.ready_to_checkpoint():
                    rmleague.add_player(player.checkpoint())
        for idx in range(n_agents):
            player = rmleague.get_player(idx)
            new_agent = sub_comms[idx].recv()
            player.set_agent(new_agent)
        # Report progress
        for idx in range(n_agents):
            print("Player", str(idx), "steps:", rmleague.get_player_agent(idx).get_steps())
        print("Players:", len(rmleague._payoff._players))
        print("Payoff games:", rmleague._payoff._games.values())
        print("Payoff wins:", rmleague._payoff._wins.values())
        print("Payoff draws:", rmleague._payoff._draws.values())
        print("Payoff loses:", rmleague._payoff._losses.values())
        sys.stdout.flush()
        # Save progress
        save_rmleague(rmleague, league_file)
        save_main_player(rmleague, os.path.join(os.getcwd(), 'data', 'main_player'))
        # Run eval if main agent has enough steps
        if rmleague.get_player_agent(0).get_steps() >= eval_intv_steps:
            main_player_agent = rmleague.get_player_agent(0)
            eval_comm.send(main_player_agent, dest=0)
            eval_intv_steps += eval_intv_steps


def save_rmleague(rmleague, league_file):
    with open(league_file, 'wb') as f:
        pickle.dump(rmleague, f)


def load_rmleague(league_file):
    if os.path.isfile(league_file):
        with open(league_file, 'rb') as f:
            return pickle.load(f)
    else:
        return League()


def save_main_player(rmleague,  mp_file):
    with open(mp_file, 'wb') as f:
        pickle.dump(rmleague.get_player(0), f)


if __name__ == '__main__':
  main()
  