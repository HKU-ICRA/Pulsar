import os
import sys
from mpi4py import MPI

from rmleague.league import League
from environment.envs.simple import make_env
from environment.envhandler import EnvHandler


def main():
  """Trains the AlphaStar league."""
  rmleague = League()

  return

  n_agents = 1
  for idx in range(n_agents):
    sub_comm  = MPI.COMM_SELF.Spawn_multiple([sys.executable, sys.executable],
                                 args=[['actor.py', str(idx), str(0.95), str(0.99)], ['learner.py', str(idx)]],
                                 maxprocs=[1, 1])
    
  return
  # Start all processes
  learner_ps, actor_ps = [], []
  for l in learners:
    l_p = Process(target=l.run())
    learner_ps.append(l_p.start())
  for a in actors:
    a_p = Process(target=a.run())
    actor_ps.append(a_p.start())

  # Wait for training to finish.
  for l_p in learner_ps:
    l_p.join()
  for a_p in actor_ps:
    a_p.join()

if __name__ == '__main__':
  main()
  