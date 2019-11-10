def main():
  """Trains the AlphaStar league."""
  league = League(
      initial_agents={
          race: get_supervised_agent(race)
          for race in ("Protoss", "Zerg", "Terran")
      })
  coordinator = Coordinator(league)
  learners = []
  actors = []
  for idx in range(12):
    player = league.get_player(idx)
    learner = Learner(player)
    actors.extend([ActorLoop(player, coordinator) for _ in range(16000)])

  for l in learners:
    l.run()
  for a in actors:
    a.run()

  # Wait for training to finish.
  join()

if __name__ == '__main__':
  main()
  