class ActorLoop:
    """A single actor loop that generates trajectories.
    We don't use batched inference here, but it was used in practice.
    """

    def __init__(self, player, coordinator):
        self.player = player
        self.environment = make_env()
        self.coordinator = coordinator

    def run(self):
        while True:
            opponent = self.player.get_match()
            trajectory = []
            start_time = time()  # in seconds.
            while time() - start_time < 60 * 60:
                home_observation, away_observation, is_final, z = self.environment.reset()
                student_state = self.player.initial_state()
                opponent_state = opponent.initial_state()

                while not is_final:
                    student_action, student_logits, student_state = self.player.step(home_observation, student_state)
                    # We mask out the logits of unused action arguments.
                    action_masks = get_mask(student_action)
                    opponent_action, _, _ = opponent.step(away_observation, opponent_state)
                    teacher_logits = self.teacher(observation, student_action, teacher_state)

                    observation, is_final, rewards = self.environment.step(student_action, opponent_action)
                    trajectory.append(Trajectory(
                                    observation=home_observation,
                                    opponent_observation=away_observation,
                                    state=student_state,
                                    is_final=is_final,
                                    behavior_logits=student_logits,
                                    teacher_logits=teacher_logits,
                                    masks=action_masks,
                                    action=student_action,
                                    z=z,
                                    reward=rewards,
                                    ))

                    if len(trajectory) > TRAJECTORY_LENGTH:
                        trajectory = stack_namedtuple(trajectory)
                        self.learner.send_trajectory(trajectory)
                        trajectory = []
                self.coordinator.send_outcome(student, opponent, self.environment.outcome())
