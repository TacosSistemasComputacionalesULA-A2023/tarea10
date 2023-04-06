import time
import gym
import gym_taco_environments
from agent import ValueIteration

def play(env, agent):
    observation, _ = env.reset()
    env.render()
    time.sleep(1)
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = agent.get_action(observation)
        new_observation, _, terminated, truncated, _ = env.step(action)
        observation = new_observation


if __name__ == "__main__":
    ENVIRONMENT = "Princess-v0"
    env = gym.make(ENVIRONMENT)

    agent = ValueIteration(env.observation_space.n, env.action_space.n, env.P, gamma=0.1)

    agent.solve(
        policy_evaluations=100,
        iterations=100,
        delta=0.9,
        method="politer",
    )
    agent.render()
    env.close()

    env = gym.make(ENVIRONMENT, render_mode="human")
    play(env, agent)
    env.close()
