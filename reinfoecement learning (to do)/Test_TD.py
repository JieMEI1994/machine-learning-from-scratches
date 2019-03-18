from GridWorld import GridWorld
from TD import TDAgent


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.pick_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            next_observation, reward, done = env.step(action)

            # RL choose action based on next observation
            next_action = RL.pick_action(str(next_observation))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(next_observation), next_action)

            # swap observation and action
            observation = next_observation
            action = next_action

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = GridWorld()
    RL = TDAgent(actions=list(range(env.action_space)))

    env.after(100, update)
    env.mainloop()

