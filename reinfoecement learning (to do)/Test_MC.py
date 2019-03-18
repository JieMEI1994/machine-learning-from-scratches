from GridWorld import GridWorld
from MontaCarlo import MontaCarloAgent


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.pick_action(str(observation))

            # RL take action and get next observation and reward
            next_observation, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(next_observation))

            # swap observation
            observation = next_observation      

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = GridWorld()
    RL = MontaCarloAgent(actions=list(range(env.action_space)))

    env.after(100, update)
    env.mainloop()