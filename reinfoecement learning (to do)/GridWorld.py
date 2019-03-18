import numpy as np
import time
import sys
# Graphical User Interface package
import tkinter as tk

# numbers of grid
GRIDSIZE = 40
# screen size
WORLD_HIGHT = 9
WORLD_WIGHT = 9


class GridWorld(tk.Tk, object):
    def __init__(self):
        super(GridWorld, self).__init__()
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.action_space = len(self.actions)
        self.title('GridWorld')
        self.geometry('{0}x{1}'.format(WORLD_HIGHT * GRIDSIZE, WORLD_WIGHT * GRIDSIZE))
        self._build_maze()

    def BbuildMaze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=WORLD_HIGHT * GRIDSIZE,
                           width=WORLD_WIGHT * GRIDSIZE)

        # borderlines
        for column in range(0, WORLD_WIGHT * GRIDSIZE, GRIDSIZE):
            x0, y0, x1, y1 = column, 0, column, WORLD_WIGHT* GRIDSIZE
            self.canvas.create_line(x0, y0, x1, y1)
        for row in range(0, WORLD_HIGHT * GRIDSIZE, GRIDSIZE):
            x0, y0, x1, y1 = 0, row, WORLD_HIGHT * GRIDSIZE, row
            self.canvas.create_line(x0, y0, x1, y1)

        # center
        center = np.array([20, 20])

        # obstacle 1
        obstacle1_center = center + np.array([GRIDSIZE * 0, GRIDSIZE * 4])
        self.obstacle1 = self.canvas.create_rectangle(
            obstacle1_center[0] - 15, obstacle1_center[1] - 15,
            obstacle1_center[0] + 15, obstacle1_center[1] + 15,
            fill='black')
        # obstacle 2
        obstacle2_center = center + np.array([GRIDSIZE * 1, GRIDSIZE * 4])
        self.obstacle2 = self.canvas.create_rectangle(
            obstacle2_center[0] - 15, obstacle2_center[1] - 15,
            obstacle2_center[0] + 15, obstacle2_center[1] + 15,
            fill='black')
        # obstacle 3
        obstacle3_center = center + np.array([GRIDSIZE * 2, GRIDSIZE * 4])
        self.obstacle3 = self.canvas.create_rectangle(
            obstacle3_center[0] - 15, obstacle3_center[1] - 15,
            obstacle3_center[0] + 15, obstacle3_center[1] + 15,
            fill='black')
        # obstacle 4
        obstacle4_center = center + np.array([GRIDSIZE * 3, GRIDSIZE * 4])
        self.obstacle4 = self.canvas.create_rectangle(
            obstacle4_center[0] - 15, obstacle3_center[1] - 15,
            obstacle4_center[0] + 15, obstacle3_center[1] + 15,
            fill='black')
        # obstacle 5
        obstacle5_center = center + np.array([GRIDSIZE * 4, GRIDSIZE * 4])
        self.obstacle5 = self.canvas.create_rectangle(
            obstacle5_center[0] - 15, obstacle3_center[1] - 15,
            obstacle5_center[0] + 15, obstacle3_center[1] + 15,
            fill='black')
        # obstacle 6
        obstacle6_center = center + np.array([GRIDSIZE * 5, GRIDSIZE * 4])
        self.obstacle6 = self.canvas.create_rectangle(
            obstacle6_center[0] - 15, obstacle3_center[1] - 15,
            obstacle6_center[0] + 15, obstacle3_center[1] + 15,
            fill='black')        
        # obstacle 7
        obstacle7_center = center + np.array([GRIDSIZE * 6, GRIDSIZE * 4])
        self.obstacle7 = self.canvas.create_rectangle(
            obstacle7_center[0] - 15, obstacle3_center[1] - 15,
            obstacle7_center[0] + 15, obstacle3_center[1] + 15,
            fill='black') 
        # obstacle 8
        obstacle8_center = center + np.array([GRIDSIZE * 7, GRIDSIZE * 4])
        self.obstacle8 = self.canvas.create_rectangle(
            obstacle8_center[0] - 15, obstacle3_center[1] - 15,
            obstacle8_center[0] + 15, obstacle3_center[1] + 15,
            fill='black') 
        
        
        # tresure
        oval_center = center + np.array([GRIDSIZE * 0, GRIDSIZE * 8])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # agent
        self.rect = self.canvas.create_rectangle(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        center = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        state = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if state[1] > GRIDSIZE:
                base_action[1] -= GRIDSIZE
        elif action == 1:   # down
            if state[1] < (WORLD_HIGHT- 1) * GRIDSIZE:
                base_action[1] += GRIDSIZE
        elif action == 2:   # right
            if state[0] < (WORLD_WIGHT  - 1) * GRIDSIZE:
                base_action[0] += GRIDSIZE
        elif action == 3:   # left
            if state[0] > GRIDSIZE:
                base_action[0] -= GRIDSIZE

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_state = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_state == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif next_state in [self.canvas.coords(self.obstacle1), 
                            self.canvas.coords(self.obstacle2), 
                            self.canvas.coords(self.obstacle3),
                            self.canvas.coords(self.obstacle4),
                            self.canvas.coords(self.obstacle5),
                            self.canvas.coords(self.obstacle6),
                            self.canvas.coords(self.obstacle7),
                            self.canvas.coords(self.obstacle8)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return next_state, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        state = env.reset()
        while True:
            env.render()
            action = 1
            state, reward, done = env.step(action)
            if done:
                break

if __name__ == '__main__':
    env = GridWorld()
    env.after(100, update)
    env.mainloop()

