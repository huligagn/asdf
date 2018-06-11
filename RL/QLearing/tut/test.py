from maze_env import Maze
from RL_brain import QLearningTable


## 整个算法的过程
def update():
    for _ in range(100):
        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(str(observation))     # 选择下一步行动

            observation_, reward, done = env.step(action)   # 获取下一步状态、奖励、是否到达

            RL.learn(str(observation), action, reward, str(observation_))   # 参数更新过程

            observation = observation_                      # 状态更新

            if done:
                break

    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()