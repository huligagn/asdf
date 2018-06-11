import numpy as np
import pandas as pd

## Q表
class QLearningTable(object):

    ## 初始化所有参数
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions      # 选择
        self.lr = learning_rate     # 学习率
        self.gamma = reward_decay   # 潜在影响因子，下一步对当前选择的影响因子
        self.epsilon = e_greedy     # epsilon greedy参数
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # Q表


    ## 选择下一步的行动
    def choose_action(self, observation):
        self.check_state_exist(observation)     # 判断当前状态是否存在于Q表，没有则加入
        if np.random.uniform() < self.epsilon:  # epsilon greedy
            state_action = self.q_table.ix[observation, :]  # 获取当前状态
            state_action = state_action.reindex(np.random.permutation(state_action.index))  # 重排序？
            action = state_action.idxmax()      # 获取下一步动作
        else:
            action = np.random.choice(self.actions) # 随机选择下一步动作
        return action


    ## 参数更新的过程
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)  # 判断下一个状态是否存在
        q_predict = self.q_table.ix[s, a]   # 估计收益
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()    # 现实收益
        else:
            q_target = r
        
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)       # 参数更新，向估计收益靠拢


    ## 检测该状态是否存在， 如果没有那么把该状态加入Q表并初始化为0
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )
