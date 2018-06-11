import numpy as np
import pandas as pd
import time

## 各个参数
N_STATES = 6                    # 状态的个数
ACTIONS = ['left', 'right']     # 所有选择
EPSILON = 0.9                   # epsilon-greedy的超参，0.9的概率按价值选择，0.1的概率随机选择
ALPHA = 0.1                     # 学习率
GAMMA = 0.9                     # 潜在影响因子，下一步对当前选择的影响因子
MAX_EPISODES = 13               # 迭代次数
FRESH_TIME = 0.3                # UI 刷新时间


## 建立Q表，全零初始化
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )

    return table


## 依据Q表进行选择，epsilon-greedy方式
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name


## 依据状态和选择获取下一步状态和奖励
def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1

    return S_, R


## UI显示
def update_env(S, episoode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode {}: total_steps = {}'.format(episoode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                       ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


## Q-Learning算法
def RL():
    q_table = build_q_table(N_STATES, ACTIONS)                  # 建立Q表
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)                       # 做出选择
            S_, R = get_env_feedback(S, A)                      # 计算下一步的状态和奖励
            q_predict = q_table.loc[S, A]                       # 当前选择的估计价值？
            if S_ != 'terminal':
                q_target = R + GAMMA*q_table.iloc[S_, :].max()  # 当前选择的潜在价值？
            else:
                q_target = R
                is_terminated = True

            q_table.loc[S, A] += ALPHA*(q_target - q_predict)   # 参数更新
            S = S_                                              # 状态更新

            update_env(S, episode, step_counter+1)              # UI更新
            step_counter += 1

    return q_table


if __name__ == '__main__':
    q_table = RL()
    print('\r\n Q-table:\n')
    print(q_table)