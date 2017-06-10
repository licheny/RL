from gui import Maze
import pandas as pd
import numpy as np

#初始化基本值
learning_rate=0.01
reward_decay=0.9
e_greedy=0.9
actions=[0,1,2,3]
#创建Qtable
qtable=pd.DataFrame(columns=actions)
#选择运动方向
def select_action(observation):
    global learning_rate
    global reward_decay
    global e_greedy
    global actions
    global qtable
    qtable_exists(observation)
    if np.random.uniform() < e_greedy:#选择目前Q值最大的方向走
            state_action = qtable.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
    else:#随机选择一个方向走
            action = np.random.choice(actions)
    return action

#学习算法更新
def qtable_update(s,a,r,s_):
    global learning_rate
    global reward_decay
    global e_greedy
    global actions
    global qtable
    qtable_exists(s_)
    q_predict=qtable.ix[s,a]
    if s_!="terminal":
        q_target=r+reward_decay*qtable.ix[s_,:].max()
    else:
        q_target=r
    qtable.ix[s,a]=qtable.ix[s,a]+learning_rate*(q_target-q_predict)

#检查Qtable是否有当前项
def qtable_exists(state):
    global learning_rate
    global reward_decay
    global e_greedy
    global actions
    global qtable
    if state not in qtable.index:
        qtable = qtable.append(
                pd.Series(
                    [0]*len(actions),
                    index=qtable.columns,
                    name=state,
                ) 
        )

#多次迭代求解
def main_update():
    for i in range(100):
        observation=envi.reset()#初始化页面，返回起始位置
        while True:
            envi.render()
            action=select_action(str(observation))
            observation_,reward,done=envi.step(action)
            qtable_update(str(observation),action,reward,str(observation_))
            observation=observation_
            if done:
                break
    print("game over")
    envi.destroy()




if __name__=="__main__": 
    envi=Maze()  
    envi.after(100,main_update)
    envi.mainloop()