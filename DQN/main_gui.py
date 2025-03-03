# import sys
# from os import path
# sys.path.append(path.join(path.abspath('.'), 'gym_game'))
# sys.path.append(path.join(path.abspath('.'), 'gym_game','env'))
# sys.path.append(path.join(path.abspath('.'), 'gym_game','gui'))


# import tensorflow as tf
# # from tensorflow.compat import v1 as tfv1
# import numpy as np
# from experience_replay import ExpReplay
# from custom_env import CustomEnv
# from ui import Gui_environment
# from QNET import QNET
# from DQN.epsilon_methods import exponential_decay_method, decrement_epsilon
# import csv
# import datetime
# from matplotlib import pyplot as plt
# import pandas as pd
# import seaborn as sns

# dict_actions = {0:'FORWARD', 1:'TURN_LEFT', 2:'TURN_RIGHT', 3: 'GRAB', 4:'SHOOT'}

# class DeepQAgent:
#     def __init__(self, env, hidden_units=256):
#         # set hyper parameters
#         self.max_episodes = 20
#         #self.max_actions = 1000
#         self.exploration_rate = 1.0
#         self.exploration_decay = 0.995
#         self.epsilon_min = 0.1  
        
#         # set environment
#         self.env = env
#         self.states = env.observation_space.n
#         self.actions = env.action_space.n
#         self.max_actions = self.env.environment.board.max_steps
        
#         # Experience Replay for batch learning
#         self.exp = ExpReplay()
#         # the number of experience per batch for batch learning
#         self.batch_size = 64 
        
#         # Deep Q Network
#         self.qnet = QNET(self.states, self.actions, self.exp)
#         # For execute Deep Q Network
#         session = tf.compat.v1.InteractiveSession()
#         session.run(tf.compat.v1.global_variables_initializer())
#         #self.qnet.set_session(session)
#         self.qnet.session = session

#         self.gui_env = Gui_environment(self.env.environment.board.size_env)


#     def train(self):
#         #RECORD INFO EPISODES
#         list_info_train = []
#         info_train = {'rewards': None,'episode': None, 'step_per_episode': None,
#                     'has_gold': None, 'killed_wumpus': None, 'get_gold_and_return_home': None}
#         # set hyper parameters
#         max_episodes = self.max_episodes
#         max_actions = self.max_actions
#         exploration_rate = self.exploration_rate
#         exploration_decay = self.exploration_decay
#         batch_size = self.batch_size
        
#         # start training
#         grab_gold = 0
#         has_gold_safe_home = 0
#         killed_wumpus = 0

#         self.env.render()
#         self.gui_env.update_pos_board(self.env)

#         for i in range(int(max_episodes)):
#             print(f'Episode {i}')
#             # self.gui_env.update_board(episode=i)
#             amount_steps_environment = 0
#             total_rewards = 0
#             grab_gold = 0
#             killed_wumpus = 0
#             has_gold_safe_home = 0
#             info_train = {'rewards': None,'episode': None, 'step_per_episode': None,
#                     'has_gold': None, 'killed_wumpus': None, 'get_gold_and_return_home': None}

#             state = self.env.reset()
#             state = state.reshape((1, self.states))

#             for j in range(max_actions):
#                 self.gui_env.update_pos_board(self.env)
#                 #self.env.render() # Uncomment this line to render the environment
#                 action = self.qnet.get_action(state, exploration_rate)
#                 next_state, reward, done, info = self.env.step(dict_actions[action])
#                 next_state = next_state.reshape((1, self.states))

#                 total_rewards += reward
#                 amount_steps_environment += 1

#                 if done:
#                     self.exp.add(state, action, reward, next_state, done)
#                     self.qnet.batch_train(batch_size)

#                     #record information of the episode
#                     grab_gold = 1 if self.env.environment.board.components['Agent'].has_gold else 0
#                     killed_wumpus = 1 if not self.env.environment.board.components['Agent'].wumpus_alive else 0
#                     if next_state[0][0] == 0 and self.env.environment.board.components['Agent'].has_gold:
#                         has_gold_safe_home = 1
#                     info_train['episode'] = i
#                     info_train['rewards'] = total_rewards
#                     info_train['has_gold'] = grab_gold
#                     info_train['step_per_episode'] = amount_steps_environment
#                     info_train['killed_wumpus'] = killed_wumpus
#                     info_train['get_gold_and_return_home'] = has_gold_safe_home
#                     list_info_train.append(info_train)

#                     #break
#                     return info_train
                    
#                 self.exp.add(state, action, reward, next_state, done)
#                 self.qnet.batch_train(batch_size)
                
#                 # next episode
#                 state = next_state
#             print(f'Reward of episode {i}: {total_rewards}\nEpsilon: {exploration_rate}')

#             #Update exploration rate
#             # exploration_rate = 0.01 + (exploration_rate-0.01)*np.exp(-exploration_decay*(i+1))
#             # exploration_rate = decrement_epsilon(exploration_rate, self.epsilon_min, self.exploration_decay)
#             exploration_rate = exponential_decay_method(i, max_episodes, self.epsilon_min)
#             self.gui_env.update_board(i, total_rewards)
#         self.write_executions(list_info_train)    
    
#     def write_executions(self, infos_train: list):
#         headers = ['episode', 'rewards', 'has_gold', 'step_per_episode', 'killed_wumpus', 'get_gold_and_return_home']

#         directory = path.join(path.abspath('.'), 'gym_game\DQN\executions')
#         with open(path.abspath(path.join(directory, file_name)), 'w') as file:
#             writer = csv.DictWriter(file, fieldnames=headers)
#             writer.writeheader()
#             writer.writerows(infos_train)

#     def read_executions(self):
#         date = '2022-10-15'
#         file_name_date_exec = f'dqn_execution_{dim}x{dim}-{date}.csv'
#         directory = path.join(path.abspath('.'), 'gym_game\DQN\executions\\', file_name_date_exec)
#         data = pd.read_csv(directory)

#         return data

#     def graph(self):
#         data = self.read_executions()

#         # moving average
#         window = 100
#         data['moving_average'] = data.rewards.rolling(window).mean()
    
#         sns.lineplot(x = 'episode', y='rewards', data=data, label='Reward per episodes')
#         sns.lineplot(x='episode', y='moving_average', data=data, label='Move Average Rewards')
#         plt.xlabel('Episodes')
#         plt.ylabel('Rewards')
#         plt.grid()
#         plt.savefig(f'graph_rewards_dqn_{dim}x{dim}.png')
#         # plt.show()
    
#     def save_model(self):
#         saver = tf.compat.train.Saver()
#         name_model = 'model_dqn'
#         directory = path.join(path.abspath('.'), 'gym_game\DQN\models\\', name_model)
#         saver.save(self.qnet.session, directory)
    
#     def load_model(self):
#         self.qnet.session = tf.compat.Session()
#         name_model = 'model_dqn.meta'
#         directory = path.join(path.abspath('.'), 'gym_game\DQN\models\\', name_model)
#         saver = tf.compat.train.import_meta_graph(directory)
#         saver.restore(self.qnet.session, tf.compat.train.latest_checkpoint('.\\'))

# def reset_data():
#     directory = path.join(path.abspath('.'), 'gym_game\DQN\executions\\', file_name)
#     open(directory,"wb").close()

# if __name__ == '__main__':
#     tf.compat.disable_eager_execution()
#     dict_max_steps = {4: 100, 8: 150, 10: 200} #size environment is key and value is amount max steps
#     dict_values_seed = {4: 123, 8: 99, 10: 917} #size environment is key and value is values seed
#     dim = 4
#     date_execution = datetime.datetime.now().strftime('%Y-%m-%d')
#     file_name = f'dqn_execution_{dim}x{dim}-{date_execution}.csv'
#     reset_data()
#     env = CustomEnv(nrow=dim,ncol=dim, max_steps=dict_max_steps[dim], value_seed=dict_values_seed[dim])
#     agent = DeepQAgent(env)
#     agent.train()
#     # agent.graph()