import sys
from os import path
sys.path.append(path.join(path.abspath('.'), 'gym_game'))
sys.path.append(path.join(path.abspath('.'), 'gym_game','env'))

import tensorflow as tf
# from tensorflow.compat import v1 as tfv1
import numpy as np
from experience_replay import ExpReplay
from custom_env import CustomEnv
from QNET import QNET
from epsilon_methods import exponential_decay_method
import csv
import datetime
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import datetime

import os
import csv


dict_actions = {0:'FORWARD', 1:'TURN_LEFT', 2:'TURN_RIGHT', 3: 'GRAB', 4:'SHOOT'}
base_path = 'C:\\Users\\osean\\Documents\\UFPA\\TCC\\CODIGO\\Wumpus_World-RL\\DQN'

class DeepQAgent:
    
    def __init__(self, env, individual, max_episodes=1000, hidden_units=256):
        # set hyper parameters
        self.max_episodes = max_episodes
        #self.max_actions = 1000
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.epsilon_min = 0.1  
        
        # set environment
        self.env = env
        self.states = env.observation_space.n
        self.actions = env.action_space.n
        self.max_actions = self.env.environment.board.max_steps
             

        # Experience Replay for batch learning
        self.exp = ExpReplay()
        # the number of experience per batch for batch learning
        self.batch_size = 64 

        # Deep Q Network
        self.qnet = QNET(self.states, self.actions, self.exp, individual)

        # For execute Deep Q Network
        session = tf.compat.v1.InteractiveSession()
        session.run(tf.compat.v1.global_variables_initializer())
        #self.qnet.set_session(session)
        self.qnet.session = session


    # def train(self):
    #     #RECORD INFO EPISODES
    #     list_info_train = []
    #     info_train = {'rewards': None,'episode': None, 'step_per_episode': None,
    #                 'has_gold': None, 'killed_wumpus': None, 'get_gold_and_return_home': None}
        
    #     # set hyper parameters
    #     max_episodes = self.max_episodes
    #     max_actions = self.max_actions
    #     exploration_rate = self.exploration_rate
    #     exploration_decay = self.exploration_decay
    #     batch_size = self.batch_size
        
    #     # start training
    
    #     for i in range(max_episodes):
    #         print(f'Episode {i}')
    #         amount_steps_environment = 0
    #         total_rewards = 0
    #         grab_gold = 0
    #         killed_wumpus = 0
    #         has_gold_safe_home = 0
    #         info_train = {'rewards': None,'episode': None, 'step_per_episode': None,
    #                 'has_gold': None, 'killed_wumpus': None, 'get_gold_and_return_home': None}

    #         state = self.env.reset()
    #         state = state.reshape((1, self.states))            

    #         for j in range(max_actions):
    #             #self.env.render() # Uncomment this line to render the environment
    #             action = self.qnet.get_action(state, exploration_rate)
    #             next_state, reward, done, info = self.env.step(dict_actions[action])
    #             next_state = next_state.reshape((1, self.states))

    #             total_rewards += reward
    #             amount_steps_environment += 1
                
    #             if done:
    #                 self.exp.add(state, action, reward, next_state, done)
    #                 self.qnet.batch_train(batch_size)

    #                 #record information of the episode
    #                 grab_gold = 1 if self.env.environment.board.components['Agent'].has_gold else 0
    #                 killed_wumpus = 1 if not self.env.environment.board.components['Agent'].wumpus_alive else 0
    #                 if next_state[0][0] == 0 and self.env.environment.board.components['Agent'].has_gold:
    #                     has_gold_safe_home = 1
    #                 info_train['episode'] = i
    #                 info_train['rewards'] = total_rewards
    #                 info_train['has_gold'] = grab_gold
    #                 info_train['step_per_episode'] = amount_steps_environment
    #                 info_train['killed_wumpus'] = killed_wumpus
    #                 info_train['get_gold_and_return_home'] = has_gold_safe_home
    #                 list_info_train.append(info_train)

    #                 break
                    
                    
    #             self.exp.add(state, action, reward, next_state, done)
    #             self.qnet.batch_train(batch_size)
                
    #             # next episode
    #             state = next_state
                

    #         exploration_rate = exponential_decay_method(i, max_episodes, self.epsilon_min)
    #         # print(f'Reward of episode {i}: {total_rewards}\nEpsilon: {exploration_rate}')

    #     self.write_executions(list_info_train)
    #     self.env.render()
            
    def train(self):
        # RECORD INFO EPISODES
        list_info_train = []
        info_train = {'rewards': None, 'episode': None, 'step_per_episode': None,
                    'has_gold': None, 'killed_wumpus': None, 'get_gold_and_return_home': None}
        
        # Set hyperparameters
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay
        batch_size = self.batch_size
        
        # Start training
        for i in range(max_episodes):
            # print(f'Episode {i}')
            amount_steps_environment = 0
            total_rewards = 0
            grab_gold = 0
            killed_wumpus = 0
            has_gold_safe_home = 0
            cells_explored = set()  # Conjunto para rastrear as células exploradas
            info_train = {'rewards': None, 'episode': None, 'step_per_episode': None,
                        'has_gold': None, 'killed_wumpus': None, 'get_gold_and_return_home': None}

            state = self.env.reset()
            state = state.reshape((1, self.states))            
            initial_position = tuple(state[0])  # Posição inicial do agente
            cells_explored.add(initial_position)

            for j in range(max_actions):
                # self.env.render() # Uncomment this line to render the environment
                action = self.qnet.get_action(state, exploration_rate)
                next_state, reward, done, info = self.env.step(dict_actions[action])
                next_state = next_state.reshape((1, self.states))

                total_rewards += reward
                amount_steps_environment += 1
                
                # Adicionar célula visitada ao conjunto de células exploradas
                current_position = tuple(next_state[0])
                cells_explored.add(current_position)
                
                if done:
                    self.exp.add(state, action, reward, next_state, done)
                    self.qnet.batch_train(batch_size)

                    # Record information of the episode
                    grab_gold = 1 if self.env.environment.board.components['Agent'].has_gold else 0
                    killed_wumpus = 1 if not self.env.environment.board.components['Agent'].wumpus_alive else 0
                    if next_state[0][0] == 0 and self.env.environment.board.components['Agent'].has_gold:
                        has_gold_safe_home = 1
                    info_train['episode'] = i
                    info_train['rewards'] = total_rewards
                    info_train['has_gold'] = grab_gold
                    info_train['step_per_episode'] = amount_steps_environment
                    info_train['killed_wumpus'] = killed_wumpus
                    info_train['get_gold_and_return_home'] = has_gold_safe_home
                    list_info_train.append(info_train)

                    break

                self.exp.add(state, action, reward, next_state, done)
                self.qnet.batch_train(batch_size)
                
                # Próximo estado
                state = next_state

            # Decaimento da taxa de exploração
            exploration_rate = exponential_decay_method(i, max_episodes, self.epsilon_min)

        # Final do treinamento
        self.write_executions(list_info_train)
        # self.env.render()

        # Retorna as informações relevantes para avaliação
        return total_rewards, amount_steps_environment, len(cells_explored), grab_gold, killed_wumpus, has_gold_safe_home

        
    
    def write_executions(self, infos_train: list):

        
        headers = ['episode', 'rewards', 'has_gold', 'step_per_episode', 'killed_wumpus', 'get_gold_and_return_home']

        agora = datetime.datetime.now()
        agora_formatado = agora.strftime("%d%m%Y_%H%M%S")

        # directory = path.join(path.abspath('.'), 'gym_game\\DQN\\executions')
        directory = path.join(base_path, '11_08_2024\executions')
        file_path = path.join(directory, "dqn_execution_"+agora_formatado+".csv")
        
        if not path.exists(directory):
            os.makedirs(directory)

        arquivo_existe = path.exists(file_path)
        if not arquivo_existe:
            with open(file_path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                writer.writeheader()
                writer.writerows(infos_train)
                
    def read_executions(self):
        date = '11_08_2024'
        file_name_date_exec = f'dqn_execution_{dim}x{dim}-{date}.csv'

        directory = path.join(base_path, '\\executions\\', file_name_date_exec)
        data = pd.read_csv(directory)

        return data

    def graph(self):
        data = self.read_executions()

        #moving average
        window = 100
        data['moving_average'] = data.rewards.rolling(window).mean()
    
        sns.lineplot(x = 'episode', y='rewards', data=data, label='Reward per episodes')
        sns.lineplot(x='episode', y='moving_average', data=data, label='Move Average Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.grid()
        plt.savefig(f'graph_rewards_dqn_{dim}x{dim}.png')
        # plt.show()
    
    def save_model(self):
        saver = tf.compat.v1.train.Saver()
        name_model = 'model_dqn'
        directory = path.join(path.abspath('.'), 'gym_game\\DQN\\models\\', name_model)
        saver.save(self.qnet.session, directory)
    
    def load_model(self):
        self.qnet.session = tf.compat.v1.Session()
        name_model = 'model_dqn.meta'
        directory = path.join(path.abspath('.'), 'gym_game\\DQN\\models\\', name_model)
        saver = tf.compat.v1.train.import_meta_graph(directory)
        saver.restore(self.qnet.session, tf.compat.v1.train.latest_checkpoint('.\\'))

    def close(self):
        self.qnet.close()

def reset_data():
    # directory = path.join(path.abspath('.'), 'gym_game\\DQN\executions\\', file_name)
    directory = path.join(base_path, 'executions\\', "dqn_execution_15x15-2024-04-13.csv")
    open(directory,"wb").close()



    

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    dict_max_steps = {4: 100, 8: 150, 10: 200, 15:250} #size environment is key and value is amount max steps
    dict_values_seed = {4: 123, 8: 99, 10: 917, 15:77} #size environment is key and value is values seed
    dim = 15
    date_execution = datetime.datetime.now().strftime('%Y-%m-%d')
    file_name = f'dqn_execution_{dim}x{dim}-{date_execution}.csv'
    reset_data()
    env = CustomEnv(nrow=dim,ncol=dim, max_steps=dict_max_steps[dim], value_seed=dict_values_seed[dim])
    agent = DeepQAgent(env)
    agent.train()
    # agent.graph()




