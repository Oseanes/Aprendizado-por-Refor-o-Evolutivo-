import tensorflow as tf
# from tensorflow.compat import v1 as tfv1
# from TNET import TNET
import numpy as np


class QNET():
    def __init__(self, in_units, out_units, exp, individual, hidden_units=256):
        # Target Network
        # self.tnet = TNET(in_units, out_units)

        # experience replay
        self.exp = exp
        
        # Q network architecture
        self.in_units = in_units
        self.out_units = out_units
        self.hidden_units = hidden_units

  
        # # Indices correspondentes aos tamanhos calculados
        # structured_solution = [
        #     individual[:1792],  # Pesos da primeira camada
        #     individual[1792:1792 + 256],  # Bias da primeira camada
        #     individual[1792 + 256:1792 + 256 + 65536],  # Pesos da segunda camada
        #     individual[1792 + 256 + 65536:1792 + 256 + 65536 + 256],  # Bias da segunda camada
        #     individual[1792 + 256 + 65536 + 256:1792 + 256 + 65536 + 256 + 65536],  # Pesos da terceira camada
        #     individual[1792 + 256 + 65536 + 256 + 65536:1792 + 256 + 65536 + 256 + 65536 + 256],  # Bias da terceira camada
        #     individual[1792 + 256 + 65536 + 256 + 65536 + 256:1792 + 256 + 65536 + 256 + 65536 + 256 + 65536],  # Pesos da quarta camada
        #     individual[1792 + 256 + 65536 + 256 + 65536 + 256 + 65536:1792 + 256 + 65536 + 256 + 65536 + 256 + 65536 + 256],  # Bias da quarta camada
        #     individual[1792 + 256 + 65536 + 256 + 65536 + 256 + 65536 + 256:],  # Pesos da camada de saída
        # ]
        in_units = 7         # Número de entradas
        out_units = 5        # Número de saídas

        # Calcular o número de parâmetros por camada
        NUM_W1 = in_units * hidden_units
        NUM_W2_W3_W4 = hidden_units * hidden_units
        NUM_b1_b2_b3_b4 = hidden_units    
        NUM_W5 = hidden_units * out_units
        structured_solution = [
        individual[:NUM_W1],  # Pesos da primeira camada
        individual[NUM_W1:NUM_W1 + NUM_b1_b2_b3_b4],  # Bias da primeira camada
        individual[NUM_W1 + NUM_b1_b2_b3_b4:NUM_W1 + NUM_b1_b2_b3_b4 + NUM_W2_W3_W4],  # Pesos da segunda camada
        individual[NUM_W1 + NUM_b1_b2_b3_b4 + NUM_W2_W3_W4:NUM_W1 + NUM_b1_b2_b3_b4 + NUM_W2_W3_W4 + NUM_b1_b2_b3_b4],  # Bias da segunda camada
        individual[NUM_W1 + NUM_b1_b2_b3_b4 + NUM_W2_W3_W4 + NUM_b1_b2_b3_b4:NUM_W1 + NUM_b1_b2_b3_b4 + 2 * NUM_W2_W3_W4 + NUM_b1_b2_b3_b4],  # Pesos da terceira camada
        individual[NUM_W1 + NUM_b1_b2_b3_b4 + 2 * NUM_W2_W3_W4 + NUM_b1_b2_b3_b4:NUM_W1 + NUM_b1_b2_b3_b4 + 2 * NUM_W2_W3_W4 + 2 * NUM_b1_b2_b3_b4],  # Bias da terceira camada
        individual[NUM_W1 + NUM_b1_b2_b3_b4 + 2 * NUM_W2_W3_W4 + 2 * NUM_b1_b2_b3_b4:NUM_W1 + NUM_b1_b2_b3_b4 + 3 * NUM_W2_W3_W4 + 2 * NUM_b1_b2_b3_b4],  # Pesos da quarta camada
        individual[NUM_W1 + NUM_b1_b2_b3_b4 + 3 * NUM_W2_W3_W4 + 2 * NUM_b1_b2_b3_b4:NUM_W1 + NUM_b1_b2_b3_b4 + 3 * NUM_W2_W3_W4 + 3 * NUM_b1_b2_b3_b4],  # Bias da quarta camada
        individual[NUM_W1 + NUM_b1_b2_b3_b4 + 3 * NUM_W2_W3_W4 + 3 * NUM_b1_b2_b3_b4:]  # Pesos da camada de saída
    ]
        self.individual = [np.array(ind) for ind in structured_solution]

        tf.compat.v1.disable_eager_execution()

        # Placeholder para a entrada
        self.x = tf.compat.v1.placeholder(tf.float32, shape=(None, self.in_units))       
        self.qnet = self._model()        

        # self.session = None
            # self._model()
        self._batch_learning_model()
        
        self.session = tf.compat.v1.Session()
        self.session.run(tf.compat.v1.global_variables_initializer())
        
    def _model(self):

        """ Q-network architecture """
        
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.variable_scope('qnet', reuse=tf.compat.v1.AUTO_REUSE):         
            # tf.compat.v1.disable_eager_execution()     
            self.x = tf.compat.v1.placeholder(tf.float32, shape=(None, self.in_units))
            # Inicializa os pesos e biases a partir da lista `individual`, sem especificar shape
            W1 = tf.compat.v1.get_variable('W1', initializer=tf.constant(self.individual[0].reshape(self.in_units, self.hidden_units), dtype=tf.float32))
            b1 = tf.compat.v1.get_variable('b1', initializer=tf.constant(self.individual[1], dtype=tf.float32))

            W2 = tf.compat.v1.get_variable('W2', initializer=tf.constant(self.individual[2].reshape(self.hidden_units, self.hidden_units), dtype=tf.float32))
            b2 = tf.compat.v1.get_variable('b2', initializer=tf.constant(self.individual[3], dtype=tf.float32))

            W3 = tf.compat.v1.get_variable('W3', initializer=tf.constant(self.individual[4].reshape(self.hidden_units, self.hidden_units), dtype=tf.float32))
            b3 = tf.compat.v1.get_variable('b3', initializer=tf.constant(self.individual[5], dtype=tf.float32))

            W4 = tf.compat.v1.get_variable('W4', initializer=tf.constant(self.individual[6].reshape(self.hidden_units, self.hidden_units), dtype=tf.float32))
            b4 = tf.compat.v1.get_variable('b4', initializer=tf.constant(self.individual[7], dtype=tf.float32))

            W5 = tf.compat.v1.get_variable('W5', initializer=tf.constant(self.individual[8].reshape(self.hidden_units, self.out_units), dtype=tf.float32))
                

            # Camadas ocultas da rede neural
            h1 = tf.nn.tanh(tf.matmul(self.x, W1) + b1)
            h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
            h3 = tf.nn.tanh(tf.matmul(h2, W3) + b3)
            h4 = tf.nn.tanh(tf.matmul(h3, W4) + b4)
            self.q = tf.matmul(h4, W5)

            return self

        
    def update(self):
        """Execution for Target network update"""
        self.session.run(self.update_opt)
    
    def get_action(self, state, e_rate):
        """ for training stage of the Agent, exploitation or exploration"""
        if np.random.random()<e_rate: # exploration
            return np.random.choice(self.out_units)
        else: # exploitation
            return np.argmax(self.session.run(self.q, feed_dict={self.x: state}))

    def _batch_learning_model(self):
        """For batch learning"""
        with tf.compat.v1.variable_scope('qnet'):
            # TD-target
            self.target = tf.compat.v1.placeholder(tf.float32, shape=(None, ))
            # Action index
            self.selected_idx = tf.compat.v1.placeholder(tf.int32, shape=(None, 2))
            # Q-value
            self.selected_q = tf.gather_nd(self.q, self.selected_idx)
            
            self.params = tf.compat.v1.get_collection( tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='qnet')
            
            # Q-network optimization alogrithms
            loss = tf.losses.mean_squared_error(self.target, self.selected_q)
            gradients = tf.gradients(loss, self.params)
            self.train_opt = tf.compat.v1.train.AdamOptimizer(3e-4).apply_gradients(zip(gradients, self.params))
    
    def batch_train(self, batch_size=64):
        """Implement Double DQN Algorithm, batch training"""
        if self.exp.get_num() < self.exp.get_min():
            #The number of experiences is not enough for batch training
            return

        # get a batch of experiences
        state, action, reward, next_state, done = self.exp.get_batch(batch_size)
        state = state.reshape(batch_size, self.in_units)
        next_state = next_state.reshape(batch_size, self.in_units)
        
        # get actions by Q-network
        qnet_q_values = self.session.run(self.q, feed_dict={self.x:next_state})
        qnet_actions = np.argmax(qnet_q_values, axis=1)
        #take pega os valores da lista qnet_q_values nas posicoes qnet_actions
        qnet_q = [np.take(qnet_q_values[i], qnet_actions[i]) for i in range(batch_size)]
        
        # Update Q-values of Q-network
        qnet_update_q = [r+0.95*q if not d else r for r, q, d in zip(reward, qnet_q, done)]
        # print("qnet_update_q:", qnet_update_q)

        
        # optimization
        indices=[[i,action[i]] for i in range(batch_size)]
        feed_dict={self.x:state, self.target:qnet_update_q, self.selected_idx:indices}
        self.session.run(self.train_opt, feed_dict)



    def close(self):
            """Close the TensorFlow session."""
            if self.session:
                self.session.close()
                self.session = None