import tensorflow as tf
import datetime
from custom_env import CustomEnv
import QNET
from deepQAgent import DeepQAgent
import tensorflow as tf
from deap import base
from deap import creator
from deap import tools

import random
import numpy
import elitism


# Genetic Algorithm constants:
POPULATION_SIZE = 10
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # probability for mutating an individual
MAX_GENERATIONS = 30  
HALL_OF_FAME_SIZE = 5
CROWDING_FACTOR = 4.0  # crowding factor for crossover and mutation

# # set the random seed:
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)


dict_max_steps = {4: 100, 8: 150, 10: 200, 15:250} #size environment is key and value is amount max steps
dict_values_seed = {4: 123, 8: 99, 10: 917, 15:77} #size environment is key and value is values seed
dim = 15
env = CustomEnv(nrow=dim,ncol=dim, max_steps=dict_max_steps[dim], value_seed=dict_values_seed[dim])

env.observation_space.n
env.action_space.n
env.environment.board.max_steps


# Número de parâmetros em cada parte da arquitetura DQN
in_units = 7         # Número de entradas
hidden_units = 256   # Número de neurônios em cada camada oculta
out_units = 5        # Número de saídas

# Calcular o número de parâmetros por camada
NUM_W1 = in_units * hidden_units
NUM_W2_W3_W4= hidden_units * hidden_units
NUM_b1_b2_b3_b4 = hidden_units    
NUM_W5 = hidden_units * out_units

# Total de parâmetros
NUM_OF_PARAMS = (
    NUM_W1 +                          # Pesos da primeira camada
    4 * NUM_b1_b2_b3_b4 +             # Biases da primeira, segunda, terceira e quarta camadas
    3 * NUM_W2_W3_W4 +                # Pesos da segunda, terceira e quarta camadas
    NUM_W5                            # Pesos da camada de saída
)


print("Total de parâmetros:", NUM_OF_PARAMS)  # Deve resultar em 480


# weight and bias values are bound between -1 and 1:
BOUNDS_LOW, BOUNDS_HIGH = -1.0, 1.0  # boundaries for all dimensions

toolbox = base.Toolbox()

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# helper function for creating random real numbers uniformly distributed within a given range [low, up]
# it assumes that the range is the same for every dimension
def randomFloat(low, up):

    solution = []
    solution.extend([random.uniform(l, u) for l, u in zip([low] * NUM_W1, [up] * NUM_W1)])  # Pesos da primeira camada
    solution.extend([random.uniform(l, u) for l, u in zip([low] * NUM_b1_b2_b3_b4, [up] * NUM_b1_b2_b3_b4)])  # Bias da primeira camada
    solution.extend([random.uniform(l, u) for l, u in zip([low] * NUM_W2_W3_W4, [up] * NUM_W2_W3_W4)])  # Pesos da segunda camada
    solution.extend([random.uniform(l, u) for l, u in zip([low] * NUM_b1_b2_b3_b4, [up] * NUM_b1_b2_b3_b4)])  # Bias da segunda camada
    solution.extend([random.uniform(l, u) for l, u in zip([low] * NUM_W2_W3_W4, [up] * NUM_W2_W3_W4)])  # Pesos da terceira camada
    solution.extend([random.uniform(l, u) for l, u in zip([low] * NUM_b1_b2_b3_b4, [up] * NUM_b1_b2_b3_b4)])  # Bias da terceira camada
    solution.extend([random.uniform(l, u) for l, u in zip([low] * NUM_W2_W3_W4, [up] * NUM_W2_W3_W4)])  # Pesos da quarta camada
    solution.extend([random.uniform(l, u) for l, u in zip([low] * NUM_b1_b2_b3_b4, [up] * NUM_b1_b2_b3_b4)])  # Bias da quarta camada
    solution.extend([random.uniform(l, u) for l, u in zip([low] * NUM_W5, [up] * NUM_W5)])  # Pesos da camada de saída

    return solution
 

# create an operator that randomly returns a float in the desired range:
toolbox.register("attrFloat", randomFloat, BOUNDS_LOW, BOUNDS_HIGH)

# create an operator that fills up an Individual instance:
toolbox.register("individualCreator",
                 tools.initIterate,
                 creator.Individual,
                 toolbox.attrFloat)

# create an operator that generates a list of individuals:
toolbox.register("populationCreator",
                 tools.initRepeat,
                 list,
                 toolbox.individualCreator)




def score(individual):
    gold_bonus = 80  # Recompensa por pegar o ouro
    wumpus_bonus = 50  # Recompensa por matar o Wumpus
    home_bonus = 100  # Recompensa por voltar à casa com o ouro
    tf.compat.v1.disable_eager_execution()
    agent = DeepQAgent(env, individual)
    
    rewards, steps, cells_explored, has_gold, killed_wumpus, returned_home = agent.train()  # Supondo que agora `train()` retorne mais informações
    
    exploration_bonus = cells_explored * 0.5 # O valor de alpha deve encorajar a exploração do ambiente 0.1 a 0.5 pode ser um bom ponto de partida
    penalty_for_inactivity = steps *  0.3 # penalizar o agente por tomar muitos passos sem fazer progresso significativo (como pegar ouro, matar o Wumpus ou retornar à base)
    mission_rewards = (has_gold * gold_bonus) + (killed_wumpus * wumpus_bonus) + (returned_home * home_bonus)

    final_score = rewards + exploration_bonus - penalty_for_inactivity + mission_rewards
    
    agent.close()
    print("fitness:", final_score)
    return final_score,


toolbox.register("evaluate", score)


# genetic operators:
# toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("select", tools.selRoulette)

toolbox.register("mate",
                 tools.cxSimulatedBinaryBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR)

toolbox.register("mutate",
                 tools.mutPolynomialBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR,
                 indpb=1.0/NUM_OF_PARAMS)


def test_best_solution(best, env):
    # Gerar o agente com a melhor solução
    agent = DeepQAgent(env, best, max_episodes=1000)
    
    score = agent.train()
    print("scores = ", score)



# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    # print best solution found:
    best = hof.items[0]
    print()
    print("Best Solution = ", best)
    print("Best Score = ", best.fitness.values[0])
    print()


    test_best_solution(best, env) 


if __name__ == "__main__":
    main()





# import tensorflow as tf
# import datetime
# from custom_env import CustomEnv
# import QNET
# from deepQAgent import DeepQAgent
# from deap import base, creator, tools
# import random
# import numpy
# import elitism


# # Constantes do Algoritmo Genético
# POPULATION_SIZE = 8          # Aumentado para mais diversidade
# P_CROSSOVER_INITIAL = 0.9    # Probabilidade inicial de crossover
# P_MUTATION_INITIAL = 0.5     # Probabilidade inicial de mutação
# MAX_GENERATIONS = 20
# HALL_OF_FAME_SIZE = 5
# CROWDING_FACTOR = 4.0
# STOP_THRESHOLD = 0.001       # Critério de parada antecipada
# NO_IMPROVEMENT_LIMIT = 5     # Gerações sem melhoria significativa

# # Parâmetros do ambiente
# dict_max_steps = {4: 100, 8: 150, 10: 200, 15: 250}
# dict_values_seed = {4: 123, 8: 99, 10: 917, 15: 77}
# dim = 15
# env = CustomEnv(nrow=dim, ncol=dim, max_steps=dict_max_steps[dim], value_seed=dict_values_seed[dim])

# # Número de parâmetros em cada parte da arquitetura DQN
# in_units = 7
# hidden_units = 256
# out_units = 5

# # Cálculo do total de parâmetros
# NUM_W1 = in_units * hidden_units
# NUM_W2_W3_W4 = hidden_units * hidden_units
# NUM_b1_b2_b3_b4 = hidden_units
# NUM_W5 = hidden_units * out_units
# NUM_OF_PARAMS = NUM_W1 + 4 * NUM_b1_b2_b3_b4 + 3 * NUM_W2_W3_W4 + NUM_W5

# # Limites para pesos e biases
# BOUNDS_LOW, BOUNDS_HIGH = -1.0, 1.0

# # Configuração do DEAP
# toolbox = base.Toolbox()
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)

# # Função para criação de pesos e biases aleatórios dentro do intervalo
# def randomFloat(low, up):
#     solution = []
#     solution.extend([random.uniform(low, up) for _ in range(NUM_OF_PARAMS)])
#     return solution

# toolbox.register("attrFloat", randomFloat, BOUNDS_LOW, BOUNDS_HIGH)
# toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)
# toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# # Função de avaliação
# def score(individual):
#     gold_bonus = 80
#     wumpus_bonus = 50
#     home_bonus = 100
#     tf.compat.v1.disable_eager_execution()
#     agent = DeepQAgent(env, individual)
    
#     rewards, steps, cells_explored, has_gold, killed_wumpus, returned_home = agent.train()
    
#     exploration_bonus = cells_explored * 0.5
#     penalty_for_inactivity = steps * 0.3
#     mission_rewards = (has_gold * gold_bonus) + (killed_wumpus * wumpus_bonus) + (returned_home * home_bonus)

#     final_score = rewards + exploration_bonus - penalty_for_inactivity + mission_rewards
    
#     agent.close()
#     return final_score,

# toolbox.register("evaluate", score)

# # Operadores Genéticos
# # toolbox.register("select", tools.selTournament, tournsize=2)
# toolbox.register("select", tools.selRoulette)
# toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR)
# toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR, indpb=1.0/NUM_OF_PARAMS)

# # Função para teste da melhor solução
# def test_best_solution(best, env):
#     agent = DeepQAgent(env, best, max_episodes=150)
#     score = agent.train()
#     print("Score = ", score)

# # Fluxo do Algoritmo Genético com adaptação e parada antecipada
# def main():
#     population = toolbox.populationCreator(n=POPULATION_SIZE)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("max", numpy.max)
#     stats.register("avg", numpy.mean)
#     hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
#     no_improvement_count = 0

#     # Taxas de crossover e mutação variáveis
#     crossover_rate = P_CROSSOVER_INITIAL
#     mutation_rate = P_MUTATION_INITIAL

#     previous_max_fitness = None

#     for generation in range(MAX_GENERATIONS):
#         population, logbook = elitism.eaSimpleWithElitism(
#             population,
#             toolbox,
#             cxpb=crossover_rate,
#             mutpb=mutation_rate,
#             ngen=1,
#             stats=stats,
#             halloffame=hof,
#             verbose=True
#         )

#         current_max_fitness = max(ind.fitness.values[0] for ind in population)
        
#         # Critério de parada antecipada se o fitness não melhorar significativamente
#         if previous_max_fitness is not None:
#             improvement = abs(current_max_fitness - previous_max_fitness)
#             if improvement < STOP_THRESHOLD:
#                 no_improvement_count += 1
#                 if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
#                     print(f"Parada antecipada na geração {generation} devido a falta de melhoria.")
#                     break
#             else:
#                 no_improvement_count = 0

#         previous_max_fitness = current_max_fitness

#         # Adaptação da taxa de mutação e crossover
#         crossover_rate = max(0.6, crossover_rate * 0.95)  # Redução da taxa de crossover
#         mutation_rate = max(0.2, mutation_rate * 0.95)    # Redução da taxa de mutação

#     best = hof.items[0]
#     print("\nBest Solution = ", best)
#     print("Best Score = ", best.fitness.values[0])

#     test_best_solution(best, env)

# if __name__ == "__main__":
#     main()
