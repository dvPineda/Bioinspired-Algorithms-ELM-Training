import numpy as np
import numpy.matlib
from bio_classes import Individual, Particle, Larvae, Population, Swarm, Reef

''' Train ELM model and output the results '''
def train_model_and_output_results(best_weights:np.ndarray, best_chromosome:np.ndarray, best_bias:np.ndarray, D: int, C:float, trainingX:np.ndarray, trainingY:np.ndarray, testX:np.ndarray, testY:np.ndarray) -> float:
  # Weights with feature selection
  w = np.multiply(best_weights, best_chromosome[:, np.newaxis])

  # Amplify the matrix to the size of Xtraining and Xtestwith repmat
  BiasTrainingMatrix = np.matlib.repmat(best_bias, trainingX.shape[0], 1)
  BiasTestMatrix = np.matlib.repmat(best_bias, testX.shape[0], 1)

  # H (Sigmoide function) 
  H_training = 1 / (1 + np.exp(-(np.dot(trainingX, w) + BiasTrainingMatrix)))
  H_test = 1 / (1 + np.exp(-(np.dot(testX, w) + BiasTestMatrix)))

  # Beta with regularization, complete formula: inv( (eye(D) ./ C) + delta + aux) * H_Training' * Ytraining;
  aux = np.dot(H_training.T, H_training)
  delta = np.identity(D) / C
  Beta = np.dot(np.linalg.pinv(delta +  aux), np.dot(H_training.T, trainingY))

  # Prediction
  predicted_labels = np.dot(H_test, Beta)
  predicted_labels = np.rint(predicted_labels) # @ also computes the dot product
  
  correct_prediction = np.sum(predicted_labels == testY)

  CCR = correct_prediction / testY.shape[0]
  return CCR * 100

'''Computes the fitness for a given Individual using Extreme Learning Machine (ELM) model'''
def compute_individual_fitness(individual:Individual, D: int, C:float, trainingX:np.ndarray, trainingY:np.ndarray, testX:np.ndarray, testY:np.ndarray):
  # If the individual's fitness has already been computed, return it.
  if not individual.needs_update:
      return individual.fitness

  # Weights with feature selection 
  individualWeights = np.multiply(individual.weights, individual.chromosome[:, np.newaxis])

  # Amplify the bias matrix to the size of Xtraining and Xtestwith repmat
  BiasTrainingMatrix = np.matlib.repmat(individual.bias, trainingX.shape[0], 1)
  BiasTestMatrix = np.matlib.repmat(individual.bias, testX.shape[0], 1)

  # H (Sigmoide function) 
  H_training = 1 / (1 + np.exp(-(np.dot(trainingX, individualWeights) + BiasTrainingMatrix)))
  H_test = 1 / (1 + np.exp(-(np.dot(testX, individualWeights) + BiasTestMatrix)))

  # Beta with regularization, complete formula: inv( (eye(D) ./ C) + aux) * H_Training' * Ytraining;
  aux = np.dot(H_training.T, H_training)
  delta = np.identity(D) / C
  Beta = np.dot(np.linalg.pinv(delta +  aux), np.dot(H_training.T, trainingY))

  # Output
  Y_predicted = np.dot(H_test, Beta)
  fitness = np.linalg.norm(Y_predicted - testY)
  
  # Update the individual's fitness and return it.
  individual.fitness = fitness
  individual.needs_update = False
 
'''Computes and udpates the fitnesses of a given Population'''
def compute_population_fitness(population: Population, D:int, C:float, trainingX:np.ndarray, trainingY:np.ndarray, testX:np.ndarray, testY:np.ndarray):
  for individual in population.genes_list:
    compute_individual_fitness(individual, D, C, trainingX, trainingY, testX, testY) # Compute fitness

    # Update best gene in Population
    if population.best_gene is None:
      population.best_gene = individual
    else:
      if individual.fitness < population.best_gene.fitness:
        population.best_gene = individual
  # end for

'''Computes and udpates the fitnesses of a given Swarm'''
def compute_swarm_fitness(swarm: Swarm, D:int, C:float, trainingX:np.ndarray, trainingY:np.ndarray, testX:np.ndarray, testY:np.ndarray):
  for index, particle in enumerate(swarm.genes_list):  

    # Evaluate the particle's fitness    
    compute_individual_fitness(particle, D, C, trainingX, trainingY, testX, testY) # Compute fitness

    # Update personal bests and global best
    if particle.fitness < particle.best_fitness:
      # Update personal bests
      particle.best_fitness = particle.fitness      
      particle.best_weights_position = particle.weights
      particle.best_bias_position = particle.bias
      particle.best_chromosome_position = particle.chromosome
      
      # Update global best
      if swarm.best_gene is None:
        swarm.best_gene = particle
      else:
        if particle.fitness < swarm.best_gene.best_fitness:
          swarm.best_gene = particle
  # end for

  # Update Swarm's best
  if swarm.best_gene.best_fitness < swarm.global_best_fitness:
    swarm.global_best_fitness = swarm.best_gene.best_fitness
    swarm.global_best_weights = swarm.best_gene.best_weights_position
    swarm.global_best_bias = swarm.best_gene.best_bias_position
    swarm.global_best_chromosome = swarm.best_gene.best_chromosome_position

'''Computes and udpates the fitnesses of a given Reef'''
def compute_reef_fitness(reef: Reef, D:int, C:float, trainingX:np.ndarray, trainingY:np.ndarray, testX:np.ndarray, testY:np.ndarray):
  for coral in reef.corals_list:  
    if coral is None: # If the coral is empty, skip it
      continue

    compute_individual_fitness(coral, D, C, trainingX, trainingY, testX, testY) 
    
    # Update best gene in Reef
    if reef.best_coral is None:
      reef.best_coral = coral
    else:
      if coral.fitness < reef.best_coral.fitness:
        reef.best_coral = coral
