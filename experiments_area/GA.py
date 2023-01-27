import numpy as np
from bio_classes import Individual, Population
from ELM import compute_population_fitness

''' Takes a newly created individual and mutates it.
    1. The individual is mutated by mutating the chromosome, weights and bias.
    2. The individual fitness is not calculated yet and the needs_update flag is already set to true.'''
def internal_reproduction(gen: Individual):
  # Mutate chromosome
  mutate_chromosome_point = np.random.randint(0, gen.chromosome.shape[0]) # Choose a random feature to mutate
  gen.chromosome[mutate_chromosome_point] = 1 - gen.chromosome[mutate_chromosome_point] # Activate or deactivate the feature
  
  # Mutate weights
  total_rows = gen.weights.shape[0] 
  total_columns = gen.weights.shape[1] 
  mutate_column = np.random.randint(0, total_columns) # Choose a random column to mutate
  
  gen.weights[:,mutate_column] = np.random.uniform(-1,1,total_rows) # Mutate the column

  # Mutate bias
  mutate_bias_point = np.random.randint(0, gen.bias.shape[1]) # Choose a random column to mutate
  gen.bias[0,mutate_bias_point] = np.random.uniform(-1,1) # Mutate the column

def external_reproduction(father:Individual, mother:Individual):
  crossover_point = np.random.randint(0, father.chromosome.shape[0])
  crossover_column = np.random.randint(0, father.weights.shape[1])
  
  # Offsprings are created by combining father and mother chromosomes and weights
  offspring1 = Individual( chromosome = np.concatenate((father.chromosome[:crossover_point], mother.chromosome[crossover_point:])), 
                           weights = np.concatenate((father.weights[:,:crossover_column], mother.weights[:,crossover_column:]), axis=1),
                           bias = np.concatenate((father.bias[:,:crossover_column], mother.bias[:,crossover_column:]), axis=1) )
  
  offspring2 = Individual( chromosome = np.concatenate((mother.chromosome[:crossover_point], father.chromosome[crossover_point:])), 
                           weights = np.concatenate((mother.weights[:,:crossover_column], father.weights[:,crossover_column:]), axis=1),
                           bias = np.concatenate((mother.bias[:,:crossover_column], father.bias[:,crossover_column:]), axis=1) )
                       
  return offspring1, offspring2

def roulette_wheel_selection(population:Population):
    if population.size == 0:
        raise ValueError("Population is empty.")
    # Create an empty population for the chosen individuals of the Roulette Wheel
    roulettePopulation = Population.create_population(population.size, is_empty=True)
    roulettePopulation.insert_best_gene(population.best_gene) # Make sure we don't lose our best gene in the roulette 

    # Total population fitness (S)
    S = np.sum([individual.fitness for individual in population.genes_list])

    # Population chromosomes' relative probabilities
    rel_prob = [individual.fitness/S for individual in population.genes_list]

    # Create the list of accumulated relative probabilities
    acc_rel_prob = np.cumsum(rel_prob)

    for idx in range(1, roulettePopulation.size): # -1 because we already inserted the best gene
        r = np.random.uniform() 
        # Find the first index for which q_i < r
        for index, acc_prob in enumerate(acc_rel_prob): 
            if r < acc_prob:
                roulettePopulation.add_gene_to_list_at_index(population.genes_list[index], idx)
                break
    # end roulette population loop
    return roulettePopulation
  
''' This functions reproduces a population.
    It crossover the parents (external reproduction) and mutate the offspring (internal reproduction).
    Returns the input population, updated with the crossover and mutated childs '''
def reproduce_population(population: Population, CROSSOVER_PROBABILITY:float, MUTATION_PROBABILITY:float):
  childs = np.empty(shape=0, dtype=Individual) # list to store future offsprings
  gene_list_size = len(population.genes_list)

  for index,individual in enumerate(population.genes_list):
    # Generate a random mother if the crossover probability is met
    if(np.random.random() < CROSSOVER_PROBABILITY):
      random_mother_index = np.random.randint(0,gene_list_size) # Choose a random mother

      while(random_mother_index == index): # If the mother is the same as the father, choose another one
        random_mother_index = np.random.randint(0,gene_list_size)
      #end while
      
      # Crossover parents
      mother = population.genes_list[random_mother_index]
      offspring1, offspring2 = external_reproduction(individual, mother)

      # Mutate offsprings if probability is over the mutation probability
      if(np.random.random() < MUTATION_PROBABILITY): 
        internal_reproduction(offspring1) # Mutate offspring 1

      if(np.random.random() < MUTATION_PROBABILITY):
        internal_reproduction(offspring2) # Mutate offspring 2

      # Add offsprings to the childs list
      childs = np.append(childs, [offspring1, offspring2])
  
  # Extend the population with the new childs
  population.genes_list = np.concatenate( (population.genes_list, childs) )
  return population

  ''' This functions apply the Evolutionary-based Genetic Algorithm to a given population '''
def ga(population: Population, max_generations:int, crossover_prob:float, mutation_prob:float, OPTIMAL_D:int, OPTIMAL_C:int, trainX:np.ndarray, trainY:np.ndarray, testX:np.ndarray, testY: np.ndarray):
  t = 0
  while t < max_generations:
    # Reproduction
    population = reproduce_population(population, crossover_prob, mutation_prob)

    # Compute Fitness
    compute_population_fitness(population= population, D= OPTIMAL_D, 
                               C= OPTIMAL_C, trainingX= trainX,
                               trainingY= trainY, testX= testX,
                               testY= testY)

    # Roulette Selection
    population = roulette_wheel_selection(population)

    # Continue iterating
    t += 1
  # end while 
  return population