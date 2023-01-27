import numpy as np
from bio_classes import Larvae, Reef
from copy import deepcopy
from ELM import compute_individual_fitness, compute_reef_fitness

def broadcast(father:Larvae, mother:Larvae, attempts:int):
    crossover_point_chromosome = np.random.randint(0, father.chromosome.shape[0])
    crossover_column = np.random.randint(0, father.weights.shape[1])

    # Offsprings are created by combining father and mother chromosomes and weights
    new_larvae = Larvae( chromosome = np.concatenate((father.chromosome[:crossover_point_chromosome], mother.chromosome[crossover_point_chromosome:])), 
                         weights = np.concatenate((father.weights[:,:crossover_column], mother.weights[:,crossover_column:]), axis=1),
                         bias = np.concatenate((father.bias[:,:crossover_column], mother.bias[:,crossover_column:]), axis=1),
                         attempts= attempts)
                         
    # Larvae created by broadcast has needs_update flag set to True by default
    return new_larvae

def brooding(larvae: Larvae, attempts:int):

    # Auxiliar larvae to mutate instead of the original
    aux_larvae = deepcopy(larvae)

    # Mutate chromosome
    mutate_chromosome_point = np.random.randint(0, aux_larvae.chromosome.shape[0]) # Choose a random feature to mutate
    aux_larvae.chromosome[mutate_chromosome_point] = 1 - aux_larvae.chromosome[mutate_chromosome_point] # Activate or deactivate the feature

    # Mutate weights
    total_rows = aux_larvae.weights.shape[0] 
    total_columns = aux_larvae.weights.shape[1] 
    mutate_column = np.random.randint(0, total_columns) # Choose a random column to mutate

    larvae.weights[:,mutate_column] = np.random.uniform(-1,1,total_rows) # Mutate the column

    # Mutate bias
    mutate_bias_point = np.random.randint(0, aux_larvae.bias.shape[1]) # Choose a random column to mutate
    aux_larvae.bias[0,mutate_bias_point] = np.random.uniform(-1,1) # Mutate the column

    # Because the larvae was created by brooding, it needs to be updated and reset attempts
    aux_larvae.needs_update = True
    aux_larvae.attempts = attempts    
    return aux_larvae

def split_reef_candidates(candidates:list, fraction:float):
    number_of_candidates = len(candidates)
    split_point = int(number_of_candidates * fraction)

    # Split candidates in two lists
    broadcast_candidates = candidates[:split_point]
    brooding_candidates = candidates[split_point:]
    
    # If broadcast candidates are odd, add one more candidate to the list and remove it from the brooding candidates
    if len(broadcast_candidates) % 2 != 0:
        broadcast_candidates.append(candidates[split_point])
        brooding_candidates.remove(candidates[split_point])

    return [broadcast_candidates, brooding_candidates]

def larvae_setting(pool:np.ndarray, reef:Reef, OPTIMAL_D:int, OPTIMAL_C:int, trainX:np.ndarray, trainY:np.ndarray, testX:np.ndarray, testY: np.ndarray):
    alive_larvae = []
    # Try to settle the larvaes from the pool
    for i in range(len(pool)):
        # Compute fitness of the larvae
        compute_individual_fitness(pool[i], OPTIMAL_D, OPTIMAL_C, trainX, trainY, testX, testY)

        # Try to settle the larvae in a random hole
        random_hole = np.random.randint(0, reef.size)
        if reef.corals_list[random_hole] is None or pool[i].fitness < reef.corals_list[random_hole].fitness:
            reef.insert_new_larvae_in_hole(pool[i], random_hole)
            alive_larvae.append(pool[i])
        else:
            if pool[i].attempts > 0:
                pool[i].attempts -= 1
                alive_larvae.append(pool[i])
    # end for         
    pool = alive_larvae # Update pool

def predation(reef:Reef, predation_fraction:int, predation_probability:float):
  # Predate from the last corals with worse health
  for index in range(reef.size - predation_fraction, reef.size):
    if np.random.uniform(0, 1) < predation_probability:      
      reef.corals_list[reef.sorted_indexes[index]] = None

def reproduce_reef(sexual_corals:list, f_broadcast:float, new_larvaes_attempts:int):
  sexual_larvaes_pool = np.array([]) # pool of candidates from broadcast and brooding

  # Split reproduction sets
  broadcast_set, brooding_set = split_reef_candidates(sexual_corals, f_broadcast)

  # Broadcast    
  if len(broadcast_set) > 0: # If broadcast set is not empty, crossover
    while len(broadcast_set) > 0:
      # Select random father and a mother to crossover
      [father_candidate, mother_candidate] = np.random.choice(broadcast_set, 2, replace=False)
      broadcast_larvae = broadcast(father = father_candidate, 
                                   mother =  mother_candidate,
                                   attempts= new_larvaes_attempts)

      # Add offspring to the pool
      sexual_larvaes_pool = np.append(sexual_larvaes_pool,broadcast_larvae) # Two parents reproduce only one coral larva
      
      # Remove fathers of the reproduction list
      broadcast_set.remove(father_candidate)
      broadcast_set.remove(mother_candidate)
  ##

  #  Brooding
  if len(brooding_set) > 0: # If broadcast set is not empty, mutate every individual
    for c in brooding_set:
      brooding_larvae = brooding(c, new_larvaes_attempts)
      sexual_larvaes_pool = np.append(sexual_larvaes_pool, brooding_larvae) # Add the mutated coral to the pool
  ##
  return sexual_larvaes_pool

def cro(reef:Reef, max_generations: int, f_broadcast:float, f_asexual:float, f_predation:float, asexual_probability:float, predation_probability:float, OPTIMAL_D:int, OPTIMAL_C:int, trainX:np.ndarray, trainY:np.ndarray, testX:np.ndarray, testY: np.ndarray):

  pool = np.array([]) # Empty pool

  # Constants
  predation_fraction = int(reef.size * f_predation) # Number of corals to depredate at each generation
  budding_corals = int(len(reef.corals_list) * f_asexual) # Number of corals to reproduce by budding at each generation
  
  # Evaluate reef at the beginning
  compute_reef_fitness(reef, OPTIMAL_D, OPTIMAL_C, trainX, trainY, testX, testY)
  
  t = 0
  while t < max_generations:
    # Non empty corals in the reef
    non_empty_corals = [c for c in reef.corals_list if c is not None]
    
    # Sort the reef by corals' health function
    reef.sort_by_fitness()

    # Asexual reproduction if the probability is met and there are corals to reproduce
    if np.random.uniform(0, 1) < asexual_probability and len(non_empty_corals) > 0 and budding_corals > 0: 
      for index in reef.sorted_indexes[:budding_corals]: 
        if reef.corals_list[index] is not None:
          asexual_larvae = deepcopy(reef.corals_list[index]) # Budding
          asexual_larvae.attempts = reef.larvaes_attempts # Reset attempts
          pool = np.append(pool, asexual_larvae) # Add larvae to the pool
    ##
    
    # Broadcast and brooding
    sexual_larvaes = reproduce_reef(non_empty_corals, f_broadcast, new_larvaes_attempts=reef.larvaes_attempts)
    pool = np.append(pool, sexual_larvaes)
    
    # Larvae settings    
    larvae_setting(pool, reef, OPTIMAL_D, OPTIMAL_C, trainX, trainY, testX, testY)

    # Predation
    reef.sort_by_fitness() # Sort reef before predation so the worst corals are predated   
    predation(reef, predation_fraction, predation_probability)

    # Increase step
    t += 1
  #end while

  reef.sort_by_fitness() # Final sort
  reef.best_coral = reef.corals_list[reef.sorted_indexes[0]] # Best coral

  return reef
