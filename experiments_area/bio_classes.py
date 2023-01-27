import numpy as np
import math

# Individual class for the genetic algorithm.
class Individual:
  """
  A class that represents an individual solution.

  Attributes:
    chromosome (1D list): The chromosome of the individual in [0,1] interval.
    weights (2D list): The weights of the individual in U[-1,1] interval.
    bias (1D list): The bias of the individual in U[-1,1] interval.
    fitness (float): The fitness of the individual.
  """

  def __init__(self, chromosome: np.ndarray, weights: np.ndarray, bias: np.ndarray,  fitness:float = math.inf):
    self._chromosome = chromosome
    self._weights = weights
    self._bias = bias
    
    if fitness < 0.0:
      raise ValueError('The fitness of an individual cannot be negative.')
    else:
      self._fitness = fitness

    self._needs_update = True # Flag to indicate if the individual's fitness needs to be updated.
  
  def __str__(self):
    return f"""
    Individual:
        Feature-selection: {self._chromosome}
        Weights: {self._weights}
        Biases: {self._bias}
        Fitness: {self._fitness}
        Needs update: {self._needs_update}
    """
  @property
  def chromosome(self):
    return self._chromosome
  
  @property
  def weights(self):
    return self._weights
  
  @property
  def bias(self):
    return self._bias

  @property
  def fitness(self):
    return self._fitness
    
  @property
  def needs_update(self):
    return self._needs_update

  @chromosome.setter
  def chromosome(self, chromosome):
    self._chromosome = chromosome

  @weights.setter
  def weights(self, weights):
    self._weights = weights

  @bias.setter
  def bias(self, bias):
    self._bias = bias

  @fitness.setter
  def fitness(self,f):
    if f < 0.0:
      raise ValueError('The fitness of an individual cannot be negative.')
    else:
      self._fitness = f

  @needs_update.setter
  def needs_update(self, needs_update):
    self._needs_update = needs_update

  @staticmethod
  def create_individual(K:int = 0, D:int = 0):
    """
    Generates a random individual.

    Args:
      D (int): The number of hidden neurons.
      K (int): The number of input features.

    Returns:
      Individual: A random individual.
    """

    chromosome= np.random.randint(low= 0, high= 2, size = K) # 0 or 1 
    weights= np.random.uniform(low= -1, high= 1, size=(K,D)) # U[-1,1]
    bias= np.random.uniform(low= -1, high= 1, size=(1,D)) # U[-1,1]
    return Individual(chromosome, weights, bias)

# Particle class for the particle swarm optimization.
class Particle(Individual):
  """
  A class that represents an individual solution in Swarm-based problems.

  Attributes:
  -----------
    (same as Individual class), and additional attributes:

    best_fitness (float): The best fitness of the individual.
    best_weights_position (numpy.ndarray): The best weight position of the individual.
    best_chromosome_position (numpy.ndarray): The best chromosome position of the individual.
    weights_velocity (numpy.ndarray): The velocity of the weights of the individual.
    chromosome_velocity (numpy.ndarray): The velocity of the chromosome of the individual.  
  """
  
  def __init__(self, chromosome: np.ndarray, weights: np.ndarray, bias: np.ndarray, fitness:float = math.inf):
    super().__init__(chromosome, weights, bias, fitness)
    
    self._best_fitness = fitness # At the beginning, the best fitness is the input fitness
    self._best_weights_position = weights # By default, the best weight position is the starting weight position.
    self._best_bias_position = bias # By default, the best bias position is the starting bias position.
    self._best_chromosome_position = chromosome # By default, the best chromosome position is the starting chromosome position.
    self._weights_velocity = np.zeros(weights.shape) # All swarm's weights velocities are set to 0 at the start
    self._bias_velocity = np.zeros(bias.shape) # All swarm's bias velocities are set to 0 at the start
    self._chromosome_velocity = np.zeros(chromosome.shape) # All swarm's chromosome velocities are set to 0 at the start

  def __str__(self):
    return f"""
    Particle:
        Fitness: {self.fitness}
        Actual weights velocity: {self._weights_velocity}
        Actual bias velocity: {self._bias_velocity}
        Actual feature-selection velocity: {self._chromosome_velocity}
        Best fitness: {self._best_fitness}
    """

  @property
  def best_fitness(self):
    return self._best_fitness

  @property
  def best_weights_position(self):
    return self._best_weights_position
  
  @property
  def best_bias_position(self):
    return self._best_bias_position

  @property
  def best_chromosome_position(self):
    return self._best_chromosome_position
  
  @property
  def weights_velocity(self):
    return self._weights_velocity

  @property
  def bias_velocity(self):
    return self._bias_velocity

  @property
  def chromosome_velocity(self):
    return self._chromosome_velocity

  @best_fitness.setter
  def best_fitness(self,f):
    self._best_fitness = f

  @best_weights_position.setter
  def best_weights_position(self,w):
    self._best_weights_position = w

  @best_bias_position.setter
  def best_bias_position(self,b):
    self._best_bias_position = b

  @best_chromosome_position.setter
  def best_chromosome_position(self,c):
    self._best_chromosome_position = c

  @weights_velocity.setter
  def weights_velocity(self,w):
    self._weights_velocity = w

  @bias_velocity.setter
  def bias_velocity(self,b):
    self._bias_velocity = b
    
  @chromosome_velocity.setter
  def chromosome_velocity(self,c):
    self._chromosome_velocity = c

  @staticmethod
  def create_particle(K:int = 0, D:int = 0):
    """
    Generates a random particle.

    Args:
      D (int): The number of hidden neurons.
      K (int): The number of input features.

    Returns:
      Particle: A random particle.
    """
    chromosome= np.random.randint(low= 0, high= 2, size = K)
    weights= np.random.uniform(low= -1, high= 1, size=(K,D))
    bias= np.random.uniform(low= -1, high= 1, size=(1,D))
    return Particle(chromosome, weights, bias)

# Larvae class for the coral reef optimization.
class Larvae(Individual):
  """
  A class that represents an individual solution in CRO problems.

  Attributes:
  -----------
    (same as Individual class), and additional attributes:

    attempts  
  """
  
  def __init__(self, chromosome: np.ndarray, weights: np.ndarray, bias: np.ndarray, fitness:float = math.inf, attempts:int = 0):
    super().__init__(chromosome, weights, bias, fitness)
    self._attempt = attempts
  
  def __str__(self):
    return f"""
    Larvae:
        Fitness: {self.fitness}
        Attempts: {self._attempt}
        Feature-selection: {self.chromosome}
        Weights: {self.weights}
        Bias: {self.bias}
    """
  
  @property
  def attempts(self):
    return self._attempt
  
  @attempts.setter
  def attempts(self,a):
    self._attempt = a
  
  @staticmethod
  def create_larvae(K:int = 0, D:int = 0, attempts:int = 0):
    """
    Generates a random larvae.

    Args:
      D (int): The number of hidden neurons.
      K (int): The number of input features.
      Attempts (int): The number of attempts.

    Returns:
      Larvae: A random larvae.
    """
    chromosome= np.random.randint(low= 0, high= 1 + 1, size = K)
    weights= np.random.uniform(low= -1, high= 1, size=(K,D))
    bias= np.random.uniform(low= -1, high= 1, size=(1,D))
    return Larvae(chromosome, weights, bias, attempts = attempts)

# Population class for the GA.
class Population:
  """
  A class that represents a population of individual solutions.

  Attributes:
  -----------
    size (int): The size of the population.
    genes_list (list): The list of individuals solutions.
    best_gene (Individual): The best individual of the population.
  
  Methods:
  -----------
    insert_best_gene(Individual): Inserts the best individual of the population into the gene_list.
    add_gene_to_list_at_index(Individual): Adds an individual to the gene_list.
  """

  def __init__(self, size:int, K:int = 0, D:int = 0, is_empty:bool = False, is_swarm:bool = False):
    self._size = size
    self._best_gene = None

    if is_empty == True:
      self._genes_list = np.empty(size, dtype=Individual)
    else:
      if is_swarm == True:
        self._genes_list = np.array([Particle.create_particle(K,D) for _ in range(size)]) 
      else:
        self._genes_list = np.array([Individual.create_individual(K,D) for _ in range(size)]) 
  
  def __str__(self):
    return f"""
    Population:
        Size: {self._size}
        Gene list: {self._genes_list}
        Best gene: {self._best_gene}
    """
  @property
  def size(self):
    return self._size
  
  @property
  def genes_list(self):
    return self._genes_list

  @property
  def best_gene(self):
    return self._best_gene

  @genes_list.setter
  def genes_list(self, genes_list):
    self._genes_list = genes_list
    
  @best_gene.setter
  def best_gene(self,gen):  
    self._best_gene = gen
  
  @staticmethod
  def create_population(size:int, K:int = 0, D:int = 0, is_empty:bool = False, is_swarm:bool = False):
    """
    Generates a random population.

    Args:
      size (int): The size of the population.
      D (int): The number of hidden neurons.
      K (int): The number of input features.
      is_empty (bool): If True, the population will be empty.

    Returns:
      Population: A random population.
    """
    return Population(size, K, D, is_empty, is_swarm)

  def add_gene_to_list_at_index(cls,gen, index:int):
    np.put(cls._genes_list, index, gen)

  def insert_best_gene(cls,gen):
    cls.best_gene = gen
    cls._genes_list[0] = gen  # Insert the best gene at the beginning of the list

# Swarm class for the PSO.
class Swarm(Population):
  """
  A class that represents a population of solutions in Swarm-based problems.

  Attributes:
  -----------
    (same as Population), and additional attributes:

    global_best_fitness (float): The best fitness of the population.
    global_best_weights (numpy.ndarray): The best weight position of the population.
    global_best_chromosome (numpy.ndarray): The best chromosome position of the population.    
  
  Methods:
  -----------

  """
  
  def __init__(self, size:int, K:int = 0, D:int = 0, is_empty:bool = False):
    super().__init__(size, K, D, is_empty, is_swarm=True)
    self._global_best_fitness = math.inf
    self._global_best_weights = None
    self._global_best_chromosome = None
    self._global_best_bias = None

  def __str__(self):
    return f"""
    Swarm:
        Size: {self._size}
        Particles list: {self.genes_list}
        Best particle: {self.best_gene}
        Global best fitness: {self._global_best_fitness}
        Global best weights: {self._global_best_weights}
        Global best bias: {self._global_best_bias}
        Global best feature-selection: {self._global_best_chromosome}
    """
  @property
  def global_best_fitness(self):
    return self._global_best_fitness

  @property
  def global_best_weights(self):
    return self._global_best_weights

  @property
  def global_best_chromosome(self):
    return self._global_best_chromosome

  @property
  def global_best_bias(self):
    return self._global_best_bias

  @global_best_fitness.setter
  def global_best_fitness(self,f):
    self._global_best_fitness = f
  
  @global_best_weights.setter
  def global_best_weights(self,w):
    self._global_best_weights = w
  
  @global_best_chromosome.setter
  def global_best_chromosome(self,c):
    self._global_best_chromosome = c

  @global_best_bias.setter
  def global_best_bias(self,b):
    self._global_best_bias = b

  @staticmethod
  def create_swarm(size:int, K:int = 0, D:int = 0, is_empty:bool = False):
    """
    Generates a random swarm.

    Args:
      size (int): The size of the swarm.
      D (int): The number of hidden neurons.
      K (int): The number of input features.
      is_empty (bool): If True, the swarm will be empty.

    Returns:
      Swarm: A random swarm.
    """
    return Swarm(size, K, D, is_empty)

# Reef class for the CRO.
class Reef:
  """
  A class that represents a population of solutions in Evolutionary-based problems.

  Attributes:
  ----------- 
    size (int): The size of the population.
    free_occupied_rate (float): The rate of occupied individuals in the reef.
    corals_list (numpy.ndarray): The list of corals in the reef.
    best_coral (Individual): The best individual of the population.
    sorted_indexes (numpy.ndarray): The indexes of the sorted corals_list.
    
  Methods:
  -----------
  """

  def __init__(self, size:int, rate:float, K:int, D:int, larvaes_attempts:int):
    self._size = size
    self._free_occupied_rate = rate
    self._best_coral = None
    self._corals_list = np.full(shape = [size], fill_value = None) # Empty reef
    self._sorted_indexes = np.array([i for i in range(size)]) # Array with indexes of corals
    self._larvaes_attempts = larvaes_attempts

    occupiedHoles = int(size * rate) # Partially occupy the reef
    self._corals_list[:occupiedHoles] = np.array([Larvae.create_larvae(K,D, larvaes_attempts) for _ in range(occupiedHoles)]) # Fill the reef with individuals

  def __str__(self):
    return f"""
    Reef:
        Size: {self._size}
        Free-occupied rate: {self._free_occupied_rate}
        Corals list: {self._corals_list}
        Best coral: {self._best_coral}
        Larvaes attempts: {self._larvaes_attempts}
    """
  
  @property
  def size(self):
    return self._size

  @property
  def free_occupied_rate(self):
    return self._free_occupied_rate
  
  @property
  def corals_list(self):
    return self._corals_list

  @property
  def best_coral(self):
    return self._best_coral

  @property
  def sorted_indexes(self):
    return self._sorted_indexes

  @property
  def larvaes_attempts(self):
    return self._larvaes_attempts
  
  @corals_list.setter
  def corals_list(self,corals_list):
    self._corals_list = corals_list

  @best_coral.setter
  def best_coral(self,coral):
    self._best_coral = coral
  
  @staticmethod
  def create_reef(size:int, rate:float, K:int, D:int, larvaes_attempts:int):
    """
    Generates a reef.

    Args:
      size (int): The size of the reef.
      rate (float): The rate of occupied corals in the reef.
      D (int): The number of hidden neurons.
      K (int): The number of input features.
      Larvae_attempts (int): The number of attempts of a newborn larvae.

    Returns:
      Reef: A reef.
    """
    return Reef(size, rate, K, D, larvaes_attempts)

  def insert_new_larvae_in_hole(cls, larvae: Larvae, hole_index:int):
    cls.corals_list[hole_index] = larvae

  def remove_coral_from_hole(cls, hole_index:int):
    cls.corals_list[hole_index] = None
  
  def sort_by_fitness(cls):
    # Sort best_indexes by fitness from corals in corals_list, if corals_list[i] is None, then corals_list[i] is the worst
    cls._sorted_indexes = np.argsort([coral.fitness if coral is not None else math.inf for coral in cls._corals_list])