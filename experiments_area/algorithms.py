import pandas as pd
import numpy as np
from bio_classes import Population, Swarm, Reef
from ELM import train_model_and_output_results
from GA import ga
from PSO import pso
from CRO import cro

''' Executes Genetic Algorithm to find the best weights, bias and feature-selection for the ELM model'''
def run_ga(size:int, K:int, D:int, C:float, max_generations:int, crossover_probability:float, mutation_probability:float, train_X_val:np.ndarray, train_Y_val: np.ndarray, test_X_val: np.ndarray, test_Y_val: np.ndarray,  X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray) -> float: 

  # Create population
  population = Population.create_population(size, K, D)
  
  # Run GA
  evolved_population = ga(population= population, 
                          max_generations= max_generations,
                          crossover_prob= crossover_probability, 
                          mutation_prob= mutation_probability,
                          OPTIMAL_D= D,
                          OPTIMAL_C= C,
                          trainX= train_X_val, 
                          trainY= train_Y_val,
                          testX= test_X_val,
                          testY= test_Y_val)

  # Train model with best individual
  GA_CCR = train_model_and_output_results(best_weights= evolved_population.best_gene.weights, 
                                          best_chromosome = evolved_population.best_gene.chromosome, 
                                          best_bias= evolved_population.best_gene.bias,
                                          D = D,
                                          C = C, 
                                          trainingX = X_train,
                                          trainingY = Y_train,
                                          testX = X_test,
                                          testY = Y_test)
  return GA_CCR

''' Executes Particle Swarm Optimization to find the best weights, bias and feature-selection for the ELM model '''
def run_pso(size:int, K:int, D:int, C:float, max_generations:int, w_max:float, w_min:float, c1:float, c2:float, train_X_val:np.ndarray, train_Y_val: np.ndarray, test_X_val: np.ndarray, test_Y_val: np.ndarray,  X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray) -> float:
  # Create swarm
  swarm = Swarm.create_swarm(size, K, D)

  # Run PSO
  evolved_swarm = pso(swarm= swarm,
                      max_generations= max_generations,
                      w_max= w_max,
                      w_min= w_min,
                      c1= c1,
                      c2= c2,
                      OPTIMAL_D= D,
                      OPTIMAL_C= C,                      
                      trainX= train_X_val,
                      trainY= train_Y_val,
                      testX= test_X_val,
                      testY= test_Y_val)
                

  # Train model with best particle
  PSO_CCR = train_model_and_output_results(best_weights= evolved_swarm.global_best_weights,
                                          best_chromosome = evolved_swarm.global_best_chromosome,
                                          best_bias= evolved_swarm.global_best_bias,
                                          D = D,
                                          C = C,
                                          trainingX = X_train,
                                          trainingY = Y_train,
                                          testX = X_test,
                                          testY = Y_test)
                                          
  return PSO_CCR

''' Executes Coral Reef Optimization to find the best weights, bias and feature-selection for the ELM model '''
def run_cro(size:int, K:int, D:int, C:float, rho_0:float, eta:int, max_generations:int, f_broadcast:float, f_asexual:float, f_predation:float, asexual_probability:float, predation_probability:float, train_X_val:np.ndarray, train_Y_val: np.ndarray, test_X_val: np.ndarray, test_Y_val: np.ndarray,  X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray) -> float:
  # Create reef
  reef = Reef.create_reef(size, rho_0, K, D, eta)

  # Run CRO
  last_reef = cro(reef= reef,
                  max_generations= max_generations,
                  f_broadcast= f_broadcast,
                  f_asexual= f_asexual,
                  f_predation= f_predation,
                  asexual_probability= asexual_probability,
                  predation_probability= predation_probability,
                  OPTIMAL_D= D,
                  OPTIMAL_C= C,
                  trainX= train_X_val,
                  trainY= train_Y_val,
                  testX= test_X_val,
                  testY= test_Y_val)

  # Train model with best individual
  CRO_CCR = train_model_and_output_results(best_weights= last_reef.best_coral.weights, 
                                          best_chromosome = last_reef.best_coral.chromosome, 
                                          best_bias = last_reef.best_coral.bias,
                                          D = D,
                                          C = C, 
                                          trainingX = X_train,
                                          trainingY = Y_train,
                                          testX = X_test,
                                          testY = Y_test)
  return CRO_CCR