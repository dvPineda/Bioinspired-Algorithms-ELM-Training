import numpy as np
from bio_classes import Swarm
from ELM import compute_swarm_fitness

def update_velocities_and_positions_pso(swarm: Swarm, w:float, c1:float, c2:float):
  # Random numbers
  r1 = np.random.uniform()
  r2 = np.random.uniform()

  for index, particle in enumerate(swarm.genes_list): # For each particle in the swarm
    # Weights update 
    particle.weights_velocity = ( 
                                  w * particle.weights_velocity +
                                  c1*r1*(particle.best_weights_position - particle.weights) +
                                  c2*r2*(swarm.best_gene.best_weights_position - particle.weights)
                                )

    particle.weights += particle.weights_velocity # Update to next position
    
    # Bias update
    particle.bias_velocity = ( 
                                w * particle.bias_velocity +
                                c1*r1*(particle.best_bias_position - particle.bias) +
                                c2*r2*(swarm.best_gene.best_bias_position - particle.bias)
                             )

    particle.bias += particle.bias_velocity # Update to next position
    
    # Chromosome update
    next_velocity= (
                      w * particle.chromosome_velocity +
                      c1*r1*(particle.best_chromosome_position - particle.chromosome) +
                      c2*r2*(swarm.best_gene.best_chromosome_position - particle.chromosome)
                   ) 

    # Prevent velocities from going out of bounds
    np.clip(next_velocity, -6, 6, out=particle.chromosome_velocity)
    velocityProbability = 2 / np.pi * np.arctan((np.pi*0.5)*particle.chromosome_velocity) # PSO-ELM paper (Mirjalili and Lewis, 2013)
    particle.chromosome = np.where(np.random.random() < velocityProbability, 1, 0)

    # Particle has been updated
    particle.needs_update = True
  # end particle loop
  return swarm

''' This method apply the Swarm-based algorithm of PSO to a given swarm'''
def pso(swarm: Swarm, max_generations:int, w_max:float,  w_min:float, c1:float, c2:float, OPTIMAL_D:int, OPTIMAL_C:int, trainX:np.ndarray, trainY:np.ndarray, testX:np.ndarray, testY: np.ndarray):
  t = 0
  while t < max_generations:
    # Linearly decrease the inertia weight
    w = w_max - (w_max - w_min) * t / max_generations

    # Evaluate Particles' Fitness and update Pb_i and Gb
    compute_swarm_fitness(swarm, OPTIMAL_D, OPTIMAL_C, trainX, trainY, testX, testY)

    # Update Velocity and Position of each particle
    update_velocities_and_positions_pso(swarm, w, c1, c2)

    # Increase step
    t += 1
  # end while
  return swarm