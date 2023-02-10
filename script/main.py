import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.matlib
import os
import time

from algorithms import run_ga, run_pso, run_cro

def process_data(df:pd.DataFrame):
    end = df.shape[1]

    # - - - DATAFRAME CONSTANTS - - - #
    X = df.iloc[:, 0:end-1].values
    Y = df.iloc[:, end-1].values

    J = len(np.unique(Y)) # Number of classes
    N,K = X.shape[0],X.shape[1]  # N = number of samples, K = number of features

    # - - - DATAFRAME PARTITIONS - - - #
    X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) # Scaling X (min-max normalization)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42) # First partition with 20% test data
    X_trainVal, X_testVal, Y_trainVal, Y_testVal = train_test_split(X_train, Y_train, test_size=0.2, random_state=42) # Second partition with 20% validation data
    # - - -    - - - #
    return X_trainVal, X_testVal, Y_trainVal, Y_testVal, X_train, Y_train, X_test, Y_test, J, N, K

def main(seed:int, POPULATION_SIZE:int, MAX_GENERATIONS:int, OPTIMAL_D:int, OPTIMAL_C:int, CROSSOVER_PROBABILITY:float, MUTATION_PROBABILITY:float, W_MAX:float, W_MIN:float, C1:float, C2:float, RHO_0:float, ETA:float, BROADCAST_FRACTION:float, ASEXUAL_FRACTION:float, PREDATION_FRACTION:float, ASEXUAL_PROBABILITY:float ,PREDATION_PROBABILITY:float):
    # Set the path to the data directory
    PORTATIL_DEV = False
    if PORTATIL_DEV:
        CSV_PATH = "C:/Users/david/OneDrive/Escritorio/MrRobot/IITV/4/TFG/data-Vito-PC/"
    else:
        CSV_PATH = "C:/Users/david/OneDrive/MrRobot/IITV/4/TFG/data-Vito-PC/"
        
    # List all the csv files in the data directory
    data_directory = os.listdir(CSV_PATH) 

    # Create a dataframe with the parameters we want to save as columns
    results_df = pd.DataFrame(columns=['Dataset',
                                       'Generations', 'Population_size', 
                                       'GA', 'PSO', 'CRO', 
                                       'GA_time', 'PSO_time', 'CRO_time', 
                                       'D', 'C',
                                       'Crossover_probability', 'Mutation_probability',
                                       'w_max', 'w_min', 'c1', 'c2',
                                       'rho0', 'eta', 'f_broadcast', 'f_asexual', 'f_predation', 'predation_probability'])

    # Iterate over all the csv files
    for index,csv in enumerate(data_directory):
        if (csv == "HandWriting.csv" or 
            csv == 'image-segmentation.csv' or 
            csv == 'ionosphere.csv' or 
            csv == "optical-recognition-handwritten-digits.csv" or
            csv == "seismic-bumps.csv"):
            continue
        
        # Set the global seed 
        print(f"Executing ELM with seed {seed} for dataset: {csv}, {index+1} out of {len(data_directory)}")
                    
        # Read CSV
        df = pd.read_csv(CSV_PATH + csv, sep=" ", header=None)
        end = df.shape[1]

        # - - - DATAFRAME CONSTANTS - - - #
        X = df.iloc[:, 0:end-1].values
        Y = df.iloc[:, end-1].values

        J = len(np.unique(Y)) # Number of classes
        N,K = X.shape[0],X.shape[1]  # N = number of samples, K = number of features

        # - - - DATAFRAME PARTITIONS - - - #
        X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) # Scaling X (min-max normalization)
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42) # First partition with 20% test data
        X_trainVal, X_testVal, Y_trainVal, Y_testVal = train_test_split(X_train, Y_train, test_size=0.2, random_state=42) # Second partition with 20% validation data
        # - - -    - - - #

        # Execute GA
        start_ga_time = time.time()

        GA_CCR = run_ga(size= POPULATION_SIZE,
                        K= K,
                        D= OPTIMAL_D,
                        C= OPTIMAL_C,
                        max_generations= MAX_GENERATIONS,
                        crossover_probability= CROSSOVER_PROBABILITY,
                        mutation_probability= MUTATION_PROBABILITY,
                        train_X_val= X_trainVal,
                        train_Y_val= Y_trainVal,
                        test_X_val= X_testVal,
                        test_Y_val= Y_testVal,
                        X_train= X_train,
                        Y_train= Y_train,
                        X_test= X_test,
                        Y_test= Y_test)

        ga_execution_time = time.time() - start_ga_time

        # Execute PSO
        start_pso_time = time.time()

        PSO_CCR = run_pso(size= POPULATION_SIZE,
                            K= K,
                            D= OPTIMAL_D,
                            C= OPTIMAL_C,
                            max_generations= MAX_GENERATIONS,
                            w_max= W_MAX,
                            w_min= W_MIN,
                            c1= C1,
                            c2= C2,
                            train_X_val= X_trainVal,
                            train_Y_val= Y_trainVal,
                            test_X_val= X_testVal,
                            test_Y_val= Y_testVal,
                            X_train= X_train,
                            Y_train= Y_train,
                            X_test= X_test,
                            Y_test= Y_test)

        pso_execution_time = time.time() - start_pso_time

        # Execute CRO
        start_cro_time = time.time()

        CRO_CCR = run_cro(size= POPULATION_SIZE,
                            K= K,
                            D= OPTIMAL_D,
                            C= OPTIMAL_C,
                            rho_0= RHO_0,
                            eta= ETA,
                            max_generations= MAX_GENERATIONS,
                            f_broadcast= BROADCAST_FRACTION,
                            f_asexual= ASEXUAL_FRACTION,
                            f_predation= PREDATION_FRACTION,
                            asexual_probability= ASEXUAL_PROBABILITY,
                            predation_probability= PREDATION_PROBABILITY,
                            train_X_val= X_trainVal,
                            train_Y_val= Y_trainVal,
                            test_X_val= X_testVal,
                            test_Y_val= Y_testVal,
                            X_train= X_train,
                            Y_train= Y_train,
                            X_test= X_test,
                            Y_test= Y_test)
        
        cro_execution_time = time.time() - start_cro_time
        csv_results = pd.DataFrame({'Dataset': csv,
                                    'Generations': MAX_GENERATIONS, 'Population_size': POPULATION_SIZE,
                                    'GA': GA_CCR, 'PSO': PSO_CCR, 'CRO': CRO_CCR,
                                    'GA_time': ga_execution_time, 'PSO_time': pso_execution_time, 'CRO_time': cro_execution_time,
                                    'D': OPTIMAL_D, 'C': OPTIMAL_C,
                                    'Crossover_probability': CROSSOVER_PROBABILITY, 'Mutation_probability': MUTATION_PROBABILITY,
                                    'w_max': W_MAX, 'w_min': W_MIN, 'c1': C1, 'c2': C2,
                                    'rho0': RHO_0, 'eta': ETA, 'f_broadcast': BROADCAST_FRACTION, 'f_asexual': ASEXUAL_FRACTION, 'f_predation': PREDATION_FRACTION, 'predation_probability': PREDATION_PROBABILITY
                                }, index=[0])

        # Concat results of the experiments to the results dataframe
        results_df = pd.concat([results_df, csv_results], ignore_index=True)       
    # end csv for

    # Create a directory named 'excel_files'
    folder_name = 'results'
    parent_dir = os.getcwd()

    directory = os.path.join(parent_dir, folder_name)

    if not os.path.exists(directory):
        os.mkdir(directory)
    
    # Save results to excel file
    output_file = os.path.join(directory, f"optimized_seed_{seed}.xlsx")

    if os.path.isfile(output_file): # File exists so we append the results            
        with pd.ExcelWriter(output_file, mode="a", engine="openpyxl", if_sheet_exists='replace') as writer:
            results_df.to_excel(writer, sheet_name=f"seed_{seed}", index=False) 
    else: # File doesn't exist so we create it
        with pd.ExcelWriter(output_file) as writer:
            results_df.to_excel(writer, sheet_name=f"seed_{seed}", index=False)

        
    print(f"Seed {seed} finished")