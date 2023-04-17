import random as rnd
from scipy.spatial import distance
import pandas as pd
from sklearn import preprocessing
import warnings
import sys
from tqdm import tqdm

#Used for ordering evaluations
class individual:
    def __init__(self, index, score, distance, num_changes, aval_norm, dist_norm, predict_proba):
        self.index = index #Indicates the instance's position in the dataframe
        self.score = score #Indicates the score in relation to the proximity of the class boundary
        self.distance = distance #Indicates the distance from the original instance
        self.num_changes = num_changes #Indicates the number of changes for class change
        self.aval_norm = aval_norm #Indicates the final fitness with standardized metrics
        self.dist_norm = dist_norm #Indicates the normalized distance (distance and number of changes)
        self.predict_proba = predict_proba #Indicates de individual's class
    def __repr__(self):
        return repr((self.index, self.score, self.distance, self.num_changes, self.aval_norm, self.dist_norm, self.predict_proba))

class counter_change:
    def __init__(self, column, value):
        self.column = column 
        self.value = value
    def __eq__(self, other):
        if self.column == other.column and self.value == other.value:
            return True
        else:
            return False
    def __repr__(self):
        return repr((self.column, self.value))    

#Used to generate a random value in the mutation operation
class feature_range:
    def __init__(self, column, col_type, min_value, max_value):
        self.column = column 
        self.col_type = col_type
        self.min_value = min_value
        self.max_value = max_value

    #Returns a random value to perform mutation operation
    def get_random_value(self):
        if self.col_type == 'int64' or self.col_type == 'int' or self.col_type == 'int16' or self.col_type == 'int8' or (self.col_type == 'uint8'):
            value = rnd.randint(self.min_value, self.max_value)
        else:  
            value = round(rnd.uniform(self.min_value, self.max_value), 2)
        return value
    
    #Checks if the attribute has only one value.
    def unique_value(self):
        if self.min_value != self.max_value:
            return False
        else:  
            return True    

    def __repr__(self):
        return repr((self.column, self.col_type, self.min_value, self.max_value)) 
        
class CSSE(object):
    
    def __init__(self, input_dataset, model, static_list = [], K = 3, num_gen = 30, pop_size = 100, per_elit = 0.1, cros_proba = 0.8, mutation_proba = 0.1, L1 = 1, L2 = 1):
        #User Options
        self.static_list = static_list #List of static features
        self.K = K #Number of counterfactuals desired
        #Model
        self.input_dataset = input_dataset
        self.model = model
        #GA Parameters
        self.num_gen = num_gen
        self.pop_size = pop_size
        self.per_elit = per_elit
        self.cros_proba = cros_proba
        self.mutation_proba = mutation_proba
        #Objective function parameters
        self.L1 = L1 #weight assigned the distance to the original instance
        self.L2 = L2 #weight assigned the number of changes needed in the original instance   
    
    #Get which index in the SHAP corresponding to the current class
    def getBadClass(self):   
        if self.current_class == self.model.classes_[0]:
            ind_cur_class = 0
        else:
            ind_cur_class = 1
        
        return ind_cur_class
    
    #Gets the valid values range for each feature
    def getFeaturesRange(self):
        features_range = []
       
        for i in range (0, self.input_dataset.columns.size):
            col_name = self.input_dataset.columns[i]
            col_type = self.input_dataset[col_name].dtype
            min_value = min(self.input_dataset[col_name])
            max_value = max(self.input_dataset[col_name])
            
            feature_range_ind = feature_range(col_name, col_type, min_value, max_value)
            features_range.append(feature_range_ind)
        
        return features_range
       
    def getMutationValue(self, currentValue, index, ind_feature_range):
        new_value = ind_feature_range.get_random_value()
        
        while currentValue == new_value:
            new_value = ind_feature_range.get_random_value()
        
        return new_value
    
    def equal(self, individual, population):
        aux = 0
        for i in range ( 1, len(population)):
            c = population.loc[i].copy()
            dst = distance.euclidean(individual, c)
            if dst == 0:
                aux = 1
        
        return aux

    def getPopInicial (self, df, features_range): 
        #The reference individual will always be in the 0 position of the df - so that it is normalized as well (it will be used later in the distance function)
        df.loc[0] = self.original_ind.copy()
        
        #Counting numbers of repeated individuals
        number_repetitions = 0
        
        #One more position is used because the zero position was reserved for the reference individual
        while len(df) < self.pop_size + 1:
            #Draw a feature to change
            index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            while df.columns[index_a] in self.static_list:
                index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
            if not features_range[index_a].unique_value():
                #Mutation
                mutant = self.original_ind.copy()

                new_value =  self.getMutationValue(mutant.iloc[index_a], index_a, features_range[index_a])
                mutant.iloc[index_a] = new_value

                ni = self.equal(mutant, df)
                if ni == 0:
                    df.loc[len(df)] = mutant.copy()
                else:
                    #Assesses whether the GA is producing too many repeated individuals.
                    number_repetitions = number_repetitions + 1
                    if number_repetitions == 2*self.pop_size:
                        self.pop_size = round(self.pop_size - self.pop_size*0.1)
                        self.mutation_proba = self.mutation_proba + 0.1
                        #print('Adjusting population size...', self.pop_size)
                        number_repetitions = 0
    
    #Complete the standardized proximity and similarity assessments for each individual
    def getNormalEvaluation(self, evaluation, aval_norma):
        scaler2 = preprocessing.MinMaxScaler()
        aval_norma2 = scaler2.fit_transform(aval_norma)
    
        i = 0
        while i < len(evaluation):
            evaluation[i].aval_norm = self.L1*aval_norma2[i,0] + self.L2*aval_norma2[i,1] + aval_norma2[i,2]
            evaluation[i].dist_norm = self.L1*aval_norma2[i,0] + self.L2*aval_norma2[i,1]
        
            i = i + 1
    
    def numChanges(self, ind_con):
        num = 0
        for i in range(len(self.original_ind)):
            if self.original_ind[i] != ind_con[i]:
                num = num + 1
        
        return num
        
    def fitness(self, population, evaluation, ind_cur_class):
        def getProximityEvaluation (proba):
            #Penalizes the individual who is in the negative class
            if proba < 0.5:
                predict_score = 0
            else:
                predict_score= proba
             
            return predict_score
               
        #Calculates similarity to the original instance
        def getEvaluationDist (ind, X_train_minmax):
            #Normalizes the data so that the different scales do not bias the distance
            a = X_train_minmax[0]
            b = X_train_minmax[ind]
            dst = distance.euclidean(a, b)
  
            return dst
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
            predict_proba = self.model.predict_proba(population)
                    
        #Calculating the distance between instances
        scaler = preprocessing.MinMaxScaler()
        X_train_minmax = scaler.fit_transform(population)
    
        i = 0
        aval_norma = [] 
        while i < len(population):
            proximityEvaluation = getProximityEvaluation(predict_proba[i, ind_cur_class])
            evaldist = getEvaluationDist(i, X_train_minmax)
            #The original individual is in the 1st position
            numChanges = self.numChanges(population.loc[i])
        
            ind = individual(i, proximityEvaluation, evaldist, numChanges, 0, 0, predict_proba[i, ind_cur_class])
            aval_norma.append([evaldist, numChanges, proximityEvaluation])
            evaluation.append(ind)
            i = i + 1

        self.getNormalEvaluation(evaluation, aval_norma)
       
    #Given a counterfactual solution returns the list of modified columns
    def getColumns(self, counter_solution):
        colums = []
        for j in range (0, len(counter_solution)):
            colums.append(counter_solution[j].column)
        
        return colums      
             
    #Checks if the new solution is contained in the solutions already found
    def contained_solution(self, original_instance, current_list, current_column_list, new_solution, new_column_solution):
        contained = False
        for i in range (0, len(current_list)):              
            if set(current_column_list[i]).issubset(new_column_solution):
                for j in range (0, len(current_list[i])):
                    pos = new_column_solution.index(current_list[i][j].column)
                    distancia_a = abs(original_instance[current_list[i][j].column] - current_list[i][j].value)
                    distancia_b = abs(original_instance[current_list[i][j].column] - new_solution[pos].value)
                    if distancia_b >= distancia_a:
                        contained = True

        return contained
      
    def elitism(self, evaluation, df, parents):
         
        num_elit = round(self.per_elit*self.pop_size)
        
        aval = []
        aval = evaluation.copy()
        aval.sort(key=lambda individual: individual.aval_norm)
        
        #contrafactual_ind = pd.DataFrame(columns=self.input_dataset.columns)
        solution_list = []
        solution_colums_list = []
        
        i = 0
        numContraf = 0
        while i < len(aval) and numContraf <= num_elit + 1:
            #Checks if the example belongs to the counterfactual class
            if aval[i].predict_proba < 0.5:
                ind_changes = []
                ind_colums_change = []
         
                #Gets counterfactual example change list
                ind_changes = self.getChanges(aval[i].index, parents)
                #Generates the list of columns modified in the counterfactual to check if there is already a solution with that set of columns
                ind_colums_change = self.getColumns(ind_changes)
                
                if ind_colums_change not in solution_colums_list:
                    #Check if one solution is a subset of the other
                    if not self.contained_solution(self.original_ind, solution_list, solution_colums_list, ind_changes, ind_colums_change):
                        #Include counterfactual in the list of examples of the final solution                    
                        df.loc[len(df)] = parents.iloc[aval[i].index].copy()                     
                                
                        #Add to the list of solutions (changes only)       
                        solution_list.append(ind_changes)
                        #Used to compare with the next counterfactuals (to ensure diversity)
                        solution_colums_list.append(ind_colums_change)
                                        
                        numContraf = numContraf + 1
                      
            i = i + 1
        return solution_list
    
    def roulette_wheel(self, evaluation):
        summation = 0
        #Performs roulette wheel to select parents who will undergo genetic operations
        for i in range (1, len(evaluation)): 
            summation = summation + 1/evaluation[i].aval_norm
    
        roulette = rnd.uniform( 0, summation )
    
        roulette_score = 1/evaluation[1].aval_norm
        i = 1
        while roulette_score < roulette:
            i += 1
            roulette_score += 1/evaluation[i].aval_norm
        
        return i
            
    def crossover (self, df, parents, evaluation, number_cross_repetitions):
        child = []
            
        corte = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
        index1 = self.roulette_wheel(evaluation)
        index2 = self.roulette_wheel(evaluation)
        
        ind_a = parents.iloc[index1].copy()
        ind_b = parents.iloc[index2].copy()
            
        crossover_op = rnd.random()
        if crossover_op <= self.cros_proba:
            child[ :corte ] = ind_a[ :corte ].copy()
            child[ corte: ] = ind_b[ corte: ].copy()
        else:
            child = ind_a.copy()
        
        ni = self.equal(child, df)
        if ni == 0:
            df.loc[len(df)] = child.copy()
        else:
            #Assesses whether the GA is producing too many repeated individuals.
            number_cross_repetitions = number_cross_repetitions + 1
            if number_cross_repetitions == self.pop_size:
                self.pop_size = round(self.pop_size - self.pop_size*0.1)
                self.mutation_proba = self.mutation_proba + 0.1
                #print('Adjusting population size...', self.pop_size)
                number_cross_repetitions = 0
        #    print('repeated')
        return number_cross_repetitions
                       
    def mutation (self, df, individual_pos, features_range):
        ni = 1
        #Does not allow repeated individual
        while ni == 1:
            #Draw a feature to change
            index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            while df.columns[index_a] in self.static_list:
                index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
            if not features_range[index_a].unique_value():
                #Mutation
                mutant = df.iloc[individual_pos].copy()
            
                #Draw the value to be changed
                new_value =  self.getMutationValue(mutant.iloc[index_a], index_a, features_range[index_a])  
                mutant.iloc[index_a] = new_value

                ni = self.equal(mutant, df)
                if ni == 0:
                    df.loc[individual_pos] = mutant.copy()
                #else:
                #    print('repeated')
     
    def getChanges(self, ind, dfComp):
        changes = []
        
        for i in range (len(dfComp.iloc[ind])):
            if self.original_ind[i] != dfComp.loc[ind][i]:
                counter_change_ind = counter_change(dfComp.columns[i], dfComp.loc[ind][i])
                changes.append(counter_change_ind)

        return changes
    
    #Generates the solution from the final population
    def getContrafactual(self, df, aval):
        
        contrafactual_ind = pd.DataFrame(columns=self.input_dataset.columns)
        solution_list = []
        solution_colums_list = []
        
        i = 0
        numContraf = 0
        while i < len(aval) and numContraf < self.K:
            #Checks if the example belongs to the counterfactual class
            if aval[i].predict_proba < 0.5:
                ind_changes = []
                ind_colums_change = []
         
                #Gets counterfactual example change list
                ind_changes = self.getChanges(aval[i].index, df)
                #Generates the list of columns modified in the counterfactual to check if there is already a solution with that set of columns
                ind_colums_change = self.getColumns(ind_changes)
                
                if ind_colums_change not in solution_colums_list:
                    #Check if one solution is a subset of the other
                    if not self.contained_solution(self.original_ind, solution_list, solution_colums_list, ind_changes, ind_colums_change):
                        #Include counterfactual in the list of examples of the final solution
                        contrafactual_ind.loc[len(contrafactual_ind)] = df.iloc[aval[i].index].copy()
                                
                        #Add to the list of solutions (changes only)       
                        solution_list.append(ind_changes)
                        #Used to compare with the next counterfactuals (to ensure diversity)
                        solution_colums_list.append(ind_colums_change)
                                        
                        numContraf = numContraf + 1
                        #print('solution_list ', solution_list)
                    #else:
                        #print('is contained ', ind_changes)
                #else:
                    #print('repeated ', ind_changes)
                      
            i = i + 1

        return contrafactual_ind, solution_list   
    
    def printResults(self, solution):
        print("Result obtained")
        if len(solution) != 0:
            for i in range(0, len(solution)): 
                print("\n")
                print(f"{'Counterfactual ' + str(i + 1):^34}")
                for j in range(0, len(solution[i])): 
                    print(f"{str(solution[i][j].column):<29} {str(solution[i][j].value):>5}")
        else:
            print('Solution not found. It may be necessary to adjust the parameters for this instance.')
                                                 
    def explain(self, original_ind, current_class):
        self.original_ind = original_ind #Original instance
        #self.ind_cur_class = ind_cur_class #Index in the shap corresponds to the original instance class
        self.current_class = current_class #Original instance class
        
        ind_cur_class = self.getBadClass()
    
        #Gets the valid values range of each feature
        features_range = []
        features_range = self.getFeaturesRange()

        #The DataFrame df will have the current population
        df = pd.DataFrame(columns=self.input_dataset.columns)
        
        #Generates the initial population with popinitial mutants        
        self.getPopInicial(df, features_range)
        
        for g in tqdm(range(self.num_gen), desc= "Processing..."):
            #To use on the parents of each generation
            parents = pd.DataFrame(columns=self.input_dataset.columns)
    
            #Copy parents to the next generation
            parents = df.copy()
            
            #df will contain the new population
            df = pd.DataFrame(columns=self.input_dataset.columns)
            
            evaluation = []                         
                   
            #Assessing generation counterfactuals
            self.fitness(parents, evaluation, ind_cur_class)
            
            #The original individual will always be in the 0 position of the df - So that it is normalized too (it will be used later in the distance function)
            df.loc[0] = self.original_ind.copy()
            
            #Copies to the next generation the per_elit best individuals
            self.elitism(evaluation, df, parents)
            
            number_cross_repetitions = 0
            while len(df) < self.pop_size + 1: #+1, as the 1st position is used to store the reference individual
                number_cross_repetitions = self.crossover(df, parents, evaluation, number_cross_repetitions)
                
                mutation_op = rnd.random()
                if mutation_op <= self.mutation_proba:
                    self.mutation(df, len(df) - 1, features_range)
                 
        evaluation = []
    
        #Evaluating the latest generation
        self.fitness(df, evaluation, ind_cur_class)
    
        #Order the last generation by distance to the original instance     
        evaluation.sort(key=lambda individual: individual.aval_norm)     
        
        #Getting the counterfactual set
        contrafactual_set = pd.DataFrame(columns=self.input_dataset.columns)
        contrafactual_set, solution_list = self.getContrafactual(df, evaluation)       
                 
        return contrafactual_set, solution_list