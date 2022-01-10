import shap
import random as rnd
from scipy.spatial import distance
import pandas as pd
from sklearn import preprocessing
import warnings
import sys

#Used for ordering evaluations
class individual:
    def __init__(self, index, score, distance, num_changes, aval_norm, dist_norm, class_ind):
        self.index = index #Indicates the instance's position in the dataframe
        self.score = score #Indicates the score in relation to the proximity of the class boundary
        self.distance = distance #Indicates the distance from the original instance
        self.num_changes = num_changes #Indicates the number of changes for class change
        self.aval_norm = aval_norm #Indicates the final fitness with standardized metrics
        self.dist_norm = dist_norm #Indicates the normalized distance (distance and number of changes)
        self.class_ind = class_ind #Indicates de individual's class
    def __repr__(self):
        return repr((self.index, self.score, self.distance, self.num_changes, self.aval_norm, self.dist_norm, self.class_ind))

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
        
class CSSE(object):
    
    def __init__(self, original_ind, current_class, static_list, K, input_dataset, x_train, model, num_gen, pop_size, per_elit, cros_proba, mutation_proba, L1, L2, L3, TreeClassifier):
        #User Options
        self.original_ind = original_ind #Original instance
        #self.ind_cur_class = ind_cur_class #Index in the shap corresponds to the original instance class
        self.current_class = current_class #Original instance class
        self.static_list = static_list #List of static features
        self.K = K #Number of counterfactuals desired
        #Model
        self.input_dataset = input_dataset
        self.x_train = x_train
        self.model = model
        #GA Parameters
        self.num_gen = num_gen
        self.pop_size = pop_size
        self.per_elit = per_elit
        self.cros_proba = cros_proba
        self.mutation_proba = mutation_proba
        #Objective function parameters
        self.L1 = L1 #weight assigned the distance to the original instance
        self.L2 = L2 #weight assigned the amount of changes needed in the original instance
        self.L3 = L3 #weight assigned to distance for counterfactual class
        #Algorithm
        #The current version of the method uses the Shap TreeExplainer for tree models and the KernelExplainer for all other algorithms
        self.TreeClassifier = TreeClassifier #Informs whether the model is tree-based (Use 'True' for tree-based models and 'False' otherwise)
           
    #Get which index in the SHAP corresponding to the current class
    def getBadClass(self):   
        if self.current_class == self.model.classes_[0]:
            ind_cur_class = 0
        else:
            ind_cur_class = 1
        
        return ind_cur_class
    
    #Generates the list of values (domain) for each feature (column)
    def getFeaturesDomain(self):
        features_domain = []
        for i in range (0, self.input_dataset.columns.size):
            features_domain.append(list(set(self.input_dataset.values[:, i])))
        
        return features_domain
    
    def getMutationValue(self, currentValue, index, features_domain):
        new_list = features_domain[index].copy()
        #Tests whether there is more than one possible value for the feature
        if len(new_list) > 1:
            new_list.remove(currentValue)
            new_value = rnd.choices(new_list)
        else:
            new_value = currentValue
        return new_value
    
    def equal(self, individual, population):
        aux = 0
        for i in range ( 1, len(population)):
            c = population.loc[i].copy()
            dst = distance.euclidean(individual, c)
            if dst == 0:
                aux = 1
        
        return aux

    def getPopInicial (self, df, features_domain): 
        #The reference individual will always be in the 0 position of the df - so that it is normalized as well (it will be used later in the distance function)
        df.loc[0] = self.original_ind.copy()

        #One more position is used because the zero position was reserved for the reference individual
        while len(df) < self.pop_size + 1:
            #Draw a feature to change
            index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            while df.columns[index_a] in self.static_list:
                index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
            if len(features_domain[index_a]) > 1:
                #Mutation
                mutant = self.original_ind.copy()

                new_value =  self.getMutationValue(mutant.iloc[index_a], index_a, features_domain)
            
                mutant.iloc[index_a] = new_value[0]

                ni = self.equal(mutant, df)
                if ni == 0:
                    df.loc[len(df)] = mutant.copy()
    
    #Complete the standardized proximity and similarity assessments for each individual
    def getAvaliacaoNormal(self, evaluation, aval_norma):
        scaler2 = preprocessing.MinMaxScaler()
        aval_norma2 = scaler2.fit_transform(aval_norma)
    
        i = 0
        while i < len(evaluation):
            evaluation[i].aval_norm = self.L1*aval_norma2[i,0] + self.L2*aval_norma2[i,1] + self.L3*aval_norma2[i,2]
            evaluation[i].dist_norm = self.L1*aval_norma2[i,0] + self.L2*aval_norma2[i,1]
        
            i = i + 1
    
    def numChanges(self, ind_con):
        num = 0
        for i in range(len(self.original_ind)):
            if self.original_ind[i] != ind_con[i]:
                num = num + 1
        
        return num
        
    def fitness(self, population, evaluation, explainerAG, ind_cur_class):
        #Calculates proximity to positive class    
        def getAvaliacao (ind, class_ind, ind_cur_class, current_class, base_value, shap_valuesAG):
            i = 0
            predict_sum = 0
            while i < len(shap_valuesAG[ind_cur_class][ind]):
                predict_sum = predict_sum + shap_valuesAG[ind_cur_class][ind,i]
                i = i + 1
            
            #Penalizes the individual who is in the negative class
            if str(class_ind) == str(current_class):
                predict_sum = predict_sum + 3
            
            return predict_sum + base_value
        
        #Calculates similarity to the original instance
        def getAvaliacaoDist (ind, X_train_minmax):
            #Normalizes the data so that the different scales do not bias the distance
            a = X_train_minmax[0]
            b = X_train_minmax[ind]
            dst = distance.euclidean(a, b)
    
            #Penalize individuals identical to the original
            if dst == 0:
                dst = 3.0
  
            return dst
        
        base_value = explainerAG.expected_value[ind_cur_class]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_valuesAG = explainerAG.shap_values(population)
        predict = self.model.predict(population)
        
        #Calculating the distance between instances
        scaler = preprocessing.MinMaxScaler()
        X_train_minmax = scaler.fit_transform(population)
    
        i = 0
        aval_norma = [] 
        while i < len(population):
            avaliacao = getAvaliacao(i, predict[i], ind_cur_class, self.current_class, base_value, shap_valuesAG)
            avaldist = getAvaliacaoDist(i, X_train_minmax)
            #The original individual is in the 1st position
            qtd = self.numChanges(population.loc[i])
        
            ind = individual(i, avaliacao, avaldist, qtd, 0, 0, predict[i])
            aval_norma.append([avaldist, qtd, avaliacao])
            evaluation.append(ind)
            i = i + 1

        self.getAvaliacaoNormal(evaluation, aval_norma)
    
    def elitism(self, evaluation, df, parents):
        num_elit = round(self.per_elit*self.pop_size)
        
        aval_tmp = []
        aval_tmp = evaluation.copy()
        aval_tmp.sort(key=lambda individual: individual.aval_norm)     
        for m in range (0, num_elit):
            df.loc[len(df)] = parents.iloc[aval_tmp[0].index].copy()
            del aval_tmp[0]
    
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
            
    def crossover (self, df, parents, evaluation):
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
        #else:
        #    print('repeated')
     
                       
    def mutation (self, df, individual_pos, features_domain):
        ni = 1
        #Does not allow repeated individual
        while ni == 1:
            #Draw a feature to change
            index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            while df.columns[index_a] in self.static_list:
                index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
            if len(features_domain[index_a]) > 1:
                #Mutation
                mutant = df.iloc[individual_pos].copy()
            
                #Draw the value to be changed
                new_value =  self.getMutationValue(mutant.iloc[index_a], index_a, features_domain)  
                mutant.iloc[index_a] = new_value[0]

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
    
    def getContrafactual(self, df, aval):
        
        #Given a counterfactual solution returns the list of modified columns
        def getColumns(counter_solution):
            colums = []
            for j in range (0, len(counter_solution)):
                colums.append(counter_solution[j].column)
        
            return colums      
             
        #Checks if the new solution is contained in the solutions already found
        def contained_solution(original_instance, current_list, current_column_list, new_solution, new_column_solution):
            contained = False
            #print('new_solution', new_solution)
            for i in range (0, len(current_list)):              
                if set(current_column_list[i]).issubset(new_column_solution):
                    for j in range (0, len(current_list[i])):
                        pos = new_column_solution.index(current_list[i][j].column)
                        distancia_a = abs(original_instance[current_list[i][j].column] - current_list[i][j].value)
                        distancia_b = abs(original_instance[current_list[i][j].column] - new_solution[pos].value)
                        if distancia_b >= distancia_a:
                            contained = True

            return contained

        contrafactual_ind = pd.DataFrame(columns=self.input_dataset.columns)
        solution_list = []
        solution_colums_list = []
        
        i = 0
        numContraf = 0
        while i < len(aval) and numContraf < self.K:
            #Checks if the example belongs to the counterfactual class
            if str(aval[i].class_ind) != str(self.current_class):
                ind_changes = []
                ind_colums_change = []
         
                #Gets counterfactual example change list
                ind_changes = self.getChanges(aval[i].index, df)
                #Generates the list of columns modified in the counterfactual to check if there is already a solution with that set of columns
                ind_colums_change = getColumns(ind_changes)
                
                if ind_colums_change not in solution_colums_list:
                    #Check if one solution is a subset of the other
                    if not contained_solution(self.original_ind, solution_list, solution_colums_list, ind_changes, ind_colums_change):
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
            
    
    #def ShowShapPlot(self, population, explainerAG)
    #    shap.force_plot(explainer.expected_value[0], shap_values[0][X,:], population.iloc[X,:])
                          
    def explain(self):    
        ind_cur_class = self.getBadClass()
    
        #Gets the domain of each feature
        features_domain = []
        features_domain = self.getFeaturesDomain()

        #The DataFrame df will have the current population
        df = pd.DataFrame(columns=self.input_dataset.columns)
        
        #Generates the initial population with popinitial mutants 
        self.getPopInicial(df, features_domain)
        
        #Generates the SHAP explanation of model
        shap.initjs()
        if self.TreeClassifier == True:
            explainerAG = shap.TreeExplainer(self.model)
        else:
            X_train_summary = shap.kmeans(self.x_train, 10)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainerAG = shap.KernelExplainer(self.model.predict_proba, X_train_summary)
        
        for g in range ( self.num_gen ):
            if self.TreeClassifier != True:
                print("GA generation ", g + 1, " - SHAP execution...")
            #To use on the parents of each generation
            parents = pd.DataFrame(columns=self.input_dataset.columns)
    
            #Copy parents to the next generation
            parents = df.copy()
            
            #df will contain the new population
            df = pd.DataFrame(columns=self.input_dataset.columns)
            
            evaluation = []                         
                   
            #Assessing generation counterfactuals
            self.fitness(parents, evaluation, explainerAG, ind_cur_class)
            
            #The original individual will always be in the 0 position of the df - So that it is normalized too (it will be used later in the distance function)
            df.loc[0] = self.original_ind.copy()
            
            #Copies to the next generation the per_elit best individuals
            self.elitism(evaluation, df, parents)
            
            i = len(df)
            while i < self.pop_size + 1: #+1, as the 1st position is used to store the reference individual
                self.crossover(df, parents, evaluation)
                
                mutation_op = rnd.random()
                if mutation_op <= self.mutation_proba:
                    self.mutation(df, len(df) - 1, features_domain)
                
                i = i + 1
                 
        evaluation = []
    
        #Evaluating the latest generation
        self.fitness(df, evaluation, explainerAG, ind_cur_class)
    
        #Order the last generation by distance to the original instance     
        evaluation.sort(key=lambda individual: individual.dist_norm)     
        
        #Getting the counterfactual set
        contrafactual_set = pd.DataFrame(columns=self.input_dataset.columns)
        contrafactual_set, solution_list = self.getContrafactual(df, evaluation)       
                 
        return contrafactual_set, solution_list