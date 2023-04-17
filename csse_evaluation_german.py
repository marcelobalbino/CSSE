#Experiment that assesses CSSE's performance in the search for counterfactuals with minimal changes to the original instance with German dataset

#This experiment can take several minutes. To reduce the time adjust the "test_size" in the classification model.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statistics

from csse import CSSE
from prepare_dataset import *

def main():
    print('This experiment can take a long time, depending on the number of decisions to explain. To reduce time, adjust the variable "num_inst".\n')
    
    # Read Dataset German
    df = prepare_german_dataset("german_credit.csv", "./German/")

    #Get the input features
    columns = df.columns
    class_name = 'default' # default = 0 = "Good class" / default = 1 = "Bad class" 
    columns_tmp = list(columns)
    columns_tmp.remove(class_name)

    x_train, x_test, y_train, y_test = train_test_split(df[columns_tmp], df[class_name], test_size=0.3)

    model = RandomForestClassifier(n_estimators = 120, n_jobs=-1, random_state=0)  
    model.fit(x_train, y_train)

    p = model.predict(x_test)

    print(classification_report(y_test, p))

    #-------Begin Parameter Adjustment-------- 
    #User preferences
    #static_list = [] #List of features that cannot be changed
    #K = 3 #Number of counterfactual explanations to be obtained

    #Genetic Algorithm parameters
    #num_gen = 30 #number of generations
    #pop_size = 100 #population size
    #per_elit = 0.1 #percentage of elitism
    #cros_proba = 0.8 #crossover probability
    #mutation_proba = 0.1 #mutation probability

    #Weights of objective function metrics
    #L1 = 0 #lambda 1 - Weight related to distance for class of interest
    #L2 = 1 #lambda 2 - Weight related to distance for original instance
      
    #List that contains how many changes were necessary in the generation of each counterfactual
    change_list = []
    
    #Count number of cases where solution was found
    count_solution = 0
    
    #Defines how many instances of the test set will be used
    #num_inst = len(x_test)
    #Running the method with 100 instances of the test set
    num_inst = 5
    print('Number of decisions to be explained: ', num_inst)
    
    for X in range ( 0, num_inst): #X Indicates the instance's position to be explained in the dataset
        #copy the original instance
        original_instance = x_test.iloc[X].copy()         
     
    #-------End Parameter Adjustment--------   
        
        #Print the original instance
        print('# Original instance ' + str(X + 1) + ' - Class ' + str(p[X]) + ' #\n')
        print(original_instance)
        print('\nGetting counterfactuals...\n')
        
        #Run CSSE - Method executed with default parameters.
        explainerCSSE = CSSE(df[columns_tmp], model)
    
        contrafactual_set, solution = explainerCSSE.explain(original_instance, p[X])
                        
        #Prints the counterfactuals found
        explainerCSSE.printResults(solution)    
        
        print('\nSummarizing results from instance ' + str(X + 1) + '\n')
        if len(solution) != 0:
            for y in range ( 0, len(solution)):
                n = len(solution[y])
                print('Counterfactual ' + str(y + 1) + ' - ' + str(n) + ' change(s)\n')
                change_list.append(n)
                count_solution = count_solution + 1     
        else:
            print('Instance ' + str(X + 1) + ' - solution not found\n')
                                
    print('Mean and standard deviation of the number of changes')
    print(f'Mean: {statistics.mean(change_list):.2f}')
    print(f'Standard deviation: {statistics.pstdev(change_list, mu=None):.2f}')
    
    #method_effi = (count_solution/(num_inst*K))*100
    #Efficacy measures the percentage of counterfactuals obtained concerning the expected total number (num_ins * K).
    #print(f'Method efficacy: {method_effi:.2f}')
    
if __name__ == "__main__":
    main()