#Compas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split

from csse import CSSE
from prepare_dataset import *

def main():
    # Read Dataset Compas
    df = prepare_compas_dataset("compas-scores-two-years.csv", "./Compas/")

    #Get the input features
    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df = df[columns]
    class_name = 'class' # class = High = "Bad class" / class = Medium-Low = "Good class"
    columns_tmp = list(columns)
    columns_tmp.remove(class_name)

    x_train, x_test, y_train, y_test = train_test_split(df[columns_tmp], df[class_name], test_size=0.1)

    model = RandomForestClassifier(n_estimators = 120, n_jobs=-1, random_state=0)  
    model.fit(x_train, y_train)

    p = model.predict(x_test)

    print(classification_report(y_test, p))

    #-------Begin Parameter Adjustment--------
       
    X = 0 #Indicates the instance's position to be explained in the dataset

    #User preferences
    #static_list = [] #List of features that cannot be changed. For example: static_list = ['age']
    #K = 3 #Number of counterfactual explanations to be obtained

    #Genetic Algorithm parameters
    #num_gen = 30 #number of generations
    #pop_size = 150 #population size
    #per_elit = 0.1 #percentage of elitism
    #cros_proba = 0.8 #crossover probability
    #mutation_proba = 0.1 #mutation probability

    #Weights of objective function metrics
    #L1 = 1 #lambda 1 - Weight assigned the distance to the original instance
    #L2 = 1 #lambda 2 - Weight assigned the amount of changes needed in the original instance

    #copy the original instance
    original_instance = x_test.iloc[X].copy() 
       
    #-------End Parameter Adjustment--------

    print('Original instance - Class ' + str(p[X]) + '\n')
    print(original_instance)
    print('\nGetting counterfactuals...\n')
            
    #Run CSSE - Method executed with default parameters
    explainerCSSE = CSSE(df[columns_tmp], model)
    
    contrafactual_set, solution = explainerCSSE.explain(original_instance, p[X]) #Method returns the list of counterfactuals and the explanations generated from them
    
    #The method returns a list of counterfactual solutions, where each solution, in turn, is a change list (each change has the "column" and "value" to be changed). To implement another output format, see the "printResults" function
    explainerCSSE.printResults(solution)

if __name__ == "__main__":
    main()