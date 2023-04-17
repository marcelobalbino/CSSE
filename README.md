# CSSE
CSSE is a counterfactual explanation method that works with tabular data and classification problems with binary output. The method is implemented in Python. Briefly, to use the method, the user must:

1) Import the method

from csse import CSSE


2) You must initialize a CSSE object with the model and the input data (without the class) as parameters. 

explainerCSSE = CSSE(input_dataset, model)


3) Then, you can generate explanations for the instance of interest. The function's parameters are the input data of the original instance and its current class.

contrafactual_set, solution = explainerCSSE.explain(original_ind, current_class)


4) The variable "counterfactual_set" contains the list of generated counterfactual examples. The variable "solution" includes only the features that were changed to create each counterfactual. You can print the solution using the function printResults(solution) or implement your function to display the results.

explainerCSSE.printResults(solution)


The "test_csse_german.py" and "test_csse_compas.py" files provide examples of an individual decision. In addition, these examples present the optional parameters of the explain () function as comments. 

To exemplify the use of an optional parameter, in "test_csse_german.py", we utilize the optional parameter K = 5 to generate five counterfactual explanations.  

K = 5

explainerCSSE = CSSE(input_dataset, model, K = K)


The experiments run in "csse_evaluation_german.py" and "csse_evaluation_compas.py" perform a count of the number of changes needed to generate counterfactuals on a separate dataset sample for the explanation. In these examples, we used 100 instances of the test set. However, running these experiments can take several hours. Therefore, we suggest reducing the sample to be explained using the variable "num_inst".
