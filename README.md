# CSSE
To use the method you must initialize an CSSE object and call the explain() function. The "test_csse_german.py" and "test_csse_compas.py" files provide examples for an individual decision.

The experiments run in "csse_evaluation_german.py" and "csse_evaluation_compas.py" perform a count of the number of changes needed to generate counterfactuals on a separate dataset sample for the explanation. In the examples, 30% of the dataset was considered for the explanation. However, running these experiments can take several hours, especially in the case of the compas dataset. Therefore, we suggest reducing the sample to be explained using the "test_size" of the classification model.
