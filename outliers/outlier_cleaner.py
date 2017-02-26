#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy as np
    cleaned_ages = ages
    cleaned_net_worths = net_worths
    errors = [None] * 90
    errors_abs = [None] * 90
    for index in range(0,90):
        errors_abs[index] = abs(predictions[index]-net_worths[index])
        errors[index] = (predictions[index]-net_worths[index])
    import operator
    for i in range(0,8):
        index, value = max(enumerate(errors_abs), key=operator.itemgetter(1))
        errors = np.delete(errors,index)
        errors_abs = np.delete(errors_abs,index)
        cleaned_ages = np.delete(cleaned_ages,index)
        cleaned_net_worths = np.delete(cleaned_net_worths,index)
    cleaned_data = zip(cleaned_ages,cleaned_net_worths,errors)

    ### your code goes here

    
    return cleaned_data

