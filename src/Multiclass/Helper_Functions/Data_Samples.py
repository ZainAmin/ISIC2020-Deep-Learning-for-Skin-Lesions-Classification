def get_samples_count(y, sampling_technique):
    """
    Determines the number of samples required for balancing a dataset.

    Parameters:
    - y (numpy.ndarray): Array of class labels.
    - sampling_technique (str): Sampling technique ('mean', 'over', or 'under').

    Returns:
    - int: Number of samples required based on the specified technique.
    """
    unique, counts = np.unique(y, return_counts=True)  

    if sampling_technique == 'mean':
        mean_samples = np.mean(counts) * 3  
    elif sampling_technique == 'over':
        mean_samples = np.max(counts) * 3  
    elif sampling_technique == 'under':
        mean_samples = np.min(counts) * 3  
    else:
        raise Exception("Error: Sampling Technique not implemented")

    return int(mean_samples)