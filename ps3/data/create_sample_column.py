import hashlib  # this library allows us to convert strings into a list of numbers-- importantly this list will be the same for the same name 
#i.e. if the str is 'Hello!' it will output 5d41402abc4b2a76b9719d911017c592 (using hexadecimal)-- everytime will return same numbers for the same string
#this allows us to deal with the issue of parsing strings for a split 
# and also to avoid any fear that there is correlation or clustering between ids
import numpy as np  

def create_sample_split(df, id_column, training_frac=0.8):
    """..."""
    
 # Get the ID column
    ids = df[id_column]
    
    # If IDs are strings, hash them to get integers
    if ids.dtype == 'object' or ids.dtype == 'string':
        # Hash each ID and convert to integer--- unifomrly distributed across possible values 
        hash_values = ids.apply(lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16))
    else:
        # Already numeric, use directly
        hash_values = ids
    
    # Use modulo to create deterministic split--- modulo will uniformly split even numbers of ids between 0-99
    #i.e. if 1000 ids, 10 at 0, 10 at 1.... 10 at 99 
    threshold = int(training_frac * 100)
    df['sample'] = np.where(hash_values % 100 < threshold, 'train', 'test')
    
    return 