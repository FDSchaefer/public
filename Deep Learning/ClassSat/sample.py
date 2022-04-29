import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

def generate_random_corner(raw_dim, sample_dim):
    return [randint( 0,raw_dim[0]-sample_dim[0]),  randint( 0,raw_dim[1]-sample_dim[1]),  0]


def extract_sample_array_from_raw(raw_array, base_corner, sample_dim):
    #print(raw_array.shape)
    #print(base_corner)
    #print(sample_dim)
    return raw_array[base_corner[0]: base_corner[0] + sample_dim[0], 
                     base_corner[1]: base_corner[1] + sample_dim[1], 
                     base_corner[2]: base_corner[2] + sample_dim[2]]

def extract_sample_arrays(raw_data, base_corner, sample_dim):
    # print("raw_data", raw_data)
    generated_samples = []
    for raw_entry in raw_data:
        generated_samples.append(extract_sample_array_from_raw(raw_entry, base_corner, sample_dim))
    return generated_samples
        
def check_candidate_samples(candidate_cubes, min_variance):
    for entry in candidate_cubes:
        #print(np.std(entry))
        if np.std(entry) >= min_variance:
            return True
    return False

def check_random_corner(generated_corners, random_corner):
    return not random_corner in generated_corners
        
def sample_images(raw_data, n_samples, sample_dim, min_variance):
    
    generated_samples = []
    generated_corners = []
    t_out = 0
    while((len(generated_samples) < n_samples)) and (t_out < 20):
        raw_dim = raw_data[0].shape
        random_corner = generate_random_corner(raw_dim, sample_dim)

        if not check_random_corner(generated_corners, random_corner):
            continue

        candidate_cubes = extract_sample_arrays(raw_data, random_corner, sample_dim)

        if check_candidate_samples(candidate_cubes, min_variance):
            generated_samples.append(candidate_cubes)
            generated_corners.append(random_corner)

        else:
            t_out += 1
            #print("cube not good.")
            pass

    
    assert(len(generated_samples) == len(generated_corners))
    return generated_samples, generated_corners
