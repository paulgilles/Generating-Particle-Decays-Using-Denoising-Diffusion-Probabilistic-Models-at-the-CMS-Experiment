import numpy as np
import torch as th
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset
from scipy.stats import entropy, wasserstein_distance
from modified_improved_diffusion.evaluation_util import load_npz
import os

path = "/net/scratch/cms/data/toysets/Zll_sorted/data_delphes.npy"

def load_all_data(path):
    data = np.load(path)
    return data

def cut_data(data, cut_pos = 256130):
    return data[:cut_pos], data[cut_pos:]

def is_number_valid(number, particle):
    if particle == "e":
        max_number = 256130
    elif particle == "mu":
        max_number = 380019
    else: 
        raise AttributeError(f"No data found for the particle called '{particle}'.")

    if type(number) == str:
        if number == "all":
            return True 
    else:
        if number <= max_number:
            return True
        else: 
            raise AttributeError(f"There are not enough datapoints. To use all, tpye 'all'. You typed {number}.")

def electron_events(number, shuffle=True):
    electrons, _ = cut_data(load_all_data(path))
    if shuffle == True:
        np.random.shuffle(electrons)
    if is_number_valid(number, "e"):
        if number == "all":
            return electrons
        else: 
            return electrons[:number]    
    
def muon_events(number, shuffle=True):
    _, muons = cut_data(load_all_data(path))
    if shuffle == True:
        np.random.shuffle(muons)
    if is_number_valid(number, "mu"):
        if number == "all":
            return muons
        else: 
            return muons[:number]

def calc_pT(data, vector_pos=None):
    if np.shape(data)[1:] == (2,4):
        px = data[:,:,1]
        py = data[:,:,2]
        single_vector = False
    elif np.shape(data)[1:] == (4,):
        px = data[:,1]
        py = data[:,2]
        single_vector = True
    else: 
        raise ValueError(f"Unknown shape for eta calculation: shape={np.shape(data)}.")
    pT = np.sqrt(px**2 + py**2)
    if vector_pos is not None and not single_vector:
        pT = pT[:, vector_pos]
    return pT


def calc_pseudo_rapidity(data, vector_pos=None):
    if np.shape(data)[1:] == (2,4):
        px = data[:,:,1]
        py = data[:,:,2]
        pz = data[:,:,3]
        single_vector = False
    elif np.shape(data)[1:] == (4,):
        px = data[:,1]
        py = data[:,2]
        pz = data[:,3]
        single_vector = True
    else: 
        raise ValueError(f"Unknown shape for eta calculation: shape={np.shape(data)}.")
    abs_p = np.sqrt(px**2 + py**2 + pz**2)
    eta = 1/2 * np.log((abs_p + pz) / (abs_p - pz))
    if vector_pos is not None and not single_vector:
        eta = eta[:, vector_pos]
    return eta


def calc_Z_vector(data):
    p1 = data[:, 0, :]
    p2 = data[:, 1, :]
    z = p1 + p2
    return z


def calc_m(data, vector_pos=None):
    if np.shape(data)[1:] == (2,4):
        E = data[:,:,0]
        px = data[:,:,1]
        py = data[:,:,2]
        pz = data[:,:,3]
        single_vector = False
    elif np.shape(data)[1:] == (4,):
        E = data[:,0]
        px = data[:,1]
        py = data[:,2]
        pz = data[:,3]
        single_vector = True
    else: 
        raise ValueError(f"Unknown shape for mass calculation: shape={np.shape(data)}.")
    m = np.sqrt(E**2 - (px**2+py**2+pz**2))
    if vector_pos is not None and not single_vector:
        m = m[:, vector_pos]
    return m


def calc_KL(distribution1, distribution2):
    range = (np.min(distribution2), np.max(distribution2))
    distribution1, _ = np.histogram(distribution1, bins=70, range=range)
    distribution2, _ = np.histogram(distribution2, bins=70, range=range)
    epsilon = 1e-7
    distribution1, distribution2 = distribution1.astype("float64"), distribution2.astype("float64")
    distribution1 += epsilon
    distribution2 += epsilon
    return entropy(distribution1, distribution2)


def calc_wasserstein_sum(result_folder, create_pdf=False, npz_file=None,
                         particle_type="muons", min_max_norm=True):
    npz = load_npz(npz_file)
    if particle_type=="muons":
        data = muon_events("all", False)
    elif particle_type=="electrons":
        data = electron_events("all", False)
    else:
        raise ValueError(f"No valid input for 'particle_type'. {particle_type}")
    _, min, max = preprocess(data, min_max_norm=min_max_norm, full_output=True)
    npz = np.transpose(npz, (0,2,1))
    npz = postprocess(npz, min_max_norm, min, max)
    np.random.shuffle(data)
    data = data[:len(npz)]

    npz_reshaped = npz.reshape(np.shape(npz)[0]*np.shape(npz)[1], np.shape(npz)[2])
    data_reshaped = data.reshape(np.shape(data)[0]*np.shape(data)[1], np.shape(data)[2])
    wasserstein_sum = 0
    for component in [0,1,2,3]:
        range = (np.min(npz_reshaped), np.max(npz_reshaped))
        npz_p , _ = np.histogram(npz_reshaped[:, component], bins=70,range=range)
        data_p, _ = np.histogram(data_reshaped[:, component], bins=70, range=range)
        wasserstein_sum += wasserstein_distance(npz_p, data_p)
    filename = f"Wasserstein={wasserstein_sum}.txt"
    target = os.path.join(result_folder, filename)
    with open(target, 'w') as file:
        file.write(f"Wasserstein-Abstand: {wasserstein_sum}")
    print("wasserstein_sum was calculated.")
    return wasserstein_sum


def preprocess(data, min_max_norm, count_number=10, intervals=[(0.2, 0.2), (0.2, 0.2)], full_output=False):
    if not min_max_norm:
        mean = np.mean(data, axis=0, keepdims=True) #@todo mean und std abspeichern
        std = np.std(data, axis=0, keepdims=True)
        normalized = (data - mean)/std
        if full_output:
            return normalized, mean, std
    elif min_max_norm:
        minimum_of_both_vectors, maximum_of_both_vectors = find_min_and_max(
            data, count_number=count_number, intervals=intervals
        )
        normalized = (data - minimum_of_both_vectors) / (maximum_of_both_vectors - minimum_of_both_vectors) * 2 - 1
        if full_output:
            return normalized, minimum_of_both_vectors, maximum_of_both_vectors
    return normalized



def postprocess(data, min_max_norm, arg1, arg2):
    """Does the preprocess ‘in reverse‘.

    Args:
        data (array_like): data which should be postprocess
        min_max_norm (bool): Whether min_max_norm was used or not
        arg1 (float): mean or minimum
        arg2 (float): std or maximum
    """
    if not min_max_norm:
        raise NotImplementedError(("postprocessing for min_max_norm=False"
                                   " is not implemented."))
    else:
        minimum = arg1
        maximum = arg2
        data = (data + 1) * (maximum - minimum) / 2 + minimum
    return data


def calc_min_and_max_counts(data, minimum, maximum, intervals):
    minimum_interval, maximum_interval = intervals
    minimum_ulimit = minimum + minimum_interval[1]
    maximum_ulimit = maximum + maximum_interval[1]
    minimum_llimit = minimum - minimum_interval[0]
    maximum_llimit = maximum - maximum_interval[0]

    minimum_count = np.sum(np.sum(np.logical_and(data > minimum_llimit, data < minimum_ulimit), axis=0), axis=0)
    maximum_count = np.sum(np.sum(np.logical_and(data > maximum_llimit, data < maximum_ulimit), axis=0), axis=0)
    return minimum_count, maximum_count



def find_min_and_max(data, count_number, intervals):
    reshaped = np.reshape(data, (np.shape(data)[0]*np.shape(data)[1],4))
    sorted = np.sort(reshaped, axis=0)
    minimum = sorted[0]
    maximum = sorted[::-1][0]

    minimum_count, maximum_count = calc_min_and_max_counts(data, 
                                                           minimum, 
                                                           maximum,
                                                           intervals)

    minimum_line = np.array([0,0,0,0])
    maximum_line = np.array([0,0,0,0])
    while ((maximum_count<count_number).any() or (minimum_count<count_number).any()):

        minimum_line += (minimum_count<count_number)
        maximum_line += (maximum_count<count_number)
        minimum = np.array([sorted[minimum_line[0]][0], sorted[minimum_line[1]][1], 
                        sorted[minimum_line[2]][2], sorted[minimum_line[3]][3]])
        maximum = np.array([sorted[::-1][maximum_line[0]][0], sorted[::-1][maximum_line[1]][1], 
                        sorted[::-1][maximum_line[2]][2], sorted[::-1][maximum_line[3]][3]])

        minimum_count, maximum_count = calc_min_and_max_counts(data, 
                                                               minimum,
                                                               maximum,
                                                               intervals)
    
    return minimum, maximum



def load_data(*, particle_type, batch_size, class_cond=False, 
              deterministic=False, preprocessing=True, min_max_norm=False,
              preprocess_count_number=10, preprocess_intervals=[(0.2,0.2), (0.2,0.2)]):
    if not particle_type:
        raise ValueError(("unspecified particle type. It has to be 'muons', "
                          "'electrons' or 'both'."))
        
    if particle_type != "all":
        if particle_type == "electrons":
            index = 0
        elif particle_type == "muons":
            index = 1
        data = cut_data(load_all_data(path))[index]
    else: 
        data = load_all_data(path)

    if preprocessing:
        data = preprocess(data, min_max_norm, preprocess_count_number, 
                          preprocess_intervals)


    classes=None
    if class_cond:
        classes = (["electrons"] * len(electron_events("all")) 
                  +["muons"] * len(muon_events("all")))

    dataset = VectorDataset(
        data=data,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )

    if deterministic:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
            )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )

    while True:
        yield from loader


class VectorDataset(Dataset):
    def __init__(self, data, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.data = data
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        sample = sample.astype(np.float32)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[index], dtype=np.int64)

        #Hier kann es sein, dass das sample noch transposed werden muss 
        #oder eine Dummy Dim hinzugefügt werden muss oder so
        return sample, out_dict
