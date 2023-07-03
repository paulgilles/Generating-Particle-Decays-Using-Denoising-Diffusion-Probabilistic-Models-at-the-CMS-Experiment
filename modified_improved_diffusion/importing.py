import numpy as np
import torch as th
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset
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

def calc_pT(data):
    px = data[:, :, 1]
    py = data[:, :, 2]
    pT = np.sqrt(px**2 + py**2)
    return pT

def calc_pseudo_rapidity(data):
    px = data[:, :, 1]
    py = data[:, :, 2]
    pz = data[:, :, 3]
    abs_p = np.sqrt(px**2 + py**2 + pz**2)
    return 1/2 * np.log((abs_p + pz) / (abs_p - pz))

def preprocess(data, min_max_norm):
    if not min_max_norm:
        mean = np.mean(data, axis=0, keepdims=True) #@todo mean und std abspeichern
        std = np.std(data, axis=0, keepdims=True)
        normalized = (data - mean)/std
    elif min_max_norm:
        minimum_of_both_vectors, maximum_of_both_vectors = find_min_and_max(
            data, count_number=10, intervals=[(0.2, 0.2), (0.2, 0.2)]
        )
        normalized = (data - minimum_of_both_vectors) / (maximum_of_both_vectors - minimum_of_both_vectors) * 2 - 1
    return normalized



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



def load_data(*, particle_type, batch_size, class_cond=False, deterministic=False, preprocessing=True, min_max_norm=False):
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
        data = preprocess(data, min_max_norm)


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
