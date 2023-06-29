import modified_improved_diffusion.importing as importing
import blobfile as bf
import numpy as np
from PIL import Image
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset

def load_data(
        *, particle_type, batch_size, number_of_vectors="all", class_cond=False, deterministic=False
):
    if not particle_type:
        raise ValueError(("unspecified particle type. Use 'e' for electrons"
                          "or 'mu' for muons"))
    
    classes = None
    #class conditioning
    # if class_cond:
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]

    data_arr = importing.muon_events(number_of_vectors, not deterministic)

    dataset = fourVectorDataset(
        data_arr,
        classes,
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
    
    print("image_datasets.py, loader: ", type(loader), np.shape(loader), loader)
    while True:
        yield from loader


class fourVectorDataset(Dataset):
    def __init__(self, data_arr, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.data_arr = data_arr[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return np.shape(self.data_arr)[0]

    def __getitem__(self, index):
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[index], dtype=np.int64)
        return self.data_arr[index], out_dict #output shape ist im original 3D
    

