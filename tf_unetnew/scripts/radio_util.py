import h5py
import numpy as np
from tf_unetnew.tf_unet.image_util import BaseDataProvider


class DataProvider(BaseDataProvider):
    channels = 1
    n_class = 2
    
    def __init__(self, nx, files, a_min=30, a_max=210):
        super(DataProvider, self).__init__(a_min, a_max)
        self.nx = nx
        self.files = files
        
        assert len(files) > 0, "No training files"
        print("Number of files used: %s"%len(files))
        self._cylce_file()
    
    def _read_chunck(self):
        with h5py.File(self.files[self.file_idx], "r") as fp:
            nx = fp["data"].shape[1]
            idx = np.random.randint(0, nx - self.nx)
            
            sl = slice(idx, (idx+self.nx))
            data = fp["data"][:, sl]
            rfi = fp["mask"][:, sl]
        return data, rfi
    
    def _next_data(self):
        data, rfi = self._read_chunck()
        nx = data.shape[1]
        while nx < self.nx:
            self._cylce_file()
            data, rfi = self._read_chunck()
            nx = data.shape[1]
            
        return data, rfi

    def _cylce_file(self):
        self.file_idx = np.random.choice(len(self.files))
        
