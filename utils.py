from PIL import Image
import numpy as np
import random
import struct


def idx_load(filename):
    IDX_DATA_TYPE = {
        0x08: 'B',
        0x09: 'b',
        0x0B: 'h',
        0x0C: 'i',
        0x0D: 'f',
        0x0E: 'd',
    }
    DATA_SIZE = {
        'B': 1,
        'b': 1,
        'h': 2,
        'i': 4,
        'f': 4,
        'd': 8,
    }
    decode = lambda x: int.from_bytes(x, byteorder="big")
    with open(filename, 'rb') as f:
        # Discard the first two bytes
        f.read(2)
        
        # Get the data type
        data_type = IDX_DATA_TYPE[decode(f.read(1))]
        data_size = DATA_SIZE[data_type]
        
        # Grab the number of dimensions
        dim = decode(f.read(1))
        
        # Sizes of each dimension
        sizes = [decode(f.read(4)) for size in range(dim)]
        
        # The result array
        result = np.zeros(sizes)
        
        indices = [0]*len(sizes)
        end = len(indices) - 1
        while indices[0] < sizes[0]:
            [result[tuple(indices)]] = struct.unpack(f">{data_type}", f.read(data_size))
            indices[end] += 1
            if indices[end] == sizes[end]:
                i = end
                while i > 0 and indices[i] >= sizes[i]:
                    indices[i] = 0
                    i -= 1
                    indices[i] += 1
        if data_type != 'f':
            return result.astype(int)
        return result
    

def shuffle_data(data, labels):
    pass