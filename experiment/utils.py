from IPython.display import clear_output
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
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

# Display a progress bar. (clears all output)
# @param {Integer} progress
# @param {String}  title
def display_progress(progress, title = None):
    WIDTH = 20
    progress = max(0, min(1.0, progress))
    blocks = math.floor(20 * progress)
    clear_output(wait = True)
    if title:
        print(title)
    print(f"Progress: [{'#'*blocks}{'-'*(WIDTH - blocks)}] {progress*100:.2f}%")
    
# Train the given agent. The agent can be trained indefinitely or until a given maximum parameter.
# Can be stopped eraly.
def train(title, agent, *args, maxError = -1, maxEpisodes = 0, epsilon=0, simLimit=0, errorCheck=50, useBest = False, restoreBest = False, **kwargs):
    def agentError(agent):
        try:
            return agent.error()
        except AttributeError:
            return -1
    s = time.time()
    i = totalSteps = 0
    bestError = error = agentError(agent)
    bestEpisode = 0
    try:
        while error >= maxError and (not maxEpisodes or agent.episode() < maxEpisodes):
            steps = agent.train(*args, epsilon=epsilon, simLimit=simLimit, **kwargs)
            percent = 1.0 if not maxEpisodes else agent.episode()/maxEpisodes
            display_progress(percent, f"Time: {time.time()-s:.2f} seconds\nEpisode: {agent.episode()}\nError: {error}")
            totalSteps += steps or 0
            i += 1
            if i % errorCheck == 0:
                error = agentError(agent)
                if useBest and error < bestError:
                    agent.rl().backup()
                    bestError = error
                    bestEpisode = agent.episode()
                elif useBest and restoreBest:
                    agent.rl().restore()
    except KeyboardInterrupt:
        pass
    if useBest:
        print("Using best error:", bestError, "from episode:", bestEpisode)
        agent.rl().restore()
    agent.plot(title)
    
def plot(title, width, data, labels):
    t = np.arange(0, width+1, 1)
    fig, ax = plt.subplots()
    for i in range(len(data)):
        ax.plot(t, np.append(data[i], data[i][0]), 'o-', label=labels[i])
    ax.set(xlabel='Position (s)', ylabel='Expected Reward Q(s, a)',
           title=title)
    ax.grid()
    plt.legend()