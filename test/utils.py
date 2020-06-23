from IPython.display import clear_output
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

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