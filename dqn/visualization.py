import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [12, 4]

def rolling_average(data, *, window_size):
    """Smoothen the 1-d data array using a rollin average.

    Taken from DQN programming assignments

    Args:
        data: 1-d numpy.array
        window_size: size of the smoothing window

    Returns:
        smooth_data: a 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

with open("results-medium.log", 'r') as logfile:
    results = logfile.readlines()
    returns = np.zeros(len(results))
    for idx, result in enumerate(results):
        reward = str.split(result, ', ')[2][8:]
        returns[idx] = float(reward)

plt.plot(rolling_average(returns, window_size=4), zorder=2)
plt.plot(returns, zorder=1)
plt.xlabel("Starcraft II Episode")
plt.ylabel("Returns")
plt.title("Starcraft DQN Medium Difficulty Returns")
plt.show()