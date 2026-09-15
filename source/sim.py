import numpy as np


def generate(jumps=5):
    """Generate a random digital signal with a given number of jumps and a fixed length of 60s and a sampling rate of 1000Hz."""
    signal_length = 60
    sampling_rate = 100

    # start with a random state and next state will be the opposite
    def next_state(state):
        return 1 - state

    states = [np.random.randint(0, 2)]
    for _ in range(jumps - 1):
        states.append(next_state(states[-1]))

    jump_times = np.random.randint(0, signal_length * sampling_rate, jumps)
    jump_times.sort()
    signal = np.zeros(signal_length * sampling_rate)
    noise = np.zeros(signal_length * sampling_rate)
    for state, jump_time in zip(states, jump_times):
        signal[jump_time:] = state
        # add noise with std of 0.1 of state is 0 and 0.3 if state is 1
        noise[jump_time:] = np.random.normal(
            0, 0.1 if state == 0 else 0.3, signal_length * sampling_rate - jump_time
        )

    return signal, signal + noise


def moving_average(x, w):
    """Calculate the moving average of a signal with the same length as the original signal with padding."""
    return np.convolve(x, np.ones(w), "same") / w
