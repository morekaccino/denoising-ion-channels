import os
import random
from typing import List, Tuple

import pandas as pd
import numpy as np
import neo
import plotly.express as px


class AxonData:
    def __init__(self, dirname: str):
        """
        Initialize a new instance of the AxonData class.

        Args:
            dirname (str): The directory path containing the ABF files to load.
        """
        self.data = {}
        self.filenames = []
        self.load_abf_dir(dirname)

    def reset(self):
        """
        Reset the data and filenames dictionary.
        """
        self.data = {}
        self.filenames = []

    @staticmethod
    def load_abf(filename: str) -> neo.AnalogSignal:
        """
        Load an ABF file and return its AnalogSignal.

        Args:
            filename (str): The ABF file to load.

        Returns:
            A neo.AnalogSignal object.
        """
        axon_io = neo.io.AxonIO(filename=filename)
        block = axon_io.read_block(lazy=False)
        return block.segments[0].analogsignals[0]

    def load_abf_dir(self, dirname: str) -> dict:
        """
        Load all the ABF files in the specified directory.

        Args:
            dirname (str): The directory path containing the ABF files to load.

        Returns:
            A dictionary containing the loaded data.
        """
        self.filenames = [dirname + '/' + f for f in os.listdir(dirname) if f.endswith('.abf')]
        self.filenames = sorted(self.filenames)
        for filename in self.filenames:
            signal = self.load_abf(filename)
            temp_df = pd.DataFrame({'time': signal.times.reshape(-1), 'signal': signal.magnitude.reshape(-1)})
            self.data[filename] = temp_df

    def plot(self, key, **kwargs):
        """
        Plot the time and signal data for the specified file.

        Args:
            key (int | str): The index or filename of the file to plot.
            **kwargs: Optional arguments to pass to the plotly.express line chart.

        Raises:
            KeyError: If the specified key does not exist in the data dictionary.
        """
        if isinstance(key, int):
            key = self.filenames[key]
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in data dictionary.")
        df = self.data[key]

        # if kwargs contains 'downsize', then downsize the data
        if 'downsize' in kwargs:
            downsize = kwargs['downsize']
            del kwargs['downsize']
            df = df.iloc[::downsize, :]

        fig = px.line(df, x='time', y='signal', **kwargs)
        fig.show()

    def __setitem__(self, key, value):
        """
        Set the value for the specified key in the data dictionary.

        Args:
            key: The index or filename of the key to set.
            value: The value to set for the key.
        """
        if isinstance(key, int):
            self.data[self.filenames[key]] = value
        else:
            self.data[key] = value

    def __getitem__(self, key):
        """
        Get the value for the specified key from the data dictionary.

        Args:
            key: The index or filename of the key to get.

        Returns:
            The value for the specified key.
        """
        if isinstance(key, int):
            return self.data[self.filenames[key]]
        else:
            return self.data[key]

    def __repr__(self):
        """
        Return a string representation of the data dictionary.
        """
        return str(self.data)

    def __str__(self):
        """
        Return a string representation of the data dictionary.
        """
        return str(self.data)

    def __len__(self):
        """
        Return the number of items in the data dictionary.
        """
        return len(self.data)


class AxonSimulator:
    def __init__(self,
                 noise_mean: float = -2.561512937724948e-06,
                 noise_std: float = 0.28626060085873606,
                 hertz: int = 10000,
                 duration: float = 60.0,
                 amplitude_range: Tuple[float] = (.1, .5, 1., 7., 8., 8.5, 9.),
                 probability_matrix: Tuple[Tuple[float]] = (
                         (0.0, 9000.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (5.0, 0.0, 7.7, 0.0, 0.0, 0.0, 0.0),
                         (0.0, 5.8, 0.0, 4.9, 0.0, 0.0, 0.0),
                         (0.0, 0.0, 10.0, 0.0, 7.1, 0.0, 0.0),
                         (0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0),
                         (0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 6.0),
                         (1.7, 0.0, 0.0, 0.0, 0.0, 12.8, 0.0),
                 )):
        """
        Initialize a new instance of the AxonSimulator class.

        Args:
            noise_mean (float): The mean of the normal distribution. The default value is -2.561512937724948e-06 which was calculated from the data.
            noise_std (float): The standard deviation of the normal distribution. The default value is 0.28626060085873606 which was calculated from the data.
            hertz (int): The sampling rate in Hz. The default value is 10000.
            duration (float): The duration of the signal in seconds. The default value is 60.0.
            amplitude_range (Tuple[float]): The range of amplitudes for each state to use for the signal. The default value is (0., 1., 2., 3., 4., 5., 6.).
            probability_matrix (Tuple[Tuple[float]]): The probability matrix to use for the signal. The default value is the matrix calculated from the data.
        """

        if len(probability_matrix) != len(amplitude_range):
            raise ValueError("The length of the rate matrix must be equal to the length of the amplitude range.")
        if len(probability_matrix[0]) != len(amplitude_range):
            raise ValueError(
                "The length of each row in the rate matrix must be equal to the length of the amplitude range.")
        if len(probability_matrix) != len(probability_matrix[0]):
            raise ValueError("The rate matrix must be a square matrix.")
        # sum of each row in the rate matrix must be 0
        for row in probability_matrix:
            if sum(row) <= 1e-6:
                raise ValueError("The sum of each row in the rate matrix must be 0.")

        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.hertz = hertz
        self.duration = duration
        self.amplitude_range = amplitude_range
        self.probability_matrix = probability_matrix

    def one_channel(self, switch: int = random.randint(1000, 2000)) -> pd.DataFrame:
        # choose a random state
        state = random.randint(0, len(self.amplitude_range) - 1)

        remaining_duration = self.duration
        signal = []
        for i in range(switch):
            if i == switch - 1:
                duration = remaining_duration
            else:
                duration = np.random.uniform(0.0, remaining_duration)
            duration = round(duration, 2)
            remaining_duration -= duration

            # get the amplitude for the current state
            amplitude = self.amplitude_range[state]
            # calculate the number of samples
            samples = round(duration * self.hertz)
            # generate the signal
            sub_signal = np.full(samples, amplitude)
            # generate the noise
            noise = np.random.normal(self.noise_mean, self.noise_std, samples)
            # add the noise to the signal
            sub_signal += noise
            # make sure the signal is not negative
            sub_signal[sub_signal < 0] = 0
            # add the signal to the master signal
            signal.extend(sub_signal)

            # get all possible next states and their probabilities
            next_states = list(range(len(self.amplitude_range)))
            probabilities = self.probability_matrix[state]
            # normalize the probabilities
            probabilities = [probability / sum(probabilities) for probability in probabilities]
            # choose a random next state
            next_state = np.random.choice(next_states, p=probabilities)
            # set the current state to the next state
            state = next_state


        # flatten the master signal
        # signal = np.concatenate(signal)

        # create the dataframe
        df = pd.DataFrame({'time': np.arange(0.0, self.duration, 1.0 / self.hertz), 'signal': signal})
        return df

    def simulate(self, channel_number: int = 2):
        """
        Simulate a signal with the specified number of channels.

        Args:
            channel_number (int): The number of channels to simulate. The default value is 2.

        Returns:
            A dataframe containing the simulated signal.
        """
        # create a list of dataframes
        dataframes = []
        # create a dataframe for each channel
        for i in range(channel_number):
            dataframes.append(self.one_channel())

        # sum all signals in the dataframes
        all_channels = pd.DataFrame({'time': dataframes[0]['time'], 'signal': sum([df['signal'] for df in dataframes])})
        return dataframes, all_channels
