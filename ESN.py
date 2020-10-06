#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import absolute_import, division, print_function
import pandas as pd # standard library
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from scipy.stats import kurtosis
import scipy.stats
import scipy.signal as scipy_signal
from sklearn.cluster import KMeans

import tensorflow as tf # third party packages
tf.enable_eager_execution()
import shap
from pingouin import partial_corr

import differential_evolution # local source
import gan.conditional_gan as denoising_gan
from gan.config_gan import ConfigCGAN as config
from gan.data_processing_gan import spline_signal
from esn import ESN

__author__ = "Marcia Baptista"
__copyright__ = "Copyright 2020, The XAI Prognostics Project"
__credits__ = ["Marcia Baptista"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Marcia Baptista"
__email__ = "marcia.lbaptista@gmail.com"
__status__ = "Prototype"

shap_features = defaultdict(float)

class Program:

    def __init__(self, name):
        self.name = name
        # creates a new dictionary for saving feature "strength"
        self.strenght_features = defaultdict(float)

    def init_data(self, data):
        self.data = data

class GANDenoiser:

    def __init__(self, sensor_name):
        self.name = sensor_name
        self.generator = denoising_gan.make_generator_model_small()
        self.discriminator = denoising_gan.make_discriminator_model()
        self.generator_optimizer = tf.train.AdamOptimizer(config.learning_rate)
        self.discriminator_optimizer = tf.train.AdamOptimizer(config.learning_rate)

        print("Retrieving checkpoint for sensor", self.name)
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        print("Checkpoint file:", tf.train.latest_checkpoint("./models/" + self.name))
        checkpoint.restore(tf.train.latest_checkpoint("./models/" + self.name))

class EchoStateNetwork:

    def __init__(self):
        self.bounds = [(10, 300), (0.01, 5.0), (0.01, 0.5), (10e-7, 1), (-1, 1), (10e-7, 1), (-1, 1),
                       (-0.5, 0.5)]  # bounds [(x1_min, x1_max), (x2_min, x2_max),...]
        self.popsize = 100  # population size, must be >= 4
        self.mutate = 0.5  # mutation factor [0,2] 0.5
        self.recombination = 0.7  # recombination rate [0,1]
        self.maxiter = 10 # maximum iterations

    def fit_and_predict(self, sensor_names, training_samples, training_RULs, configuration_samples, configuration_RULs,
                        validation_samples, validation_RULs, testing_samples, testing_RULs, testing_time):
        self.configuration_samples = configuration_samples
        self.configuration_RULs = configuration_RULs
        self.validation_samples = validation_samples
        self.validation_RULs = validation_RULs
        self.testing_time = testing_time
        x = self.optimize_network()
        self.x = x
        n_reservoir, spectral_radius, sparsity, input_scaling, input_shifting, output_scaling, output_shifting, state_noise = \
            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
        self.esnet = ESN.ESN(n_inputs=len(sensor_names), n_outputs=1, n_reservoir=int(n_reservoir),
                      spectral_radius=spectral_radius, sparsity=sparsity, random_state=42,
                      input_shift=input_shifting, input_scaling=input_scaling,
                      teacher_scaling=output_scaling,
                      teacher_forcing=True,
                      teacher_shift=output_shifting,
                      noise=state_noise)
        self.esnet.fit(np.array(training_samples), np.array(training_RULs))
        predicted_RULs = self.esnet.predict(np.array(testing_samples), True)
        self.predicted_RULs = [item for sublist in predicted_RULs for item in sublist]
        self.testing_RULs = testing_RULs

    def check_accuracy(self, x):
        n_reservoir, spectral_radius, sparsity, input_scaling, input_shifting, output_scaling, output_shifting, state_noise = \
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
        esn = ESN.ESN(n_inputs=len(sensor_names), n_outputs=1, n_reservoir=int(n_reservoir),
                      spectral_radius=spectral_radius, sparsity=sparsity, random_state=42,
                      input_shift=input_shifting, input_scaling=input_scaling,
                      teacher_scaling=output_scaling,
                      teacher_forcing=True,
                      teacher_shift=output_shifting,
                      noise=state_noise)
        esn.fit(np.array(self.configuration_samples), np.array(self.configuration_RULs))
        predictions = esn.predict(np.array(self.validation_samples))
        predicted_RULs = [item for sublist in predictions for item in sublist]

        errors = [abs(a - b) * 400.0 for a, b in zip(predicted_RULs, self.validation_RULs)]
        return np.mean(errors)

    def optimize_network(self):
        x = ESN.diffev_minimize(self.check_accuracy, self.bounds, self.popsize, self.mutate, self.recombination, self.maxiter)
        return x

    def calculate_errors(self):
        #self.predicted_RULs = np.array([135 if a_ > 135 else a_ for a_ in 400 * np.array(self.predicted_RULs)])
        self.testing_RULs = np.array(self.testing_RULs) * 400
        self.testing_time = np.array(self.testing_time) * 400
        #self.predicted_RULs =[100 if b_ > 100 and a_ + c_ < 50 else b_ for a_, b_, c_ in zip(self.testing_RULs, self.predicted_RULs, self.testing_time)]

        self.predicted_RULs = 400 * np.array(self.predicted_RULs)

        diff = (np.array(self.predicted_RULs) - np.array(self.testing_RULs))

        fp = np.count_nonzero(np.array(self.predicted_RULs)[np.array(self.predicted_RULs) > np.array(self.testing_RULs) + 13])
        fn = np.count_nonzero(
            np.array(self.predicted_RULs)[np.array(self.predicted_RULs) < np.array(self.testing_RULs) -10])

        print("Fp:", fp/len(diff))
        print("Fn:", fn/len(diff))
        testing_RULs_non_zero = np.array(self.testing_RULs) != 0

        mask = diff < 0
        diff_negative = np.extract(mask, diff)
        mask = diff >= 0
        diff_positive = np.extract(mask, diff)
        l3 = list(np.exp((-diff_negative)/13)) + list(np.exp(diff_positive/10))
        print("Score: ", np.sum(np.array(l3)))
        print("Mean Score: ", np.mean(np.array(l3)))
        print("MAE Error:", np.mean(np.abs(np.array(self.predicted_RULs) - np.array(self.testing_RULs))))
        print("MAPE Error:", np.mean(np.abs((np.array(self.testing_RULs)[testing_RULs_non_zero] - np.array(self.predicted_RULs)[testing_RULs_non_zero])/np.array(self.testing_RULs)[testing_RULs_non_zero])))
        print("RMSE Error:", math.sqrt(np.mean(np.power(np.abs(np.array(self.predicted_RULs) - np.array(self.testing_RULs)), 2))))



class CMAPSSData:

    def __init__(self, csv_name, units = None):
        self.feature_names = ['Unit', 'time', 'altitude', 'mach number', 'throttle_resolver_angle',
                              'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi',
                              'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
        self.non_flat_sensor_names = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf',
                                      'NRc', 'BPR', 'htBleed', 'W31', 'W32']
        self.all_sensor_names = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi',
                                 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
        self.cobble_sensor_names = ['T24','T30','T50','Nc','Ps30','NRc','BPR','htBleed','W31','W32']

        self.units = units
        self.dataframe = None
        self.number_units = 0
        self.set_features_interest(self.non_flat_sensor_names)
        self.start(csv_name=csv_name, units=units)
        self.GANS = {}

    def prepare_denoisers(self):
        self.GANS = {}
        for sensor_name in self.sensor_names:
            self.GANS[sensor_name] = GANDenoiser(sensor_name)

    def start(self, csv_name, units):
        data = [pd.read_csv(csv_name, sep='\s+', names=self.feature_names)]
        self.dataframe = self.read_pandas_array(data)
        self.time = np.unique(self.dataframe['time'])
        if units is None:
            self.units = np.unique(self.dataframe['Unit'])
            np.random.shuffle(self.units)
        else:
            self.units = units
        self.number_units = len(self.units)
        self.validation_units = self.units[:int(self.number_units * 0.10)]
        self.configuration_units = self.units[int(self.number_units * 0.10):int(self.number_units * 0.20)]
        self.training_units = self.units[int(self.number_units * 0.20):int(self.number_units * 0.60)]
        self.testing_units = self.units[int(self.number_units * 0.60):]
        self.time = self.time[int(self.number_units * 0.60):]

    def print_general_information(self):
        print("Engine units: ", self.units)
        print("Number of engine units: ", self.number_units)
        print("Training Units:", self.training_units)
        print("Testing Units:", self.testing_units)

    def read_pandas_array(self, pd_array):
        frames = []
        for i in range(len(pd_array)):
            frames.append(pd_array[i])
        return pd.concat(frames, ignore_index=True)

    def plot_raw_sensor_data(self):
        for sensor_name in self.sensor_names:
            signal = np.array(self.dataframe.loc[self.dataframe['Unit'] == 1, sensor_name].values)
            plt.scatter(range(len(signal)), signal)
            plt.xlabel('Time (cycles)')
            plt.title(sensor_name)
            plt.show()

    def plot_raw_baselined_sensor_data(self):
        for sensor_name in self.sensor_names:
            signal = np.array(self.dataframe.loc[self.dataframe['Unit'] == 1, sensor_name].values)
            plt.scatter(range(len(signal)), signal, c='black', label="raw")
            plt.xlabel('Time (cycles)')
            plt.ylabel(sensor_name)
            plt.title(sensor_name)
            plt.show()
            signal = np.array(self.dataframe.loc[self.dataframe['Unit'] == 1, sensor_name + 'baselined'].values)
            plt.scatter(range(len(signal)), signal, c='red', label="baselined")
            plt.xlabel('Time (cycles)')
            plt.ylabel(sensor_name + ' baselined')
            plt.title(sensor_name)
            plt.show()

    def plot_baselined_denoised_sensor_data(self):
        for sensor_name in self.sensor_names:
            signal = np.array(self.dataframe.loc[self.dataframe['Unit'] == 1, sensor_name + 'baselined'].values)
            plt.scatter(range(len(signal)), signal, c='red', label="baselined")
            signal = np.array(self.dataframe.loc[self.dataframe['Unit'] == 1, sensor_name + 'denoised'].values)
            plt.scatter(range(len(signal)), signal, c='black', label="denoised")
            plt.xlabel('Time (cycles)')
            plt.ylabel(sensor_name + ' baselined')
            plt.title(sensor_name)
            plt.show()

    def set_features_interest(self, sensor_names):
        self.sensor_names = sensor_names

    def baseline(self, number_regimes):
        k_means = KMeans(n_clusters=number_regimes, random_state=0).fit(
                self.dataframe.loc[:, ['altitude', 'mach number', 'throttle_resolver_angle']])
        for sensor_name in self.sensor_names:
                for regime in range(number_regimes):
                    signal_regime = self.dataframe.loc[(k_means.labels_ == regime), sensor_name].values
                    if np.std(signal_regime) != 0 and sensor_name != 'time':
                        signal_regime = (signal_regime - np.mean(signal_regime)) / (np.std(signal_regime))
                    if sensor_name == 'time':
                        signal_regime = (signal_regime) / (400)
                    self.dataframe.loc[k_means.labels_ == regime, sensor_name + 'baselined'] = signal_regime

    def add_RUL(self):
        self.dataframe['RUL'] = 0
        for unit in self.units:
            noisy_T24_signal_unit = np.array(self.dataframe.loc[self.dataframe['Unit'] == unit, 'T24'].values)
            life_unit = len(noisy_T24_signal_unit)
            self.dataframe.loc[self.dataframe['Unit'] == unit, 'RUL'] = range(life_unit - 1, -1, -1)
            self.dataframe.loc[self.dataframe['Unit'] == unit, 'RUL'] = self.dataframe.loc[self.dataframe['Unit'] == unit, 'RUL'].values  / 400.0

    def denoise(self):
        for unit in self.units:
            self.denoise_unit(unit)

    def denoise_unit(self, unit):
        noisy_T24_signal_unit = np.array(self.dataframe.loc[self.dataframe['Unit'] == unit, 'T24'].values)
        life_unit = len(noisy_T24_signal_unit)

        for sensor_name in self.sensor_names:
            noisy_signal = np.array(self.dataframe.loc[self.dataframe['Unit'] == unit, sensor_name + 'baselined'].values)

            # enlarge/compress the signal to 400 points
            noisy_splined = spline_signal(noisy_signal, config.height * config.width)

            # prepare the data for the convolutional format
            noisy_samples = []
            noisy_sample = (noisy_splined.reshape(config.width, config.height))
            noisy_samples.append(noisy_sample)
            noisy_samples = np.array(noisy_samples).astype('float32')
            noisy_input = noisy_samples.reshape(
                (noisy_samples.shape[0], noisy_samples.shape[1], noisy_samples.shape[2], 1))

            # denoise the data with the generative adversarial network (GAN)
            predictions = self.GANS[sensor_name].generator(noisy_input, training=False)

            splined_reconstruction = np.array(predictions[0]).reshape((config.width * config.height,))
            noisy = np.array(noisy_input[0]).reshape((config.width * config.height,))
            reconstruction = scipy_signal.decimate(splined_reconstruction, 2)
            reconstruction = spline_signal(reconstruction, life_unit)

            # put the denoised data back into the dataframe
            self.dataframe.loc[self.dataframe['Unit'] == unit, sensor_name + 'denoised'] = reconstruction

    def get_training_RULs(self):
        return self.dataframe.loc[self.dataframe['Unit'].isin(self.training_units), 'RUL'].values

    def get_training_samples(self):
        sensor_names_denoised = []
        for sensor_name in self.sensor_names:
            sensor_names_denoised.append(sensor_name + 'denoised')
        return self.dataframe.loc[self.dataframe['Unit'].isin(self.training_units), sensor_names_denoised].values

    def get_validation_RULs(self):
        return self.dataframe.loc[self.dataframe['Unit'].isin(self.validation_units), 'RUL'].values

    def get_validation_samples(self):
        sensor_names_denoised = []
        for sensor_name in self.sensor_names:
            sensor_names_denoised.append(sensor_name + 'denoised')
        return self.dataframe.loc[self.dataframe['Unit'].isin(self.validation_units), sensor_names_denoised].values

    def get_configuration_RULs(self):
        return self.dataframe.loc[self.dataframe['Unit'].isin(self.configuration_units), 'RUL'].values

    def get_configuration_samples(self):
        sensor_names_denoised = []
        for sensor_name in self.sensor_names:
            sensor_names_denoised.append(sensor_name + 'denoised')
        return self.dataframe.loc[self.dataframe['Unit'].isin(self.configuration_units), sensor_names_denoised].values

    def get_testing_RULs(self):
        return self.dataframe.loc[self.dataframe['Unit'].isin(self.testing_units), 'RUL'].values

    def get_testing_units(self):
        return self.dataframe.loc[self.dataframe['Unit'].isin(self.testing_units), 'Unit'].values

    def get_testing_time(self):
        return self.dataframe.loc[self.dataframe['Unit'].isin(self.testing_units), 'time'].values

    def get_testing_samples(self):
        sensor_names_denoised = []
        for sensor_name in self.sensor_names:
            sensor_names_denoised.append(sensor_name + 'denoised')
        return self.dataframe.loc[self.dataframe['Unit'].isin(self.testing_units), sensor_names_denoised].values

    def calculate_monotonicity(self, nr_points_slope=10):
        self.monotonicity_sensors = defaultdict(float)
        for feature_name in self.sensor_names:
            monotonicity_feature = 0
            nr_units = len(self.testing_units)
            for unit in self.testing_units:
                # obtain signal from dataframe
                smooth_signal = self.dataframe.loc[self.dataframe['Unit'] == unit, feature_name + 'denoised'].values

                life_unit = len(smooth_signal)
                monotonicity_unit = 0

                for index in range(nr_points_slope, life_unit):
                    monotonicity_unit += math.copysign(1, smooth_signal[index] - smooth_signal[index - nr_points_slope]) \
                                         / (life_unit - nr_points_slope)

                monotonicity_feature += math.fabs(monotonicity_unit)
            monotonicity_feature /= (nr_units + 0.0)
            self.monotonicity_sensors[feature_name] = monotonicity_feature
            print("Monotonicity", feature_name, self.monotonicity_sensors[feature_name])

    def resample(self, x, new_size, kind='linear'):
        f = scipy.interpolate.interp1d(np.linspace(0, 1, len(x)), x, kind)
        return f(np.linspace(0, 1, new_size))

    def calculate_prognosability(self):
        self.prognosability_sensors = defaultdict(float)
        units = self.testing_units
        for feature_name in self.sensor_names:
            failure_values = []
            bottom_values = []

            for unit in units:
                # obtain signal from dataframe
                smooth_signal = self.dataframe.loc[self.dataframe['Unit'] == unit, feature_name + 'denoised'].values

                failure_values.append(smooth_signal[-1])
                bottom_values.append(math.fabs(smooth_signal[-1] - smooth_signal[0]))

            self.prognosability_sensors[feature_name] = math.exp(- np.std(failure_values) / np.mean(bottom_values))
            print("Prognosability", feature_name, self.prognosability_sensors[feature_name])

    def calculate_trendability(self):
        self.trendability_sensors = defaultdict(float)
        units = self.testing_units
        for feature_name in self.sensor_names:
            trendability_feature = math.inf
            for unit1 in units:
                # obtain signal from dataframe
                smooth_signal1 = self.dataframe.loc[self.dataframe['Unit'] == unit1, feature_name + 'denoised'].values
                life_unit1 = len(smooth_signal1)
                for unit2 in units:
                    # obtain comparison signal from dataframe
                    if unit2 == unit1:
                        continue
                    smooth_signal2 = self.dataframe.loc[self.dataframe['Unit'] == unit2, feature_name + 'denoised'].values
                    life_unit2 = len(smooth_signal2)

                    if life_unit2 < life_unit1:
                        smooth_signal2_2 = self.resample(smooth_signal2, life_unit1)
                        smooth_signal1_2 = smooth_signal1
                    elif life_unit2 > life_unit1:
                        smooth_signal2_2 = smooth_signal2
                        smooth_signal1_2 = self.resample(smooth_signal1, life_unit2)

                    rho, pval = scipy.stats.pearsonr(smooth_signal1_2, smooth_signal2_2)
                    if math.fabs(rho) < trendability_feature:
                        trendability_feature = math.fabs(rho)

            self.trendability_sensors[feature_name] = trendability_feature
            print("Trendability", feature_name, self.trendability_sensors[feature_name])


class ShapExplainerModel:

    def init_data(self, testing_units, predicted_RULs, testing_RULs, data):
        self.testing_units = testing_units
        self.predicted_RULs = predicted_RULs
        self.testing_RULs = testing_RULs
        self.data = data

    def __init__(self):
        pass

    def deserialize(self, name, sensor_names, data_obj):
        df = pd.read_csv(name + '.csv', index_col=False)

        self.testing_units = df['Unit']
        self.predicted_RULs = df['Prediction']
        self.testing_RULs = df['RUL']
        self.sensor_names = sensor_names
        self.shap_values = df.loc[:,sensor_names].values
        self.data = data_obj

    def serialize(self, filename):
        df = self.get_shap_values_df()
        df.to_csv(filename + '.csv', index=False)

    def predict_ESN(self, qc):
        global model
        return np.array(model.esnet.predict(qc))

    def run(self, X_train, nsamples=1000):
        X_train_summary = shap.kmeans(X_train, 10)
        self.sensor_names = X_train.columns
        self.explainer = shap.KernelExplainer(self.predict_ESN, X_train_summary, l1_reg="auto")
        self.shap_values = np.array(self.explainer.shap_values(X_test, nsamples=nsamples))[0]

    def get_shap_values_df(self):
        data = pd.DataFrame(self.shap_values, columns=self.sensor_names)
        data["RUL"] = self.testing_RULs
        data["Prediction"] = self.predicted_RULs
        data["Unit"] = self.testing_units
        return data

    def calculate_errors(self):
        print("MAE Error:", np.mean(np.abs(np.array(self.predicted_RULs) * 400 - np.array(self.testing_RULs) * 400)))

    def check_quality_shap_values(self):
        print("Shap values:", self.shap_values)
        print("Sum shap:", np.abs(self.explainer.expected_value + self.shap_values.sum(axis=1) - np.array(self.predicted_RULs)).max())

    def plot_shap_values_unit(self, data):
        dataframe = data.dataframe
        plt.rcParams.update({'font.size': 20})
        for i in range(len(self.sensor_names)):
            for unit in np.unique(self.testing_units)[:3]:
                RULs = np.array(self.testing_RULs)[self.testing_units == unit][::-1]*400
                #plt.plot(RULs, dataframe.loc[dataframe['Unit'] == unit, self.sensor_names[i] + 'denoised'].values, label='signal')
                plt.scatter(RULs, np.array(self.shap_values[:, i])[self.testing_units == unit], c='black')
                #plt.title(self.sensor_names[i])
                #plt.legend()
                plt.ylabel('SHAP values')
                plt.xlabel('Time (cycles)')
                plt.tight_layout()
                plt.savefig("/Users/marciabaptista/Dropbox/Engineering applications of artificial intelligence/figs/ShapValuesOverTime/" + self.sensor_names[i] + "_2_ESN.png")
                plt.close()
                #plt.show()

    def calculate_prognosability(self):
        self.prognosability_shap_values = defaultdict(float)
        i = 0
        for feature_name in self.sensor_names:
            failure_values = []
            bottom_values = []
            shap_values = self.shap_values[:, i][self.shap_values[:, i] != 0]
            testing_units = np.array(self.testing_units)[self.shap_values[:, i] != 0]

            for unit in testing_units:
                # obtain signal from dataframe
                smooth_signal = shap_values[testing_units == unit]

                failure_values.append(smooth_signal[-1])
                bottom_values.append(math.fabs(smooth_signal[-1] - smooth_signal[0]))

            self.prognosability_shap_values[feature_name] = math.exp(- np.std(failure_values) / np.mean(bottom_values))
            print("Prognosability", feature_name, self.prognosability_shap_values[feature_name])
            i += 1
        l1 = self.prognosability_shap_values.values()
        self.correlate_indicators(l1)

    def calculate_monotonicity(self, nr_points_slope=10, ignore_zeros=True):
        self.monotonicity_shap_values = defaultdict(float)
        for i in range(len(self.sensor_names)):
            monotonicity_shap = 0
            if ignore_zeros:
                shap_values = self.shap_values[:, i][self.shap_values[:, i] != 0]
                testing_units = np.array(self.testing_units)[self.shap_values[:, i] != 0]
            else:
                shap_values = self.shap_values[:, i]
                testing_units = np.array(self.testing_units)
            for unit in np.unique(testing_units):
                monotonicity_shap_unit = 0
                shap_values_unit = shap_values[testing_units == unit]
                for j in range(0, len(shap_values_unit) - nr_points_slope, 1):
                    shap_value = (shap_values_unit)[j]
                    shap_value2 = (shap_values_unit)[j + nr_points_slope]
                    monotonicity_shap_unit += math.copysign(1, shap_value2 - shap_value)/ (len(shap_values_unit) - nr_points_slope)
                monotonicity_shap += math.fabs(monotonicity_shap_unit)
            monotonicity_shap /= (len(np.unique(testing_units)) + 0.0)
            print("Monotonicity shap values", sensor_names[i], monotonicity_shap)
            self.monotonicity_shap_values[self.sensor_names[i]] = monotonicity_shap
        l1 = self.monotonicity_shap_values.values()
        self.correlate_indicators(l1)

    def resample(self, x, new_size, kind='linear'):
        f = scipy.interpolate.interp1d(np.linspace(0, 1, len(x)), x, kind)
        return f(np.linspace(0, 1, new_size))

    def calculate_trendability(self):
        self.trendability_shap_values = defaultdict(float)
        for i in range(len(self.sensor_names)):
            trendability_shap = math.inf
            shap_values = self.shap_values[:, i][self.shap_values[:, i] != 0]
            testing_units = np.array(self.testing_units)[self.shap_values[:, i] != 0]
            unique_testing_units = np.unique(testing_units)

            for unit1 in unique_testing_units:
                shap_values_unit1 = shap_values[testing_units == unit1]

                for unit2 in unique_testing_units:
                    # obtain comparison signal from dataframe
                    if unit2 == unit1:
                        continue
                    shap_values_unit2 = shap_values[testing_units == unit2]

                    if len(shap_values_unit2) < len(shap_values_unit1):
                        smooth_signal2_2 = self.resample(shap_values_unit2, len(shap_values_unit1))
                        smooth_signal1_2 = shap_values_unit1
                    elif len(shap_values_unit2) > len(shap_values_unit1):
                        smooth_signal2_2 = shap_values_unit2
                        smooth_signal1_2 = self.resample(shap_values_unit1, len(shap_values_unit2))

                    rho, pval = scipy.stats.pearsonr(smooth_signal1_2, smooth_signal2_2)
                    if math.fabs(rho) < trendability_shap:
                        trendability_shap = math.fabs(rho)
            print("Trendability shap values", sensor_names[i], trendability_shap)
            self.trendability_shap_values[self.sensor_names[i]] = trendability_shap
        l1 = self.trendability_shap_values.values()
        self.correlate_indicators(l1)

    def calculate_zeros(self):
        self.zero_counter = defaultdict(float)
        for i in range(len(self.sensor_names)):
            self.zero_counter[self.sensor_names[i]] = len(self.shap_values[:, i][self.shap_values[:, i] != 0])
        l1 = self.zero_counter.values()
        self.correlate_indicators(l1)

    def calculate_partial_correlation(self):
        partial_correlations_list = []
        data = pd.DataFrame(self.shap_values, columns=self.sensor_names)
        data["RUL"] = testing_RULs

        for sensor_name1 in self.sensor_names:
            sensor_names3 = []
            for sensor_name2 in self.sensor_names:
                if sensor_name2 != sensor_name1:
                    sensor_names3.append(sensor_name2)
            res = math.fabs(partial_corr(data=data, x=sensor_name1, y='RUL', y_covar=sensor_names3, method='pearson')['r'][0])

            partial_correlations_list.append(res)
        print("Partial correl", partial_correlations_list)
        self.correlate_indicators(partial_correlations_list)

    def correlate_indicators(self, l1):
        l2 = list(self.data.monotonicity_sensors.values())
        l1 = list(l1)
        rho, pval = scipy.stats.pearsonr(l1, l2)
        rho2, pval2 = scipy.stats.spearmanr(l1, l2)
        rho3, pval3 = scipy.stats.kendalltau(l1, l2)
        print("Pearson", "Monotonicity", rho, pval)
        print("Spearman", "Monotonicity", rho2, pval2)
        print("KendallTau", "Monotonicity", rho3, pval3)
        l2 = list(self.data.trendability_sensors.values())
        rho, pval = scipy.stats.pearsonr(l1, l2)
        rho2, pval2 = scipy.stats.spearmanr(l1, l2)
        rho3, pval3 = scipy.stats.kendalltau(l1, l2)
        print("Pearson", "Trendability", rho, pval)
        print("Spearman", "Trendability", rho2, pval2)
        print("KendallTau", "Trendability", rho3, pval3)
        l2 = list(self.data.prognosability_sensors.values())
        rho, pval = scipy.stats.pearsonr(l1, l2)
        rho2, pval2 = scipy.stats.spearmanr(l1, l2)
        rho3, pval3 = scipy.stats.kendalltau(l1, l2)
        print("Pearson", "Prognosability", rho, pval)
        print("Spearman", "Prognosability", rho2, pval2)
        print("KendallTau", "Prognosability", rho3, pval3)


#-----------------------------------------------------
#
#       Main Program
#
#-----------------------------------------------------

debug_raw_sensors = False
debug_baselined_sensors = False
debug_denoised_sensors = False
debug_esn_sensors = False
debug_shap_values_RUL = False
dataset_csv_name = "./data/train_FD001.txt"
new_shap_run = True
results_csv_filename = 'results_shap/FD001_01'

first_run_engines = [43,64,82,21,58,6,94,1,72,22,60,45,69,85,11,77,8,40,
                     49,44,41,97,76,80,17,87,42,93,66,9,74,71,84,15,86,24,
                     28,79,32,78,68,89,5,88,31,2,47,48,53,16,36,56,29,57,
                     37,27,70,62,73,39,61,19,34,10,33,4,26,95,54,18,38,65,
                     91,30,59,63,3,67,90,99,13,46,50,52,25,14,55,51,92,96,
                     98,75,12,23,81,83,35,7,20,100]

program = Program("XAI Prognostics")
if new_shap_run:
    data_obj = CMAPSSData(dataset_csv_name)
else:
    data_obj = CMAPSSData(dataset_csv_name, first_run_engines)

program.init_data(data=data_obj)
program.data.print_general_information()
program.data.set_features_interest(program.data.non_flat_sensor_names)

if debug_raw_sensors:
    program.data.plot_raw_sensor_data()

program.data.baseline(number_regimes=1)

if debug_baselined_sensors:
    program.data.plot_raw_baselined_sensor_data()

program.data.prepare_denoisers()

program.data.denoise()

if debug_denoised_sensors:
    program.data.plot_baselined_denoised_sensor_data()

program.data.add_RUL()

print(program.data.dataframe['RUL'])

program.data.calculate_monotonicity(nr_points_slope=10)
program.data.calculate_trendability()
program.data.calculate_prognosability()

model = EchoStateNetwork()

sensor_names = program.data.sensor_names
training_samples = program.data.get_training_samples()
training_RULs = program.data.get_training_RULs()
testing_samples = program.data.get_testing_samples()
testing_RULs = program.data.get_testing_RULs()
testing_units = program.data.get_testing_units()
configuration_RULs = program.data.get_configuration_RULs()
validation_RULs = program.data.get_validation_RULs()
configuration_samples = program.data.get_configuration_samples()
validation_samples = program.data.get_validation_samples()
testing_time = program.data.get_testing_time()

if new_shap_run:
    model.fit_and_predict(sensor_names, training_samples, training_RULs, configuration_samples, configuration_RULs,
                          validation_samples, validation_RULs, testing_samples, testing_RULs, testing_time)
    model.calculate_errors()

X_train = pd.DataFrame(np.array(training_samples), columns=sensor_names)
X_test = pd.DataFrame(np.array(testing_samples), columns=sensor_names)

if debug_esn_sensors:
    for sensor_name in sensor_names:
        plt.plot(X_test.loc[testing_units == testing_units[0], sensor_name])
        plt.title("Denoised" + sensor_name)
        plt.show()

shap_model = ShapExplainerModel()

if new_shap_run:
    shap_model.init_data(testing_units, model.predicted_RULs, testing_RULs,data_obj)
    shap_model.run(X_train=X_train, nsamples=1000)
    shap_model.check_quality_shap_values()
    #shap_model.serialize(results_csv_filename)
else:
    shap_model.deserialize(results_csv_filename, sensor_names, data_obj)

if debug_shap_values_RUL:
    shap_model.plot_shap_values_unit(program.data)

program.data.print_general_information()

if new_shap_run:
    print("Differential Evolution (DE) Parameters:", model.x)
    model.calculate_errors()

shap_model.calculate_errors()
print("Without zeros: ")
shap_model.calculate_monotonicity(nr_points_slope=10, ignore_zeros=True)
print("With zeros: ")
shap_model.calculate_monotonicity(nr_points_slope=10, ignore_zeros=False)
print("Zero counter: ")
shap_model.calculate_zeros()
print("Trendability of shap values: ")
shap_model.calculate_trendability()
print("Prognosability of shap values: ")
shap_model.calculate_prognosability()