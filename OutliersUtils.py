import numpy as np


def IQROutliers(data, iqr_multiplier=1.5):
    q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q75 - q25
    cut_off = iqr * iqr_multiplier
    lower, upper = q25 - cut_off, q75 + cut_off
    return lower, upper


def SDOutliers(data, deviations=3):
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * deviations
    lower, upper = data_mean - cut_off, data_mean + cut_off
    return lower, upper
