import numpy as np

def get_second_der_ratio(pulse_data):
    data = pulse_data["raw_pulse_data"]
    peak = pulse_data["Relative_peak"]

    second_derivative_ratio = 0

    if peak:
        second_derivative = np.diff(np.diff(data))
        max_value = max(second_derivative)
        min_value = min(second_derivative)
        second_derivative_ratio = max_value/min_value

        return float(abs(second_derivative_ratio))
    else:
        return np.NaN