import numpy as np
import matplotlib.pyplot as plt

def get_beat_features(window_pulse_data, debug = False):
    """
    Calculate beat-based features using the provided window data and window pulse data.

    Args:
    - window_data: Data related to the window.
    - window_pulse_data: Pulse data within the window.
    - visualise: Flag to enable visualisation (default: False).
    - debug: Flag to enable debug output (default: False).

    Returns:
    - Tuple of beat-based features: (num_beats, median_ibi, std_ibi, cv_ibi).
    - If no peaks are found, returns (NaN, NaN, NaN, NaN).

    Error Handling:
    - If the window_pulse_data is empty, the function returns NaN for all outputs.
    """

    # Check that window_pulse_data is not empty
    if window_pulse_data:
        peaks = [window_pulse_data[key]["peak"] for key in window_pulse_data]
        peaks.sort()

        ###################################
        # Calculating beat based features #
        ###################################

        # Calculate the number of pulses per window
        num_beats = len(peaks)

        # Calculate the interbeat interval
        ibi = np.diff(peaks)

        # Calculate the median interbeat interval
        median_ibi = np.median(ibi)

        # Calculate the standard deviation of the interbeat interval
        std_ibi = np.std(ibi)

        """Calculate the coefficient of variation of the interbeat interval.
        By using the median interbeat interval in both the numerator and denominator, 
        the relative variability of the interbeat intervals based on the central tendency 
        provided by the median is being accessed. This approach takes into account the 
        spread of the data around the median (instead of the mean), making it less influenced by outliers or extreme values
        providing a measure of relative variability that is more resistant to the presence of outliers."""
        cv_ibi = std_ibi/median_ibi

        if debug:
            print("Number of beats: " + str(num_beats))
            print("Median IBI: " + str(median_ibi))
            print("Standard deviation of IBI: " + str(std_ibi))
            print("Coefficient of variation of IBI: " + str(cv_ibi))
                            
        # Return the beat based features
        beat_features = {}
        beat_features["num_beats"] = num_beats
        beat_features["median_ibi"] = median_ibi
        beat_features["std_ibi"] = std_ibi
        return beat_features
    else:
        # Return empty dictionary if no peaks are found
        return {}
        
    