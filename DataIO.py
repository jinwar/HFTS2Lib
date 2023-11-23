import h5py
import datetime
import pandas as pd

def read_h5(filename):
    """
    Reads a 2D matrix and a 1D datetime series from an HDF5 file and converts the time to Python datetime objects.

    Parameters:
    filename (str): Path to the HDF5 file.

    Returns:
    tuple: A tuple containing the following elements:
        - 2D numpy array: The sensor data.
        - 1D list: The timestamps corresponding to the sensor data, converted to Python datetime objects.
    """
    with h5py.File(filename, 'r') as file:
        # Read the 2D matrix ('RawData') and 1D series ('RawDataTime')
        raw_data = file['Acquisition/Raw[0]/RawData'][:]
        raw_data_time = file['Acquisition/Raw[0]/RawDataTime'][:]

    # Convert 'RawDataTime' to Python datetime objects
    raw_data_time = [datetime.datetime.fromtimestamp(ts / 1e6) for ts in raw_data_time]

    return raw_data, raw_data_time

# Example usage
# file_path = 'path_to_your_file.h5'
# raw_data, raw_data_time = read_sensor_data_as_datetime(file_path)
# print(raw_data.shape, len(raw_data_time))
# print(raw_data_time[0]) # to see the format of the datetime object

def read_depth_table(filename):
    depth_table = pd.read_csv(filename,skiprows=17)
    mds = depth_table.values[:,1]
    return mds

def read_pump_curves(file_path):
    """
    Reads and outputs the "Data" sub-datasets for 'Clean rate', 'Slurry rate', 'Bottomhole concentration', 
    and 'UTC_TIME' under the 'Filtered Pump Data#2' group from an HDF5 file.

    :param file_path: Path to the HDF5 file
    :return: Dictionary containing the "Data" sub-datasets for the specified datasets
    """
    group_name = 'Curves/Filtered Pump Data#2'
    dataset_names = ['Surface pressure', 'Slurry rate', 'Bottomhole concentration', 'UTC_TIME']
    data = {}

    with h5py.File(file_path, 'r') as file:
        for name in dataset_names:
            try:
                # Reading the "Data" sub-dataset under each specified dataset
                data_path = f"{group_name}/{name}/Data"
                data[name] = file[data_path][:]  # Extracting the entire "Data" sub-dataset
            except Exception as e:
                data[name] = f"Error reading dataset: {e}"
    
    df = pd.DataFrame()
    df['Time'] = [datetime.datetime.fromtimestamp(int(t/1e3)) for t in data['UTC_TIME']]
    for key in dataset_names[:-1]:
        df[key] = data[key].flatten()

    return df
