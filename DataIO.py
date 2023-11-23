import h5py
import os
import datetime
import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm
from JIN_pylib import Data2D_XT
import xml.etree.ElementTree as ET
from dateutil import parser 

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

def get_data_folder_info(folder_path):
    files = glob(folder_path+'/*.h5')
    files = np.sort(files)
    timestamps = np.array([parser.parse(f[-21:-4]) for f in files])
    return files,timestamps

def read_data_folder(folder_path,start_time=None,end_time=None):
    files,timestamps = get_data_folder_info(folder_path)

    # Check if start_time or end_time is a string and convert to datetime
    if isinstance(start_time, str):
        start_time = parser.parse(start_time)
    if isinstance(end_time, str):
        end_time = parser.parse(end_time)

    if start_time is None:
        start_time = timestamps[0]
    if end_time is None:
        end_time = timestamps[-1]
    
    ind = np.where((timestamps>=start_time) & (timestamps<=end_time))[0]
    datalist = []
    for filename in tqdm(files[ind]):
        data, timestamps = read_h5(filename)
        DASdata = Data2D_XT.Data2D()
        DASdata.data = data
        DASdata.set_time_from_datetime(timestamps)
        DASdata.chans = np.arange(data.shape[0])
        datalist.append(DASdata)
    merge_data = Data2D_XT.merge_data2D(datalist)
    return merge_data


def read_fiber_xml(filename):
    """
    Extracts specified information from an XML file without considering XML namespaces.

    :param filename: The path to the XML file.
    :return: A dictionary containing the extracted information.
    """

    # Load and parse the XML file
    tree = ET.parse(filename)
    root = tree.getroot()

    # Function to remove namespace from tags
    def remove_namespace(elem):
        for elem in elem.iter():
            elem.tag = elem.tag.split('}', 1)[-1]  # Remove the namespace URI

    # Remove namespace from all elements
    remove_namespace(root)

    # Extracting data from "acquisitionsetup"
    acquisitionsetup = root.find('.//acquisitionSetup')
    first_channel = acquisitionsetup.find('firstChannel').text if acquisitionsetup.find('firstChannel') is not None else "Not found"
    last_channel = acquisitionsetup.find('lastChannel').text if acquisitionsetup.find('lastChannel') is not None else "Not found"
    spatial_resolution = acquisitionsetup.find('spatialResolution').text if acquisitionsetup.find('spatialResolution') is not None else "Not found"

    # Extracting calibration points from "settings"
    settings = root.find('.//settings')
    calibration_points = []
    for point in settings.findall('.//calibrationPoints'):
        optical_distance = point.find('opticalDistance').text if point.find('opticalDistance') is not None else "Not found"
        physical_distance = point.find('physicalDistance').text if point.find('physicalDistance') is not None else "Not found"
        calibration_points.append({'optical_distance': optical_distance, 'physical_distance': physical_distance})
    
    chans = np.arange(int(first_channel),int(last_channel)+1)
    fiber_length = chans*float(spatial_resolution)

    from scipy.interpolate import interp1d
    optical_distance = np.array([float(p['optical_distance']) for p in calibration_points])
    physical_distance = np.array([float(p['physical_distance']) for p in calibration_points])
    f = interp1d(optical_distance, physical_distance, kind='linear', fill_value='extrapolate')
    daxis = f(fiber_length)


    # Return the extracted data as a dictionary
    return chans,daxis


# Example usage
# file_path = 'path_to_your_file.h5'
# raw_data, raw_data_time = read_sensor_data_as_datetime(file_path)
# print(raw_data.shape, len(raw_data_time))
# print(raw_data_time[0]) # to see the format of the datetime object

def read_depth_table(filename):
    depth_table = pd.read_csv(filename,skiprows=17)
    mds = depth_table.values[:,1]
    return mds

import fnmatch

def _find_group(file, pattern, path=''):
    """Recursively search for a group that matches the pattern."""
    for key in file.keys():
        new_path = f'{path}/{key}'
        if fnmatch.fnmatch(key.lower(), pattern):
            return new_path
        if isinstance(file[key], h5py.Group):  # If this item is a group, search its keys
            result = _find_group(file[key], pattern, new_path)
            if result is not None:
                return result
    return None

def read_pump_curves(file_path):
    dataset_names = ['surface pressure', 'slurry rate', 'bottomhole concentration', 'utc_time']
    data = {}

    with h5py.File(file_path, 'r') as file:
        # Find the group that contains 'Filtered Pump Data'
        group_name = _find_group(file, '*filtered pump data*')

        if group_name is None:
            raise ValueError("No group found that matches the pattern '*filtered pump data*'")

        for name in dataset_names:
            try:
                # Find the dataset that matches the name, case-insensitive
                dataset_name = next(d for d in file[group_name] if d.lower() == name)

                # Reading the "Data" sub-dataset under each specified dataset
                data_path = f"{group_name}/{dataset_name}/Data"
                data[name] = file[data_path][:]  # Extracting the entire "Data" sub-dataset
            except Exception as e:
                data[name] = f"Error reading dataset: {e}"
    
    df = pd.DataFrame()
    df['Time'] = [datetime.datetime.fromtimestamp(int(t/1e3)) for t in data['utc_time']]
    for key in dataset_names[:-1]:
        df[key] = data[key].flatten()

    return df

class DxSProject:

    def __init__(self,project_path) -> None:
        self.project_path = project_path
        self.read_fiber()
        self.current_dataset = None
        self.min_md = None
        self.max_md = None
    
    def read_fiber(self):
        filepath = self.project_path + '/Well/Fibres.xml'
        chans,daxis = read_fiber_xml(filepath)
        self.chans = chans
        self.daxis = daxis
    
    def read_pump_curves(self):
        filepath = self.project_path + '/Input/Curves/curves.h5'
        df = read_pump_curves(filepath)
        return df
    
    def set_current_dataset(self,dataset_name):
        self.current_dataset = dataset_name
    
    def set_depth_range(self,min_md,max_md):
        self.min_md = min_md
        self.max_md = max_md
    
    def read_dataset(self,start_time=None,end_time=None):
        dataset_name = self.current_dataset
        if dataset_name is None:
            print('Please set the current dataset')
            return
        datapath = self.project_path + '/Input/' + dataset_name
        data = read_data_folder(datapath,start_time=start_time,end_time=end_time)
        data.chans = self.chans
        data.daxis = self.daxis
        data.select_depth(self.min_md,self.max_md)
        return data

    def print_dataset_list(self):
        # Construct the path to the 'Input' directory
        input_path = os.path.join(self.project_path, 'Input')

        # Get a list of all items in the 'Input' directory
        items = os.listdir(input_path)

        # Filter the list to include only subdirectories
        subfolders = [item for item in items if os.path.isdir(os.path.join(input_path, item))]

        print(subfolders)
        return subfolders
    

    def print_dataset_info(self, dataset_name):
        # Construct the path to the dataset directory
        dataset_path = os.path.join(self.project_path, 'Input', dataset_name)

        # Get a list of all items in the dataset directory
        items = os.listdir(dataset_path)

        # Filter the list to include only HDF5 files
        files = [item for item in items if item.endswith('.h5')]

        # Calculate the total size of the folder
        total_size = sum(os.path.getsize(os.path.join(dataset_path, f)) for f in files)

        # Print the number of files, the first and last file name, and the total size
        print(f"Dataset: {dataset_name}")
        print(f"Number of files: {len(files)}")
        print(f"First file: {files[0]}")
        print(f"Last file: {files[-1]}")
        print(f"Total size: {total_size/1024/1024:.1f} MB")