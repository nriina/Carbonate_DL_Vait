import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os

class DataHandler():
    """Class to load data from HDF5 storages in a random and chunckwise manner"""
    def __init__(self, filepath, chunksize=1000):
        self.filepath = filepath
        self.chunksize = chunksize
        self.perception_radius = 10.0
        self.mean_filter_size = 5
        self.training_points = 0
        self.test_points = 0
        self.logger = logging.getLogger(__name__)
        
        if not os.path.exists(filepath):
            raise IOError('File does not exist: {}'.format(filepath))

        # Get the number of rows without loading any data into memory
        with pd.HDFStore(filepath) as store:
            self.nrows = store.get_storer('data').nrows
        self.logger.info('Number of rows: ' + str(self.nrows))

        # Make sure, we can return a chunk with the correct size
#        if chunksize > self.nrows:
#           raise ValueError('Chunksize is to large for the dataset: {} chunks > {} rows'.format(chunksize, self.nrows))

    
    def get_data(self):
        """
           Traffic controller - performs all functions 
           and returns prepared data for the model
        """
        #Get h5 column index values 
        df, laser_cols, goal_cols, cmd_cols = self.get_columns()
        #Create lists based on column values 
        laser, angle, norm, yaw, cmds = self.get_lists(\
                df, laser_cols, goal_cols, cmd_cols)
        #Format lists for model input
        lTrain, lTest, tTrain, tTest, vTrain, vTest = self.format_lists(\
                laser, angle, norm, yaw, cmds)

        return lTrain, lTest, tTrain, tTest, vTrain, vTest
        

    def get_columns(self):
        """
        Read h5 and retrieve column index values

        @return laser, goal, and command column values
        @rtype Tuple
        """

        #Retrieve hdf file
        df = pd.read_hdf(self.filepath, 'data')

        #Create new translational/rotational cmd columns from rolling mean
        df['filtered_linear'] = df['linear_x'].rolling(window=\
                self.mean_filter_size, center=True).mean().fillna(df['linear_x'])
        df['filtered_angular'] = df['angular_z'].rolling(window=\
                self.mean_filter_size, center=True).mean().fillna(df['angular_z'])
        self.logger.info('rolling mean calculated')

        #Retrieve column index values for lidar, target, and labeled data (TvRv)
        laser_columns = list()
        goal_columns = list()
        cmd_columns = list()
        for j,column in enumerate(df.columns):
            if column.split('_')[0] in ['laser']:
                laser_columns.append(j)
            if column.split('_')[0] in ['target'] and not column.split('_')[1] == 'id':
                goal_columns.append(j)
            if column in ['filtered_linear','filtered_angular']:
                cmd_columns.append(j)

        
        #Only use the center n_scans elements as input ((not my comment))
        n_scans = 1080
        drop_n_elements = (len(laser_columns) - n_scans) // 2        
        if drop_n_elements < 0:
                raise ValueError('Number of scans is to small: {} < {}'
                        .format(len(laser_columns), n_scans))
        elif drop_n_elements > 0:
                laser_columns = laser_columns[drop_n_elements:-drop_n_elements]
                
        #error correct list length
        if len(laser_columns) == n_scans+1:
                laser_columns = laser_columns[0:-1]
        
        return df, laser_columns, goal_columns, cmd_columns


    def get_lists(self, df, laser_columns, goal_columns, cmd_columns):
        """
           Uses column index values to create lists of necessary data
           @return laser, angle, norm, yaw, cmds
        """
        #Adjusts lidar value ranges to fit real lidar range
        laser = np.minimum(df.iloc[:,laser_columns].values, self.perception_radius)

        #Retrieve goal list (x, y, yaw)
        goal = df.iloc[:,goal_columns].values

        #Retrieve cmd velocities
        cmds = df.iloc[:,cmd_columns].values

        #Calulate angle of target
        angle = np.arctan2(goal[:,1],goal[:,0])
        
        #L2 normalization of the x column - believed to be distance to goal
        norm = np.minimum(np.linalg.norm(goal[:,0:2], ord=2, axis=1),\
                self.perception_radius)

        yaw = goal[:,2]

        return laser, angle, norm, yaw, cmds


    def format_lists(self, laser, angle, norm, yaw, cmds):    
        """
           Formats the lists as input to the model
           Randomly partitions into train/test lists
           @return Train - Test | laser, target, cmd
        """
        #Combine target information into a single list
        targets = []
        point = []
        for i in range(len(angle)):
            point = []
            point.append(angle[i])
            point.append(norm[i])
            point.append(yaw[i])
            targets.append(point)
        
        #Partition data into training and testing sets
        lTrain, lTest, tTrain, tTest, vTrain, vTest = train_test_split(\
                laser, targets, cmds, test_size=0.2, random_state=42)
        self.training_points = len(lTrain)
        self.testing_points = len(lTest)

        #Expand dimensionality of data
        lTrain = self.expand_dims(lTrain)
        lTest = self.expand_dims(lTest)
        tTrain = np.array(tTrain)
        tTest = np.array(tTest)
        vTrain = np.array(vTrain)
        vTest = np.array(vTest)

        return lTrain, lTest, tTrain, tTest, vTrain, vTest

    
    def expand_dims(self, lst):
        """ Expands dimensionality twice, resulting in 4d 
            array of shape (# of datapoints, 1, length of datapoint, 1) 
        """
        lst = np.expand_dims(lst, axis=2)
        lst = np.expand_dims(lst, axis=1)
        return lst


    def get_training_points(self):
        """ Number of datapoints for training
        """
        return self.training_points

    def get_testing_points(self):
        """ Number of datapoints for testing
        """
        return self.testing_points
