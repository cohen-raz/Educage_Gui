import pandas as pd
from Constants import *
import os

class Preprocessor:
    def __init__(self, dataPath=None):
        if dataPath:
            self._dataPath = dataPath
            self.readData()
            self.toDateTime()
            self.saveData()

        else:
            default_data_path = self._get_default_data_path()
            if default_data_path:
                self._df = pd.read_pickle(default_data_path)
                self._default_data_loaded = True
            else:
                self._df = None
                self._default_data_loaded = False

        self._miceGroup = None
        self._cleanDF = None

    def is_default_data_loaded(self):
        return self._default_data_loaded

    def _get_default_data_path(self):
        # look for .pkl file in current working directory
        files = os.listdir(os.getcwd())
        for fname in files:
            if fname.endswith(PICKL_EXTENSION):
                return fname
        return False

    def readData(self):
        self._df = pd.read_csv(self._dataPath)

    def saveData(self, pickle_path=None):
        if not pickle_path:
            self._df.to_pickle(DEFAULT_DF_PCKL_PATH)
        else:
            self._df.to_pickle(pickle_path)

    def getDF(self):
        return self._df

    def getNumOfMice(self):
        self._miceList = self.getUniqeList(MOUSE_NUM_COL_NAME)
        return len(self._miceList)

    def getUniqeList(self, colName):
        if self._df is None:
            return []
        return self._df[colName].unique().tolist()

    def groupByCol(self, colName):
        return self._df.groupby(colName)

    def _groupByMice(self):
        if not self._miceGroup:
            self._miceGroup = self.groupByCol(MOUSE_NUM_COL_NAME)

    def getMouseTableByValue(self, value):
        self._groupByMice()
        return self._miceGroup.get_group(value)

    def removeCol(self, colName):
        del self._df[colName]

    def basicClean(self):
        if not self._cleanDF:
            self._cleanDF = self._df[
                [MOUSE_NUM_COL_NAME, MOUSE_ID, LEVEL, SCORE, FREQ, TIME]]
        return self._cleanDF


    def toDateTime(self):
        self._df[TIME] = pd.to_datetime(self._df[TIME])

    def set_data(self, df):
        self._df = df

    def remove_bad_behaviour(self, df, update_self=False):
        df = df.drop(df[df[SCORE] > MAX_SCORE].index)
        if update_self:
            self._df = df
        return df


