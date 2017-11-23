import pandas as pd


class DataHelper(object):

    attributes = ['K',
                  'Na',
                  'CL',
                  'HCO3',
                  'Endotoxin',
                  'Aniongap',
                  'PLA2',
                  'SDH',
                  'GLDH',
                  'TPP',
                  'Breath rate',
                  'PCV',
                  'Pulse rate',
                  'Fibrinogen',
                  'Dimer',
                  'FibPerDim'
                  ]
    target_class = 'isColic'
    train_file = 'horseTrain.txt'
    test_file = 'horseTest.txt'

    @classmethod
    def get_attributes(cls):
        return DataHelper.attributes

    @classmethod
    def get_target_class(cls):
        return DataHelper.target_class

    @classmethod
    def get_train_x(cls):
        columns = DataHelper.attributes + [DataHelper.target_class]
        df = pd.read_csv(DataHelper.train_file, header=None, names=columns)
        df.drop([DataHelper.target_class], 1, inplace=True)
        return df

    @classmethod
    def get_train_data(cls):
        columns = DataHelper.attributes + [DataHelper.target_class]
        df = pd.read_csv(DataHelper.train_file, header=None, names=columns)
        df[DataHelper.target_class] = df[DataHelper.target_class].str.rstrip('.')
        return df

    @classmethod
    def get_train_y(cls):
        columns = DataHelper.attributes + [DataHelper.target_class]
        df = pd.read_csv(DataHelper.train_file, header=None, names=columns)
        df.drop(DataHelper.attributes, 1, inplace=True)
        df[DataHelper.target_class] = df[DataHelper.target_class].str.rstrip('.')
        return df

    @classmethod
    def get_test_x(cls):
        columns = DataHelper.attributes + [DataHelper.target_class]
        df = pd.read_csv(DataHelper.test_file, header=None, names=columns)
        df.drop([DataHelper.target_class], 1, inplace=True)
        return df

    @classmethod
    def get_test_data(cls):
        columns = DataHelper.attributes + [DataHelper.target_class]
        df = pd.read_csv(DataHelper.test_file, header=None, names=columns)
        df[DataHelper.target_class] = df[DataHelper.target_class].str.rstrip('.')
        return df

    @classmethod
    def get_test_y(cls):
        columns = DataHelper.attributes + [DataHelper.target_class]
        df = pd.read_csv(DataHelper.test_file, header=None, names=columns)
        df.drop(DataHelper.attributes, 1, inplace=True)
        df[DataHelper.target_class] = df[DataHelper.target_class].str.rstrip('.')
        return df


if __name__ == '__main__':
    trainX = DataHelper.get_train_x()
    trainY = DataHelper.get_train_y()
    trainData = DataHelper.get_train_data()
    testX = DataHelper.get_test_x()
    testY = DataHelper.get_test_y()
    testData = DataHelper.get_test_data()
