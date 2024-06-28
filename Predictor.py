import joblib
import pandas as pd


class Predictor:
    def __init__(self):
        self.model = joblib.load("./mmpc/model.pkl")

    def predict(self, x):
        y_hat = self.preprocess(x)
        y = y_hat
        #y = list(y)
        return y

    def preprocess(self, x):
        columns_name = ['date', 'time', 'sym', 'n_close', 'amount_delta', 'n_midprice',
                        'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3',
                        'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1',
                        'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4',
                        'n_ask5', 'n_asize5']
        start_feature = ['date', 'time', 'n_close', 'amount_delta', 'n_midprice', 'date_time',
                         'n_bsize1', 'n_bsize2', 'n_bsize3', 'n_bsize4', 'n_bsize5',
                         'n_asize1', 'n_asize2', 'n_asize3', 'n_asize4', 'n_asize5',
                         'n_bid1', 'n_bid2', 'n_bid3',
                         'n_ask1', 'n_ask2', 'n_ask3']
        shift_feature = ['n_close', 'amount_delta', 'n_midprice',
                         'n_bsize1', 'n_bsize2', 'n_bsize3', 'n_bsize4', 'n_bsize5',
                         'n_asize1', 'n_asize2', 'n_asize3', 'n_asize4', 'n_asize5']
        shift_feature2 = ['n_bsize1', 'n_bsize2',
                          'n_asize1', 'n_asize2', ]
        shift_feature3 = ['n_close', 'n_midprice']
        feature = start_feature
        for name in shift_feature:
            feature.append('mean_{}'.format(name))
            feature.append('mean2_{}'.format(name))
        for name in shift_feature2:
            feature.append('var3_{}'.format(name))
        for name in shift_feature3:
            feature.append('minus_{}'.format(name))
            feature.append('box_{}'.format(name))
            feature.append('mean3_{}'.format(name))
        df = pd.DataFrame(x)
        df.columns = columns_name
        df = df.reset_index(drop=True)
        df['time'] = df['time'].str.replace(':', '')
        df['time'] = df['time'].str[-6:]
        df['time'] = df['time'].astype('int')
        print(df['time'])
        df['date_time'] = 1000000 * df['date'] + df['time']
        for col in shift_feature:
            df['mean_{}'.format(col)] = df[col]
            df['mean2_{}'.format(col)] = df[col]
            df.loc[99, 'mean_{}'.format(col)] -= df.loc[99 - 2:, col].mean()
            df.loc[99, 'mean2_{}'.format(col)] -= df.loc[99 - 7:, col].mean()
        for col in shift_feature2:
            df.loc[99, 'var3_{}'.format(col)] = df.loc[99 - 11:, col].var()
        for col in shift_feature3:
            df.loc[99, 'box_{}'.format(col)] = df.loc[99-19:, col].mean() \
                                              - df.loc[99-95:, col].mean()

            df['minus_{}'.format(col)] = df[col].rolling(window=40).max()
            df['minus_{}'.format(col)] -= df[col].rolling(window=40).min()
            df.loc[99, 'mean3_{}'.format(col)] = df.loc[99-44:, 'minus_{}'.format(col)].mean()
        df = df[feature]
        output = []
        output.append(int(self.model.predict(df.tail(1))))
        output.append(int(self.model.predict(df.tail(1))))
        output.append(int(self.model.predict(df.tail(1))))
        output.append(int(self.model.predict(df.tail(1))))
        output.append(int(self.model.predict(df.tail(1))))
        return output

