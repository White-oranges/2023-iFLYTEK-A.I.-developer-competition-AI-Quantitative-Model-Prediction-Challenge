import lightgbm as lgb


class Model:
    def __init__(self):
        self.m = lgb.LGBMClassifier(num_leaves=31, reg_alpha=0.4, reg_lambda=0.25, max_depth=-1,
                                    learning_rate=0.05, min_child_samples=8, random_state=159,
                                    n_estimators=390, subsample=0.9, colsample_bytree=0.7,
                                    class_weight={0: 0.435, 1: 0.13, 2: 0.435})
