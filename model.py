import gc
import warnings

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

warnings.filterwarnings('ignore')

seed = 2020


def run_lgb(df_train, df_test):
    ycol = 'target'
    feature_names = list(
        filter(lambda x: x not in [ycol, 'date', 'uin'], df_train.columns))

    model = lgb.LGBMClassifier(num_leaves=64,
                               max_depth=10,
                               learning_rate=0.01,
                               n_estimators=10000000,
                               subsample=0.8,
                               feature_fraction=0.8,
                               reg_alpha=0.5,
                               reg_lambda=0.5,
                               random_state=seed,
                               metric=None)

    oof = []
    prediction = df_test[['uin']]
    prediction['target'] = 0
    df_importance_list = []

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train[feature_names], df_train[ycol])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][ycol]

        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][ycol]

        print('\nFold_{} Training ================================\n'.format(
            fold_id + 1))

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=500,
                              eval_metric='auc',
                              early_stopping_rounds=100)

        pred_val = lgb_model.predict_proba(
            X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        df_oof = df_train.iloc[val_idx][['uin', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = lgb_model.predict_proba(
            df_test[feature_names],
            num_iteration=lgb_model.best_iteration_)[:, 1]
        prediction['target'] += pred_test / 5

        df_importance = pd.DataFrame({
            'column':
            feature_names,
            'importance':
            lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
        gc.collect()

    return oof, prediction, df_importance_list


def run_model(res_file='studio.txt'):
    df_feature = pd.read_pickle('feature.plk')

    for f in tqdm(df_feature.select_dtypes('object')):
        if f in ['uin', 'date']:
            continue
        lbl = LabelEncoder()
        df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

    df_test = df_feature[df_feature['target'].isnull()].copy()
    df_train = df_feature[df_feature['target'].notnull()].copy()

    oof, prediction, df_importance_list = run_lgb(df_train, df_test)

    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'column'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    df_importance

    old_ratio = df_train[df_train['target'] == 1].shape[0] / df_train.shape[0]
    old_ratio = round(old_ratio, 4)
    print('正样本比例', old_ratio)

    df_oof = pd.concat(oof)
    print(df_oof.shape, prediction.shape)

    df_oof.sort_values(['pred'], inplace=True, ascending=False)
    df_oof.reset_index(drop=True, inplace=True)
    df_oof['pred_label'] = 0
    df_oof.loc[:int(prediction.shape[0] * old_ratio), 'pred_label'] = 1

    P = precision_score(df_oof['target'], df_oof['pred_label'])
    R = recall_score(df_oof['target'], df_oof['pred_label'])
    oof_score = 4 * P * R / (P + 3 * R)
    auc = roc_auc_score(df_oof['target'], df_oof['pred_label'])
    print('伪标签之前oof得分：', oof_score, P, R)

    # 伪标签

    old_oof = df_oof['uin'].values.tolist()

    postive_uin = prediction[
        prediction['target'] > 0.99]['uin'].values.tolist()
    df_train_fake = df_test[df_test['uin'].isin(postive_uin)]
    df_train_fake['target'] = 1
    df_train_fake.shape

    df_train_fake = pd.concat([df_train, df_train_fake], sort=False)
    ratio = df_train_fake[df_train_fake['target'] ==
                          1].shape[0] / df_train_fake.shape[0]
    ratio = round(ratio, 4)
    print('正样本比例', ratio)

    fake_oof, prediction, _ = run_lgb(df_train_fake, df_test)

    df_oof = pd.concat(fake_oof)

    df_oof.sort_values(['pred'], inplace=True, ascending=False)
    df_oof.reset_index(drop=True, inplace=True)
    df_oof['pred_label'] = 0
    df_oof.loc[:int(prediction.shape[0] * ratio), 'pred_label'] = 1

    print(df_oof.shape)
    df_oof = df_oof[df_oof['uin'].isin(old_oof)]
    df_oof.to_csv('lgb_oof.csv', index=False)
    print(df_oof.shape)

    P = precision_score(df_oof['target'], df_oof['pred_label'])
    R = recall_score(df_oof['target'], df_oof['pred_label'])
    oof_score = 4 * P * R / (P + 3 * R)
    auc = roc_auc_score(df_oof['target'], df_oof['pred_label'])
    print('伪标签之后oof得分：', oof_score, P, R, auc)

    prediction.sort_values(['target'], inplace=True, ascending=False)
    prediction_pos = prediction.head(int(prediction.shape[0] * old_ratio))
    prediction_pos.shape

    with open(res_file, 'w') as f:
        for id in prediction_pos['uin'].unique():
            f.write(id + '\n')
