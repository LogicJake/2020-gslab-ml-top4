import gc
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from gensim.models import Word2Vec

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

warnings.filterwarnings('ignore')
seed = 2020


def statis_feat(df_know, df_unknow):
    for group_by in [['chat_cnt'], ['clientip_3']]:
        df_temp = df_know.groupby(group_by)['target'].mean().reset_index()
        df_temp.columns = group_by + ['{}_ratio'.format('_'.join(group_by))]
        df_unknow = df_unknow.merge(df_temp, on=group_by, how='left')

    return df_unknow


def emb(df, f1, f2):
    emb_size = 16
    print(
        '====================================== {} {} ======================================'
        .format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences,
                     size=emb_size,
                     window=5,
                     min_count=5,
                     sg=0,
                     hs=1,
                     seed=seed)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model:
                vec.append(model[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)

    df_emb = pd.DataFrame(emb_matrix)
    df_emb.columns = [
        '{}_{}_emb_{}'.format(f1, f2, i) for i in range(emb_size)
    ]

    tmp = pd.concat([tmp, df_emb], axis=1)

    del model, emb_matrix, sentences
    return tmp


def run_feature(data_dir):
    login_train = pd.read_csv(os.path.join(data_dir, 'role_login',
                                           '20200301.txt'),
                              sep='|',
                              header=None)
    login_test = pd.read_csv(os.path.join(data_dir, 'role_login',
                                          '20200305.txt'),
                             sep='|',
                             header=None)

    login = pd.concat([login_train, login_test], sort=False)
    login.columns = [
        'dteventtime', 'platid', 'areaid', 'worldid', 'uin', 'roleid',
        'rolename', 'job', 'rolelevel', 'power', 'friendsnum', 'network',
        'clientip', 'deviceid'
    ]

    login.head()

    logout_train = pd.read_csv(os.path.join(data_dir, 'role_logout',
                                            '20200301.txt'),
                               sep='|',
                               header=None)
    logout_test = pd.read_csv(os.path.join(data_dir, 'role_logout',
                                           '20200305.txt'),
                              sep='|',
                              header=None)

    logout = pd.concat([logout_train, logout_test], sort=False)
    logout.columns = [
        'dteventtime', 'platid', 'areaid', 'worldid', 'uin', 'roleid',
        'rolename', 'job', 'rolelevel', 'power', 'friendsnum', 'network',
        'clientip', 'deviceid', 'onlinetime'
    ]

    logout.head()

    create_train = pd.read_csv(os.path.join(data_dir, 'role_create',
                                            '20200301.txt'),
                               sep='|',
                               header=None)
    create_test = pd.read_csv(os.path.join(data_dir, 'role_create',
                                           '20200305.txt'),
                              sep='|',
                              header=None)

    create = pd.concat([create_train, create_test], sort=False)
    create.columns = [
        'dteventtime', 'platid', 'areaid', 'worldid', 'uin', 'roleid',
        'rolename', 'job', 'regchannel', 'network', 'clientip', 'deviceid'
    ]

    create.head()

    chat_train = pd.read_csv(os.path.join(data_dir, 'uin_chat',
                                          '20200301.txt'),
                             sep='|',
                             header=None)
    chat_train['date'] = '2020-03-01'
    chat_test = pd.read_csv(os.path.join(data_dir, 'uin_chat', '20200305.txt'),
                            sep='|',
                            header=None)
    chat_test['date'] = '2020-03-05'

    chat = pd.concat([chat_train, chat_test], sort=False)
    chat.columns = ['uin', 'chat_cnt', 'date']

    chat.head()

    role_moneyflow_train = pd.read_csv(os.path.join(data_dir, 'role_moneyflow',
                                                    '20200301.txt'),
                                       sep='|',
                                       header=None)
    role_moneyflow_test = pd.read_csv(os.path.join(data_dir, 'role_moneyflow',
                                                   '20200305.txt'),
                                      sep='|',
                                      header=None)
    role_moneyflow = pd.concat([role_moneyflow_train, role_moneyflow_test],
                               sort=False)
    role_moneyflow.columns = [
        'dteventtime', 'worldid', 'uin', 'roleid', 'rolelevel', 'iMoneyType',
        'iMoney', 'AfterMoney', 'AddOrReduce', 'Reason', 'SubReason'
    ]
    role_moneyflow['dteventtime'] = pd.to_datetime(
        role_moneyflow['dteventtime'])

    role_moneyflow.head()

    role_itemflow_train = pd.read_csv(os.path.join(data_dir, 'role_itemflow',
                                                   '20200301.txt'),
                                      sep='|',
                                      header=None)
    role_itemflow_test = pd.read_csv(os.path.join(data_dir, 'role_itemflow',
                                                  '20200305.txt'),
                                     sep='|',
                                     header=None)
    role_itemflow = pd.concat([role_itemflow_train, role_itemflow_test],
                              sort=False)
    role_itemflow.columns = [
        'dteventtime', 'worldid', 'uin', 'roleid', 'rolelevel', 'Itemtype',
        'Itemid', 'Count', 'Aftercount', 'Addorreduce', 'Reason', 'SubReason'
    ]
    role_itemflow['dteventtime'] = pd.to_datetime(role_itemflow['dteventtime'])

    role_itemflow.head()

    label_1 = pd.read_csv(os.path.join(data_dir, 'label_black',
                                       '20200301.txt'),
                          sep='|',
                          header=None,
                          float_precision='round_trip')
    label_1.columns = ['uin']
    label_1['target'] = 1

    label_0 = pd.read_csv(os.path.join(data_dir, 'label_white',
                                       '20200301.txt'),
                          sep='|',
                          header=None,
                          float_precision='round_trip')
    label_0.columns = ['uin']
    label_0['target'] = 0

    df_train = pd.concat([label_0, label_1], sort=False)
    df_train['date'] = '2020-03-01'

    df_train.head()

    login['dteventtime'] = pd.to_datetime(login['dteventtime'])
    logout['dteventtime'] = pd.to_datetime(logout['dteventtime'])

    login['date'] = login['dteventtime'].dt.date
    logout['date'] = logout['dteventtime'].dt.date

    login['date'] = login['date'].astype('str')
    logout['date'] = logout['date'].astype('str')

    df_test = pd.DataFrame()
    test_uin = []
    test_uin += login[login['date'] == '2020-03-05']['uin'].values.tolist()
    test_uin += logout[logout['date'] == '2020-03-05']['uin'].values.tolist()
    test_uin = list(set(test_uin))
    test_uin.sort()

    df_test['uin'] = test_uin
    df_test['target'] = np.nan
    df_test['date'] = '2020-03-05'
    print(df_test.shape)

    df_feature = pd.concat([df_train, df_test], sort=False)
    df_feature.head()

    # 特征工程

    df_feature = df_feature.merge(chat, how='left')
    df_feature['chat_cnt'] = df_feature['chat_cnt'].fillna(0)

    operation = pd.concat([login, logout], sort=False)
    operation['clientip_3'] = operation['clientip'].apply(
        lambda x: '.'.join(x.split('.')[:3]))
    operation['clientip_2'] = operation['clientip'].apply(
        lambda x: '.'.join(x.split('.')[:2]))
    operation['hour'] = operation['dteventtime'].dt.hour
    operation = operation.sort_values(['uin', 'roleid', 'dteventtime'])
    operation.head()

    # 账户在线时长统计 百分位
    df_temp = logout.groupby(['uin'])['onlinetime'].agg({
        'uin_onlinetime_sum':
        'sum',
        'uin_onlinetime_median':
        'median',
        'uin_onlinetime_mean':
        'mean'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')
    del df_temp
    gc.collect()

    # 账户rolelevel统计
    df_temp = logout.groupby(['uin'])['rolelevel'].agg({
        'uin_rolelevel_mean':
        'mean',
        'uin_rolelevel_max':
        'max'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')
    del df_temp
    gc.collect()

    # 账户power统计
    df_temp = logout.groupby(['uin'])['power'].agg({
        'uin_power_mean': 'mean',
        'uin_power_max': 'max'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')
    del df_temp
    gc.collect()

    # 账户friendsnum统计
    df_temp = logout.groupby(['uin'])['friendsnum'].agg({
        'uin_friendsnum_mean':
        'mean',
        'uin_friendsnum_max':
        'max'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')
    del df_temp
    gc.collect()

    # 账户hour统计
    df_temp = operation.groupby(['uin'])['hour'].agg({
        'uin_hour_mean': 'mean',
        'uin_hour_min': 'min',
        'uin_hour_max': 'max'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')
    del df_temp
    gc.collect()

    # 账户 areaid
    df_temp = operation[['uin', 'areaid']].drop_duplicates()
    df_feature = df_feature.merge(df_temp, how='left')
    del df_temp
    gc.collect()

    for f in tqdm([
            'platid', 'worldid', 'roleid', 'job', 'network', 'clientip',
            'clientip_3', 'hour'
    ]):
        df_temp = operation[['uin', f]].drop_duplicates()
        df_temp = df_temp.groupby(['uin'])[f].agg({
            'uin_{}_count'.format(f):
            'count'
        }).reset_index()
        df_feature = df_feature.merge(df_temp, how='left')
        del df_temp
        gc.collect()

    df_temp = create[['uin', 'regchannel']].drop_duplicates()
    df_temp = df_temp.groupby(['uin'])['regchannel'].agg({
        'uin_{}_count'.format('regchannel'):
        'count'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')
    del df_temp
    gc.collect()

    operation.head()

    operation['offlinetime'] = operation.groupby(['uin'])['dteventtime'].diff()
    operation['offlinetime'] = operation['offlinetime'].dt.total_seconds()

    df_tmep = operation[operation['onlinetime'].isnull()]

    df_temp = df_tmep.groupby(['uin'])['offlinetime'].agg({
        'uin_offlinetime_mean':
        'mean',
        'uin_offlinetime_max':
        'max',
        'uin_offlinetime_sum':
        'sum'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')
    del df_temp
    gc.collect()

    for f in ['clientip', 'clientip_3']:
        df_temp = operation[['uin', f]]
        df_temp.drop_duplicates(inplace=True)
        df_temp = df_temp.groupby([f])['uin'].nunique().reset_index()
        df_temp.columns = [f, 'uin_count']

        df_temp2 = operation[['uin', f]]
        df_temp2.drop_duplicates(inplace=True)
        df_temp = df_temp2.merge(df_temp, how='left')

        df_temp = df_temp.groupby(['uin'])['uin_count'].agg({
            '{}_uin_count_mean'.format(f):
            'mean'
        }).reset_index()
        df_feature = df_feature.merge(df_temp, how='left')
        del df_temp, df_temp2
        gc.collect()

    for f in tqdm(['clientip_3', 'network']):
        df_temp = operation.groupby(
            ['uin'])[f].agg(lambda x: stats.mode(x)[0][0]).reset_index()
        df_feature = df_feature.merge(df_temp, how='left')
        del df_temp
        gc.collect()

    df_feature.head()

    # # role_moneyflow 相关特征

    role_moneyflow.head()

    # 用户money统计
    df_temp = role_moneyflow.groupby(['uin'])['iMoney'].agg({
        'uin_moneyflow_count':
        'size',
        'uin_moneyflow_sum':
        'sum',
        'uin_moneyflow_mean':
        'mean',
        'uin_moneyflow_max':
        'max'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    del df_temp
    gc.collect()

    df_temp = role_moneyflow.groupby(['uin'])['AfterMoney'].agg({
        'uin_AfterMoney_sum':
        'sum'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    del df_temp
    gc.collect()

    # 用户金钱增加数量
    df_temp = role_moneyflow[role_moneyflow['AddOrReduce'] == 0]
    df_temp = df_temp.groupby(['uin'])['iMoney'].agg({
        'uin_moneyflow_add_count':
        'size',
        'uin_moneyflow_add_sum':
        'sum',
        'uin_moneyflow_add_mean':
        'mean',
        'uin_moneyflow_add_max':
        'max'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    del df_temp
    gc.collect()

    # 用户金钱减少数量
    df_temp = role_moneyflow[role_moneyflow['AddOrReduce'] == 1]
    df_temp = df_temp.groupby(['uin'])['iMoney'].agg({
        'uin_moneyflow_reduce_count':
        'size',
        'uin_moneyflow_reduce_sum':
        'sum',
        'uin_moneyflow_reduce_mean':
        'mean',
        'uin_moneyflow_reduce_max':
        'max'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    del df_temp
    gc.collect()

    df_feature['uin_moneyflow_reduce_count_minus_add_count'] = df_feature[
        'uin_moneyflow_reduce_count'] - df_feature['uin_moneyflow_add_count']

    df_temp = role_moneyflow.pivot_table(index=['uin'],
                                         columns=['iMoneyType'],
                                         values=['iMoney'],
                                         aggfunc=[np.sum,
                                                  np.mean]).reset_index()
    df_temp.columns = ['uin'] + [
        '{}_{}_by_iMoneyType{}'.format(f[1], f[0], f[2])
        for f in df_temp.columns
    ][1:]

    df_feature = df_feature.merge(df_temp, how='left')

    del df_temp
    gc.collect()

    df_temp = role_moneyflow.pivot_table(index=['uin'],
                                         columns=['Reason'],
                                         values=['iMoney'],
                                         aggfunc=[np.sum,
                                                  np.mean]).reset_index()
    df_temp.columns = ['uin'] + [
        '{}_{}_by_Reason{}'.format(f[1], f[0], f[2]) for f in df_temp.columns
    ][1:]

    df_feature = df_feature.merge(df_temp, how='left')

    del df_temp
    gc.collect()

    role_moneyflow['hour'] = role_moneyflow['dteventtime'].dt.hour

    df_temp = role_moneyflow.groupby(['uin'])['hour'].agg({
        'moneyflow_hour_mean':
        'mean',
        'moneyflow_hour_max':
        'max',
        'moneyflow_hour_min':
        'min'
    }).reset_index()

    df_feature = df_feature.merge(df_temp, how='left')
    del df_temp
    gc.collect()

    df_feature.head()

    # # role_itemflow 相关特征

    role_itemflow.head()

    df_temp = role_itemflow.groupby(['uin'])['Count'].agg({
        'uin_itemflow_count':
        'size',
        'uin_itemflow_sum':
        'sum'
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    del df_temp
    gc.collect()

    df_temp = role_itemflow.pivot_table(index=['uin'],
                                        columns=['Reason'],
                                        values=['Count'],
                                        aggfunc=[np.sum]).reset_index()
    df_temp.columns = ['uin'] + [
        '{}_{}_by_Reason{}'.format(f[1], f[0], f[2]) for f in df_temp.columns
    ][1:]

    df_feature = df_feature.merge(df_temp, how='left')

    del df_temp
    gc.collect()

    role_itemflow['hour'] = role_itemflow['dteventtime'].dt.hour

    df_temp = role_itemflow.groupby(['uin'])['hour'].agg({
        'itemflow_hour_mean':
        'mean',
        'itemflow_hour_max':
        'max',
        'itemflow_hour_min':
        'min'
    }).reset_index()

    df_feature = df_feature.merge(df_temp, how='left')
    del df_temp
    gc.collect()

    df_feature.head()

    # embedding

    for f1, f2 in [['uin', 'clientip_3'], ['uin', 'worldid'], ['uin', 'job']]:
        df_emb = emb(operation, f1, f2)
        df_feature = df_feature.merge(df_emb, on=f1, how='left')

    df_feature.head()

    # 五折特征

    # 5折交叉
    df_train = df_feature[~df_feature['target'].isnull()]
    df_train = df_train.reset_index(drop=True)
    df_test = df_feature[df_feature['target'].isnull()]

    df_stas_feat = None
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    for train_index, val_index in kf.split(df_train, df_train['target']):
        df_fold_train = df_train.iloc[train_index]
        df_fold_val = df_train.iloc[val_index]

        df_fold_val = statis_feat(df_fold_train, df_fold_val)
        df_stas_feat = pd.concat([df_stas_feat, df_fold_val], axis=0)

        del (df_fold_train)
        del (df_fold_val)
        gc.collect()

    df_test = statis_feat(df_train, df_test)
    df_feature = pd.concat([df_stas_feat, df_test], axis=0)

    del (df_stas_feat)
    del (df_train)
    del (df_test)
    gc.collect()

    df_feature.head()

    df_feature.shape

    df_feature.to_pickle('feature.plk')
