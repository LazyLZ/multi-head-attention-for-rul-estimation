import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
from sklearn.cluster import KMeans


class CMAPSSDataset(Dataset):
    # [op1, op2, op3]
    OPERATION_COLS = ['op%d' % i for i in range(1, 3 + 1)]
    # [s1, s2, ..., s21]
    SENSOR_COLS = ['s%d' % i for i in range(1, 21 + 1)]
    # [id, time, op1, op2, op3, s1, s2, ..., s21]
    DATASET_COLS = ['id', 'time'] + OPERATION_COLS + SENSOR_COLS

    @staticmethod
    def z_score_normalization(df, cols, norm_params):
        if len(norm_params.shape) == 2:
            for col_i, col in enumerate(cols):
                mean, standard = norm_params[col_i]
                values = df[col].values
                values = values - mean
                if standard != 0:
                    values = values / standard
                df[col] = values
        elif len(norm_params.shape) == 3:
            op_list = df['op_type'].unique()
            op_list.sort()
            for op_i, op in enumerate(op_list):
                for col_i, col in enumerate(cols):
                    mean, standard = norm_params[op_i, col_i]
                    values = df[df['op_type'] == op][col].values
                    values = (values - mean)
                    if standard != 0:
                        values = values / standard
                    df.loc[df['op_type'] == op, col] = values
        else:
            raise ValueError('norm_params shape error')
        return df

    @staticmethod
    def min_max_normalization(df, cols, norm_params, min_norm, max_norm):
        if len(norm_params.shape) == 2:
            for col_i, col in enumerate(cols):
                min_v, max_v = norm_params[col_i]
                if max_v == min_v:
                    df[col] = (min_norm + max_norm) / 2  # median
                else:
                    df[col] = (((max_norm - min_norm) * (df[col].values - min_v)) / (max_v - min_v)) + min_norm
        elif len(norm_params.shape) == 3:
            op_list = df['op_type'].unique()
            op_list.sort()
            for op_i, op in enumerate(op_list):
                for col_i, col in enumerate(cols):
                    min_v, max_v = norm_params[op_i, col_i]
                    if max_v == min_v:
                        df.loc[df['op_type'] == op, col] = (min_norm + max_norm) / 2  # median
                    else:
                        values = df[df['op_type'] == op][col].values
                        df.loc[df['op_type'] == op, col] = (((max_norm - min_norm) * (values - min_v)) / (
                                max_v - min_v)) + min_norm

        else:
            raise ValueError('norm_params shape error')
        return df

    @staticmethod
    def clustering_operations(dataset_list):
        df_list = []
        for dataset in dataset_list:
            df_list.append(dataset.df)

        full_df = pd.concat(df_list)
        op_types = KMeans(n_clusters=6, random_state=1).fit_predict(full_df[['op1', 'op2', 'op3']].values)
        full_df.insert(2, 'op_type', op_types)
        start = 0
        for i, df in enumerate(df_list):
            dataset = dataset_list[i]
            df_len = len(dataset.df)
            # print('loc', start, df_len)
            dataset.df = full_df.iloc[start: start + df_len]
            dataset.has_cluster_operations = True
            start += df_len

    @staticmethod
    def gen_norm_params(dataset_list, norm_type, norm_by_operations=False):
        """
            Get normalization parameters from multiple dataset.
            if normalization by conditions, need cluster operations first.

        :param dataset_list: list of CMPASSDataset instances
        :param norm_type: 0-1, -1-1 or z-score
        :param norm_by_operations:
        :return: normalize by each op_type
        """
        assert norm_type in ['0-1', '-1-1', 'z-score']
        df_list = []
        feature_cols = None
        for dataset in dataset_list:
            # must have save feature_cols
            if feature_cols is not None:
                assert feature_cols == dataset.feature_cols, \
                    'multiple dataset normalization must have same feature_cols'
            # must have same norm_by_operations setting
            assert norm_by_operations == bool(dataset.norm_by_operations), \
                'multiple dataset normalization must have same norm_by_operations setting'
            if norm_by_operations:
                assert dataset.has_cluster_operations, \
                    'need cluster operations before normalization when norm_by_operations is True'

            feature_cols = dataset.feature_cols
            df_list.append(dataset.df)

        norm_cols = feature_cols

        full_df = pd.concat(df_list)

        if norm_by_operations:
            df_list = []
            op_list = full_df['op_type'].unique()
            op_list.sort()
            for op in op_list:
                df_list.append(full_df[full_df['op_type'] == op])
        else:
            df_list = [full_df]

        params_list = []
        for df in df_list:
            if norm_type == '0-1' or norm_type == '-1-1':
                col_max = np.max(df[norm_cols].values, axis=0)
                col_min = np.min(df[norm_cols].values, axis=0)
                params_list.append(np.stack((col_min, col_max), axis=1))
            if norm_type == 'z-score':
                mean = np.mean(df[norm_cols].values, axis=0)
                standard = np.std(df[norm_cols].values, axis=0)
                params_list.append(np.stack((mean, standard), axis=1))

        norm_params = np.stack(params_list, axis=0)

        return norm_params.squeeze()

    @staticmethod
    def get_datasets(dataset_root, sub_dataset, sequence_len=1, max_rul=None,
                     return_sequence_label=False, norm_type=None, cluster_operations=False,
                     norm_by_operations=False, include_cols=None, exclude_cols=None, return_id=False,
                     validation_rate=0.2, use_only_final_on_test=True, use_max_rul_on_test=False,
                     use_max_rul_on_valid=True):
        """
            Get train, valid, test dataset from dataset file.
            The parameter with the same name as in __init__ has the same effect, they are:
            sequence_len, max_rul, return_sequence_label, include_cols, exclude_cols, return id

        :param dataset_root:
            root directory of raw txt files

        :param sub_dataset:
            A string denote the dataset name, FD001/FD002/FD003/FD004

        :param sequence_len:
        :param max_rul:
        :param return_sequence_label:
        :param norm_type:
        :param cluster_operations:
        :param norm_by_operations:
        :param include_cols:
        :param exclude_cols:
        :param return_id:

        :param validation_rate:
            Number of units used in the validation set as a percentage of the total training set, default is 0.2
            validation_rate = len(validation_dataset.df['id'].unique()) / len(full_train_dataset.df['id'].unique())

        :param use_only_final_on_test:
            set only_final on test dataset, default is True

        :param use_max_rul_on_test:
            use max_rul on test dataset

        :param use_max_rul_on_valid:
            use max_rul on validation dataset

        """
        if sub_dataset == 'PHM08':
            train_df = pd.read_csv(os.path.join(dataset_root, 'train.txt'), sep=' ', header=None)
            test_df = pd.read_csv(os.path.join(dataset_root, 'test.txt'.format(sub_dataset)), sep=' ', header=None)
            # PHM08 test dataset has 218 unit
            rul = np.empty(218)
            rul[:] = np.nan
        else:
            train_df = pd.read_csv(os.path.join(dataset_root, 'train_{:s}.txt'.format(sub_dataset)), sep=' ',
                                   header=None)
            test_df = pd.read_csv(os.path.join(dataset_root, 'test_{:s}.txt'.format(sub_dataset)), sep=' ', header=None)
            rul_df = pd.read_csv(os.path.join(dataset_root, 'RUL_{:s}.txt'.format(sub_dataset)), header=None)
            rul = rul_df.values.squeeze()

        # split valid set
        # train_df[0] is unit id column
        valid_df = None
        valid_dataset = None
        assert 0 <= validation_rate <= 0.99
        if validation_rate:
            ids = train_df[0].unique()
            max_id = np.max(ids)
            valid_len = int(validation_rate * max_id)
            if valid_len:
                # random chose valid engine id
                valid_ids = np.random.choice(np.arange(1, max_id + 1), valid_len, replace=False)

                isin_df = np.isin(train_df[0].to_numpy(), valid_ids)
                valid_df = train_df.iloc[np.where(isin_df == True)]
                train_df = train_df.iloc[np.where(isin_df == False)]

        if sub_dataset in ['FD001', 'FD003']:
            norm_by_operations = False
            cluster_operations = False

        dataset_kwargs = {
            'sequence_len': sequence_len,
            'max_rul': max_rul,
            'norm_type': norm_type,
            'include_cols': include_cols,
            'exclude_cols': exclude_cols,
            'cluster_operations': cluster_operations,
            'norm_by_operations': norm_by_operations,
            'return_sequence_label': return_sequence_label,
            'return_id': return_id
        }

        # print
        train_dataset = CMAPSSDataset(
            train_df,
            init=False,
            **dataset_kwargs
        )
        dataset_kwargs['final_rul'] = rul
        if not use_max_rul_on_test and 'max_rul' in dataset_kwargs:
            dataset_kwargs.pop('max_rul')
        if use_only_final_on_test:
            dataset_kwargs['only_final'] = True

        test_dataset = CMAPSSDataset(
            test_df,
            init=False,
            **dataset_kwargs
        )

        if valid_df is not None:
            if 'final_rul' in dataset_kwargs:
                dataset_kwargs.pop('final_rul')
            if not use_max_rul_on_valid and 'max_rul' in dataset_kwargs:
                dataset_kwargs.pop('max_rul')
            if use_max_rul_on_valid and max_rul is not None:
                dataset_kwargs['max_rul'] = max_rul
            if 'only_final' in dataset_kwargs:
                dataset_kwargs.pop('only_final')
            valid_dataset = CMAPSSDataset(
                valid_df,
                init=False,
                **dataset_kwargs
            )

        dataset_list = [train_dataset, test_dataset]
        if valid_df is not None:
            dataset_list.append(valid_dataset)

        if cluster_operations:
            CMAPSSDataset.clustering_operations(dataset_list)

        if norm_type:
            norm_params = CMAPSSDataset.gen_norm_params(dataset_list, norm_type, norm_by_operations)
            for dataset in dataset_list:
                dataset._set_norm_params(norm_type, norm_params, norm_by_operations)
                dataset.normalization()

        for dataset in dataset_list:
            dataset.gen_sequence()

        return train_dataset, test_dataset, valid_dataset

    @staticmethod
    def get_data_loaders(loader_kwargs, train_kwargs=None, test_kwargs=None, valid_kwargs=None, **dataset_kwargs):
        """
        :param loader_kwargs:
            kwargs pass to all DataLoader
        :param train_kwargs:
            kwargs only pass to train DataLoader, will cover loader_kwargs's same key
        :param test_kwargs:
            kwargs only pass to test DataLoader, will cover loader_kwargs's same key
        :param valid_kwargs:
            kwargs only pass to valid DataLoader, will cover loader_kwargs's same key
        :param dataset_kwargs:
            dataset arguments which describe above
        """
        train_kwargs = train_kwargs or dict()
        test_kwargs = test_kwargs or dict()
        valid_kwargs = valid_kwargs or dict()
        train_kwargs.update(loader_kwargs)
        test_kwargs.update(loader_kwargs)
        valid_kwargs.update(loader_kwargs)

        train_kwargs['shuffle'] = True
        train_dataset, test_dataset, valid_dataset = CMAPSSDataset.get_datasets(**dataset_kwargs)
        print('tran/valid/test', len(train_dataset), len(valid_dataset) if valid_dataset else 0, len(test_dataset))
        # print('train_kwargs', train_kwargs)
        train_loader = DataLoader(train_dataset, **train_kwargs)
        test_loader = DataLoader(test_dataset, **test_kwargs)
        valid_loader = DataLoader(valid_dataset, **valid_kwargs) if valid_dataset else None

        return train_loader, test_loader, valid_loader

    def __init__(self, data_df, sequence_len=1, final_rul=None, norm_params=None, norm_type=None,
                 max_rul=None, only_final=False, init=True, return_sequence_label=False,
                 cluster_operations=False, norm_by_operations=False, include_cols=None,
                 exclude_cols=None, no_rul=False, return_id=False):
        """

            C-MAPSS Dataset, create pytorch Dataset by pd.Dataframe use original txt file,
            PHM08 Challenge Dataset is also supported.
            C-MAPSS and PHM08 Dataset download: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

        :param data_df:
            Required, pd.Dataframe from 'train_FD00X.txt/test_FD00X.txt/train.txt/test.txt.

        :param sequence_len:
            sequence length of time window pre-progress, default is 1.
            e.g.: a unit has 200 cycles, seq_length=50 means generate data as
                data = [[cycle 0 - 50],
                        [cycle 1 - 51],
                        ...
                        [cycle 150-199]]
                        # shape(sequences_num, sequence_len, features_num) = shape(150, 50, 24)

        :param final_rul:
            An list or nparray denote the RUL for the last time cycle of each unit,
            which are all set to 0 for the training set, default is None.

        :param norm_params:
            An numpy array to set the normalization params manually, default is None.
            if not provide, it will calculate the params using provided data.
            this params can be use in the situation that normalize the training and test dataset together.
            the norm_params is shaped as (sensor, params),
            sensor represents the i-th sensor and params represents the params of the normalization methods,
                for min-max normalization, the value is [min, max].
                for z-score normalization, the value is [μ, σ].
            if cluster_operations and norm_by_operation are both set to True, the norm_params is shape as
            (op_type, sensor, params), op_type represents the j-th operation.

        :param norm_type:
            A string represents the normalization type, '0-1', '-1-1' or 'z-score', default is None.

        :param max_rul:
            Number, a piece-wise RUL function on RUL, RUL exceeding max_rul will be set to max_rul.
            the read RUL will be store in self.df['real_rul'].

        :param only_final:
            only use the last time window's data and label, use in test sets, default False

        :param init:
            A Boolean denote whether generate the sequence in __init__, default is True
            if False, only set self.df and calculate max_rul on __init__ ,
            you need to call function manually, like normalization() to normalize data, clustering_operation() to
            cluster operational settings and get_sequence() to generate the sequence.

        :param return_sequence_label:
            return all RUL instead of only last RUL, default is False.

        :param cluster_operations:
            implement a K-Means cluster on three operational settings,
            an new column named 'op_type' will insert to the self.df, but not add to feature columns, default is False

        :param norm_by_operations:
            if is cluster operational settings,
            set this to True to normalize data by operation types, default is False.

        :param include_cols:
            use include_cols as features, e.g. ['s1', 's2'], default is None,
            means use all operations and sensors is feature.

        :param exclude_cols:
            exclude features, e.g. ['op3', 's2', 's3'], default is None

        :params no_rul:
            Do not count RUL, use in PHM08's test dataset

        :param return_id:
            return unit id, default is False

        """
        super().__init__()
        assert isinstance(data_df, pd.DataFrame), 'data_df need pd.DataFrame'
        assert len(data_df.columns) >= 26, 'Invalid Dataframe input (columns < 26)'

        # set self.df
        self.df = data_df
        if len(self.df.columns) >= 26:
            self.df = self.df.drop([26, 27], axis=1)
        self.df.columns = CMAPSSDataset.DATASET_COLS
        # print('init', self.df)
        # sequence_len
        assert sequence_len > 0, 'Need sequence_len > 0, got:' + str(sequence_len)
        self.sequence_len = sequence_len

        # feature cols define
        if include_cols is not None:
            self.feature_cols = include_cols
        else:
            self.feature_cols = CMAPSSDataset.OPERATION_COLS + CMAPSSDataset.SENSOR_COLS

        if exclude_cols is not None:
            for v in exclude_cols:
                if v in self.feature_cols:
                    self.feature_cols.remove(v)

        # init
        self.init = init

        # final rul
        if final_rul is None:
            self.final_rul = np.zeros(self.df['id'].nunique())
        else:
            self.final_rul = final_rul

        # norm type
        self.norm_type = None
        # norm params
        self.norm_params = None
        # normalization by operations
        self.norm_by_operations = None

        self._set_norm_params(norm_type, norm_params, norm_by_operations)

        # max rul
        if max_rul is None:
            max_rul = 999999
        self.max_rul = max_rul

        # only final
        self.only_final = only_final

        # compute RUL
        self.no_rul = no_rul
        self.count_rul()

        # cluster on operations
        self.cluster_operations = cluster_operations

        # return sequence_label
        self.return_sequence_label = return_sequence_label

        # return id
        self.return_id = return_id

        # sequence data
        self.sequence_array = None
        self.label_array = None
        self.id_array = None

        # has clustering conditions
        self.has_cluster_operations = False
        self.has_normalization = False
        self.has_gen_sequence = False

        if self.init:
            if self.cluster_operations:
                CMAPSSDataset.clustering_operations([self])

            if self.norm_type:
                self.normalization()

            self.gen_sequence()

        # print('normalizaiont setting', self.norm_by_operations, self.cluster_operations)

    def __len__(self):
        return len(self.sequence_array)

    def __getitem__(self, i):
        '''
        :param i: get i_th data
        :return: sequence, target
            sequence: tensor([time, setting1, ... , sensor21])
            target: tensor([rul])
        '''
        l = [torch.FloatTensor(self.sequence_array[i]), torch.FloatTensor([self.label_array[i]])]
        if self.return_id:
            l.append(torch.LongTensor([self.id_array[i]]))
        return tuple(l)

    def count_rul(self):
        df = self.df

        final_rul = self.final_rul
        max_rul = self.max_rul
        time_series = df.groupby('id').size()
        rul_array = time_series.values + final_rul
        rul_df = pd.DataFrame({
            'id': time_series.index,
            'rul': rul_array,
            'real_rul': rul_array
        })
        df = pd.merge(df, rul_df)
        df['real_rul'] = df.apply(lambda x: x['rul'] - x['time'], axis=1)
        df['rul'] = df.apply(lambda x: max_rul if max_rul < x['real_rul'] else x['real_rul'], axis=1)
        self.df = df

    def _set_norm_params(self, norm_type, norm_params, norm_by_operations):
        # norm_params
        self.norm_params = norm_params

        # norm_type
        assert norm_type is None or norm_type in ['z-score', '0-1', '-1-1']
        self.norm_type = norm_type

        # norm by operations
        self.norm_by_operations = norm_by_operations

    def normalization(self):
        if self.norm_type is None:
            return

        if self.cluster_operations and self.norm_by_operations and not self.has_cluster_operations:
            raise RuntimeError('need cluster operations before normalization when norm_by_operations is True')

        # TODO specific normalization cols
        df = self.df
        norm_cols = self.feature_cols
        norm_type = self.norm_type
        norm_by_operations = self.norm_by_operations
        if self.norm_params is None:
            self.norm_params = CMAPSSDataset.gen_norm_params([self], norm_type, norm_by_operations)
        norm_params = self.norm_params
        if norm_type == '0-1':
            min_norm, max_norm = 0, 1
            self.df = CMAPSSDataset.min_max_normalization(df, norm_cols, norm_params, min_norm, max_norm)
        if norm_type == '-1-1':
            min_norm, max_norm = -1, 1
            self.df = CMAPSSDataset.min_max_normalization(df, norm_cols, norm_params, min_norm, max_norm)
        if norm_type == 'z-score':
            self.df = CMAPSSDataset.z_score_normalization(df, norm_cols, norm_params)
        self.has_normalization = True

    def gen_sequence(self):
        seq_cols = ['id'] + self.feature_cols + ['rul']
        seq_len = self.sequence_len
        all_array = []
        # print('gen_sequence')
        # print(self.df)
        for id in self.df['id'].unique():
            id_df = self.df[self.df['id'] == id].sort_values(by='time', ascending=True)
            id_array = id_df[seq_cols].values
            row_num = id_array.shape[0]
            if row_num >= seq_len:
                if self.only_final:
                    all_array.append(id_array[row_num - seq_len:])
                else:
                    for i in range(0, row_num - seq_len + 1):
                        all_array.append(id_array[i:i + seq_len])
            else:
                # row number < sequence length, only one sequence
                # pad width first time-cycle value
                all_array.append(np.pad(id_array, ((seq_len - id_array.shape[0], 0), (0, 0)), 'edge'))

        all_array = np.stack(all_array)

        self.sequence_array = all_array[:, :, 1:-1]

        if self.return_sequence_label:
            self.label_array = all_array[:, :, -1]
        else:
            self.label_array = all_array[:, -1, -1]

        self.id_array = all_array[:, 0, 0]

        self.has_gen_sequence = True

