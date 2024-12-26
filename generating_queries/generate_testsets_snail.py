import os
import pickle
import numpy as np
import pandas as pd
import argparse
import tqdm
from sklearn.neighbors import KDTree
import preprocess.config as config
##########################################
# split query and database data
# save in evaluation_database.pickle / evaluation_query.pickle
##########################################

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(output, handle, protocol=1)
    print("Done ", filename)

def construct_query_and_database_sets(base_path, data_split, seqs, positive_dist, yaw_threshold, use_yaw, save_folder, use_timestamp_name):
    database_trees = []
    database_sets = {}
    query_sets = {}

    for seq_id, seq in enumerate(tqdm.tqdm(seqs)):
        seq_path = os.path.join(base_path, data_split, seq)
        tras = [name for name in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, name))]
        tras.sort()
        query = {}
        for tra_id in range(len(tras)):
            pose_path = seq_path + '/' + tras[tra_id] + '_poses.txt'
            df_locations = pd.read_table(pose_path, sep=' ', converters={'timestamp': str},
                                         names=['timestamp','r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y', 'r31', 'r32', 'r33', 'z', 'yaw'])
            df_locations = df_locations.loc[:, ['timestamp', 'x', 'y', 'z', 'yaw']]

            if use_timestamp_name:
                df_locations['timestamp'] = data_split + '/' + seq + '/' + tras[tra_id] + '/' + df_locations['timestamp'] + '.bin'
                df_locations = df_locations.rename(columns={'timestamp': 'data_file'})
            else:
                df_locations['data_file'] = range(0, len(df_locations))
                df_locations['data_file'] = df_locations['data_file'].apply(lambda x: str(x).zfill(6))
                df_locations['data_file'] = data_split + '/' + seq + '/' + tras[tra_id] + '/' + df_locations['data_file'] + '.bin'

            if tra_id == 0:
                df_database = df_locations
                database_tree = KDTree(df_database[['x', 'y']])
                database_trees.append(database_tree)
                database = {}
                for index, row in df_locations.iterrows():
                    database[len(database.keys())] = {
                        'query': row['data_file'], 'x': row['x'], 'y': row['y'], 'yaw': row['yaw']}
                database_sets[seq] = database
            else:
                df_test = df_locations
                for index, row in df_test.iterrows():
                    query[len(query.keys())] = {'query': row['data_file'], 'x': row['x'], 'y': row['y'], 'yaw': row['yaw']}
        query_sets[seq] = query

        for key in range(len(query_sets[seq].keys())):
            coor = np.array([[query_sets[seq][key]["x"], query_sets[seq][key]["y"]]])
            index = database_trees[seq_id].query_radius(coor, r=positive_dist)[0].tolist()
            if use_yaw:
                yaw = query_sets[seq][key]["yaw"]
                yaw_diff = np.abs(yaw - df_database.iloc[index]["yaw"])
                true_index = [c for c in index if np.min((yaw_diff[c], 360-yaw_diff[c])) < cfgs.yaw_threshold]
            else:
                true_index = index
            query_sets[seq][key][seq] = true_index

    os.makedirs(save_folder, exist_ok=True)
    if use_yaw:
        output_to_file(database_sets, save_folder + 'evaluation_database_' + data_split + 'sl.pickle')
        output_to_file(query_sets,
                       save_folder + 'evaluation_query_' + data_split + '_' + str(positive_dist) + 'm_' + str(yaw_threshold) + 'sl.pickle')
    else:
        output_to_file(database_sets, save_folder + 'evaluation_database_' + data_split + '.pickle')
        output_to_file(query_sets, save_folder + 'evaluation_query_' + data_split + '_' + str(positive_dist) + 'm.pickle')

# Building database and query files for evaluation
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/media/cyw/KESU/datasets/clean_radar/',
                        help='radar datasets path')
    parser.add_argument('--data_split', type=str, default='test', help='test_short or test_long')
    parser.add_argument('--positive_dist', type=float, default=9,
                        help='Positive sample distance threshold, short:5, long:9')
    parser.add_argument('--yaw_threshold', type=float, default=30, help='Yaw angle threshold, 8 or 25')
    parser.add_argument('--use_yaw', type=bool, default=True, help='If use yaw to determine a positive sample.')
    parser.add_argument('--use_timestamp_name', type=bool, default=False, help='save most similar index for loss')
    parser.add_argument('--struct_dir', type=str, default='/media/cyw/KESU/datasets/clean_radar/7n5s_xy11_remove_oculii/',
                        help='the saved path of split file ')
    parser.add_argument('--data_type', type=str, default='oculii', help='test_short or test_long')

    cfgs = parser.parse_args()
    paras = config.load_parameters(cfgs.struct_dir)
    save_path = cfgs.struct_dir

    data_path = os.path.join(cfgs.data_path, 'processed_snail_radar', cfgs.data_type)

    if 'valid' in cfgs.data_split:
        if "ars548" in data_path:
            seqs = ['if', 'iaf', 'sl'] #
        elif "oculii" in cfgs.data_type:
            seqs = ['if', 'iaf', 'sl']
        else:
            raise Exception('Loading error!')
    elif 'test' in cfgs.data_split:
        if "ars548" in data_path:
            seqs = ['bc', 'ss', 'st', 'sl', 'if', 'iaf', 'iaef', '81r']
        elif "oculii" in data_path:
            seqs = ['bc', 'ss', 'st', 'sl', 'if', 'iaf', 'iaef', '81r']
        else:
            raise Exception('Loading error!')

    construct_query_and_database_sets(data_path, cfgs.data_split, seqs, cfgs.positive_dist, cfgs.yaw_threshold,
                                      cfgs.use_yaw, save_path, cfgs.use_timestamp_name)
