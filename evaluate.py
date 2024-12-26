from torch.utils.data import DataLoader
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import faiss
import tqdm
import os
import pickle

def get_recall(opt, model, db_set, query_set, seed_worker, epoch=1, writer=None):
    torch.cuda.set_device(opt.cGPU)
    device = torch.device("cuda")

    db_data_loader = DataLoader(dataset=db_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
    query_data_loader = DataLoader(dataset=query_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                                  pin_memory=True, worker_init_fn=seed_worker)

    eval_group_len = db_set.group_len
    query_info = query_set.groups

    model.eval()
    with torch.no_grad():
        dbFeat = np.empty((len(db_set), opt.output_dim))
        print('get recall..')
        for iteration, (input, indices) in enumerate(tqdm.tqdm(db_data_loader, ncols=40), 1):
            input = input.to(device)
            vlad_encoding = model(input)
            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            del input, vlad_encoding

        queryFeat = np.empty((len(query_set), opt.output_dim))
        for iteration, (input, indices) in enumerate(tqdm.tqdm(query_data_loader, ncols=40), 1):
            input_q = input.to(device)
            vlad_encoding_q = model(input_q)
            queryFeat[indices.detach().numpy(), :] = vlad_encoding_q.detach().cpu().numpy()
            del input_q, vlad_encoding_q
    del db_data_loader, query_data_loader

    num_neighbors = 25
    recall = np.zeros(num_neighbors)
    one_percent_recalls = []
    recall_list = []

    q_start_index = 0
    db_start_index = 0
    seqs_gt = {}
    query_vaild_files = {}
    db_vaild_files = {}
    DATABASE_VECTORS = []
    QUERY_VECTORS = []
    for i, group_name in enumerate(query_info):
        valid_db_i = db_set.valid_db_index[i]
        db_end_index = db_start_index + eval_group_len[i]
        dbFeat_i = dbFeat[db_start_index: db_end_index].astype('float32')
        db_start_index = db_end_index
        DATABASE_VECTORS.append(dbFeat_i)

        vaild_query_i = query_info[group_name]['valid_index']
        q_end_index = q_start_index + len(vaild_query_i)
        queryFeat_i = queryFeat[q_start_index: q_end_index].astype('float32')
        q_start_index = q_end_index
        QUERY_VECTORS.append(queryFeat_i)

        threshold = max(int(round(len(dbFeat_i) / 100.0)), 1)

        # ----------------------------------------------------- faiss ---------------------------------------------------- #
        faiss_index = faiss.IndexFlatL2(opt.output_dim)
        faiss_index.add(dbFeat_i)
        dists, predictions = faiss_index.search(queryFeat_i, len(dbFeat_i))

        # for each query get those within threshold distance
        gt = query_info[group_name]['true_neighbors']
        correct_at_n = np.zeros(num_neighbors)
        true_gt = []

        num_evaluated = 0
        one_percent_retrieved = 0
        for qIx, pred in enumerate(predictions):
            neighbors = gt[qIx]
            true_neighbors = [valid_db_i.index(pos) for pos in neighbors if pos in valid_db_i]
            true_gt.append(true_neighbors)
            if not true_neighbors:
                continue
            num_evaluated += 1
            for j, n in enumerate(range(num_neighbors)):
                if np.any(np.in1d(pred[:n+1], true_neighbors)):
                    correct_at_n[j:] += 1
                    break
            if len(list(set(pred[0:threshold]).intersection(set(true_neighbors)))) > 0:
                one_percent_retrieved += 1

        seqs_gt[group_name] = true_gt
        query_infos = query_set.dbStruct[group_name]
        query_files_name_list = [query_infos[info_i]['query'] for i, info_i in enumerate(query_infos) if i in vaild_query_i]
        query_vaild_files[group_name] = query_files_name_list

        db_infos = db_set.dbStruct[group_name]
        db_files_name_list = [db_infos[db_infos_i]['query'] for i, db_infos_i in enumerate(db_infos) if i in valid_db_i]
        db_vaild_files[group_name] = db_files_name_list

        recall_at_n = correct_at_n / float(num_evaluated) * 100.0 if num_evaluated > 0 else 0
        one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100 if num_evaluated > 0 else 0

        recall += np.array(recall_at_n)
        recall_list.append(recall_at_n)
        one_percent_recalls.append(one_percent_recall)

    ave_recall = recall / len(eval_group_len)
    ave_one_percent_recall = np.mean(one_percent_recalls)

    print('EVAL RECALL: %s' % str(ave_recall))
    print('EVAL one_percent_recall: %s' % str(ave_one_percent_recall))

    if opt.save_features:
        discriptors_path = 'postprocess_cyw/discriptors/'
        if not os.path.exists(discriptors_path):
            os.mkdir(discriptors_path)

        with open(os.path.join(discriptors_path, opt.results.split('/')[-1].split('.')[0] + '_discriptors.pickle'), 'wb') as f:
            pickle.dump({'DATABASE_VECTORS': DATABASE_VECTORS, 'QUERY_VECTORS': QUERY_VECTORS,
                         'query_gt': seqs_gt, 'query_vaild_files':query_vaild_files, 'db_vaild_files': db_vaild_files}, f)

    if opt.mode == 'evaluate':
        print('[{}]\t'.format(opt.split), end='')
        print('recall@1: {:.2f}\t'.format(ave_recall[1]), end='')
        print('recall@5: {:.2f}\t'.format(ave_recall[5]), end='')
        print('recall@10: {:.2f}\t'.format(ave_recall[10]), end='')
        print('recall@20: {:.2f}\t'.format(ave_recall[20]))

        with open(opt.results, "w") as output:
            for recall, one_percent_recall in zip(recall_list, one_percent_recalls):
                output.write("Recall @N:\n")
                output.write(str(recall))
                output.write("\n\n")
                output.write("Top 1% Recall:\n")
                output.write(str(one_percent_recall))
                output.write("\n\n")
            output.write("Average Recall @N:\n")
            output.write(str(ave_recall))
            output.write("\n\n")
            output.write("Average Top 1% Recall:\n")
            output.write(str(ave_one_percent_recall))
            output.write("\n\n")

    return ave_recall, ave_one_percent_recall
