import os
import numpy as np
import config
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess.utils.pcl_operation import normalize_feature, rescale
import argparse
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--struct_dir", type=str, required=True, default='path/to/7n5s_xy11_remove')
parser.add_argument("--bins_folder", type=str, default='pcl')
parser.add_argument("--imgs_folder", type=str, default='img')
parser.add_argument("--split", type=str, required=True, default='test')
parser.add_argument("--dim", type=int, default=5)
args = parser.parse_args()

print("path:",args.struct_dir)
paras = config.load_parameters(args.struct_dir)


bins_path=paras["PCL_DIR"]
bins_subfolders = [f.name for f in os.scandir(bins_path) if f.is_dir()]
print("Subfolders in bins_path:", bins_subfolders)

def load_pc_file(filename, DATA_DIM):
    pc = np.fromfile(filename, dtype=np.float64)
    # pc = np.float32(pc)
    pc = np.reshape(pc,(pc.shape[0]//DATA_DIM, DATA_DIM))
    return pc

def pcl_to_img_PIL(pcl_input, frame_index, output_dir):
    img_matrix = np.zeros((2*paras['MEASURE_RANGE'], 2*paras['MEASURE_RANGE'],), dtype='float') #2*100
    pcl_input[:, 1] = np.rint(pcl_input[:, 1]) + paras['MEASURE_RANGE'] #100
    pcl_candidate = []
    for index, point in enumerate(pcl_input):
        if point[0] <= 2*paras['MEASURE_RANGE'] - 1 and point[0] >= 0 and point[1] <= 2*paras['MEASURE_RANGE'] - 1 and point[1] >= 0: #100
            pcl_candidate.append(pcl_input[index])

    pcl_candidate = np.array(pcl_candidate, dtype=np.float64)

    if paras['FEATURE'] == 'r':
        pcl_candidate = normalize_feature(pcl_input=pcl_candidate, feature_channel=[5], target='01')
        for row in pcl_candidate:
            img_matrix[int(row[0]), int(row[1])] = row[5]
    elif paras['FEATURE'] == 't':
        pcl_candidate = normalize_feature(pcl_input=pcl_candidate, feature_channel=[17], target='01')
        for row in pcl_candidate:
            img_matrix[int(row[0]), int(row[1])] = row[17]
    elif paras['FEATURE'] == '1':
        pcl_candidate = np.array(pcl_candidate, dtype=np.int32)
        img_matrix[pcl_candidate[:, 0], pcl_candidate[:, 1]] = 1

    img_matrix = np.expand_dims(img_matrix, axis=2)
    img_matrix = np.repeat(img_matrix, 3, axis=2)

    img = Image.fromarray(np.uint8(img_matrix * 255.0)).convert('RGB')
    if img.size != (200, 200):
        img = img.resize((200, 200))
    img.save(os.path.join(output_dir, frame_index.split('.')[0]+'.jpg'))
    # img.save(os.path.join('tmp', '{:0>5d}.jpg'.format(frame_index)))
    pass

# for f_i in bins_subfolders:
split_path = os.path.join(bins_path, args.split)
groups_folder = [f.name for f in os.scandir(split_path) if f.is_dir()]
for group_i in groups_folder:
    print ("group_i:", group_i)
    group_path = os.path.join(split_path, group_i)
    seqs_folder = [f.name for f in os.scandir(group_path) if f.is_dir()]
    for seq_i in seqs_folder:
        seq_path = os.path.join(group_path, seq_i)
        all_files = [f.name for f in os.scandir(seq_path) if f.is_file()]
        output_dir = os.path.join(args.struct_dir, args.imgs_folder, args.split, group_i, seq_i)
        for i in tqdm(all_files):
            pcl_file_path = os.path.join(seq_path, i)
            pc = load_pc_file(pcl_file_path, args.dim)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            pcl_to_img_PIL(pcl_input=pc, frame_index=i, output_dir=output_dir)