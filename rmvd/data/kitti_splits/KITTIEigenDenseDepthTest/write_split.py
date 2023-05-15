from pprint import pprint

import os.path as osp


def list_to_desc(l):
    sub_lists = []
    cur_list = []
    cur_list_diff = None

    for idx in range(len(l)):

        cur_ele = l[idx]

        if len(cur_list) == 0:
            cur_list.append(cur_ele)
        elif cur_list_diff is None:
            cur_list_diff = cur_ele - cur_list[-1]
            cur_list.append(cur_ele)
        elif (cur_ele - cur_list[-1]) == cur_list_diff:
            cur_list.append(cur_ele)
        else:

            if len(cur_list) == 2:
                last_ele = cur_list[-1]
                cur_list = cur_list[:-1]

                sub_lists.append(cur_list)

                cur_list = [last_ele, cur_ele]
                cur_list_diff = cur_ele - last_ele

            else:
                sub_lists.append(cur_list)

                cur_list = [cur_ele]
                cur_list_diff = None
                
    if len(cur_list) == 2:
        sub_lists.append(cur_list[:1])
        sub_lists.append(cur_list[1:])
    elif len(cur_list) > 0:
        sub_lists.append(cur_list)

    sub_list_str = []
    for sub_list in sub_lists:
        if len(sub_list) < 3:
            sub_list_str.append(','.join([str(x) for x in sub_list]))
        else:
            diff = sub_list[1] - sub_list[0]
            if diff == 1:
                sub_list_str.append('[' + str(sub_list[0]) + ':' + str(sub_list[-1]) + ']')
            else:
                sub_list_str.append('[' + str(sub_list[0]) + ':' + str(sub_list[-1]) + ':' + str(diff) + ']')

    return ','.join(sub_list_str)


def desc_to_list(desc):
    out_list = []

    sub_descs = desc.split(',')
    for sub_desc in sub_descs:
        if sub_desc.isdigit():
            out_list.append(int(sub_desc))
        else:
            sub_desc = sub_desc[1:-1]
            sub_desc = sub_desc.split(':')
            if len(sub_desc) == 2:
                start = int(sub_desc[0])
                end = int(sub_desc[1])
                out_list += list(range(start, end))
                out_list.append(int(end))
            elif len(sub_desc) == 3:
                start = int(sub_desc[0])
                end = int(sub_desc[1])
                step = int(sub_desc[2])
                out_list += list(range(start, end, step))
                out_list.append(int(end))

    return out_list


def print_split(split, desc, only_size=False):
    print()
    print(desc + ":")
    if not only_size:
        split_tmp = {}
        for seq, frames in split.items():
            split_tmp[seq] = list_to_desc(frames)
        pprint(split_tmp)

    ctr = 0
    for _, frames in split.items():
        ctr += len(frames)
    print("Total: %d" % ctr)


def write_split(split, path, all_seqs):
    with open(path, 'w') as out_file:
        for seq in all_seqs:
            out_file.write("%s\n" % list_to_desc(split[seq]))


all_seqs_path = osp.join(osp.dirname(osp.realpath(__file__)), "all_seqs.txt")
all_seqs = open(all_seqs_path).readlines()
all_seqs = [x[:-1] for x in all_seqs]

# Prepare Eigen Test Split:

original_eigen_split_test_path = osp.join(osp.dirname(osp.realpath(__file__)), "col_N.txt")
original_eigen_split_test_list = open(original_eigen_split_test_path).readlines()
original_eigen_split_test_list = [desc_to_list(x[:-1]) for x in original_eigen_split_test_list]
original_eigen_split_test = {}
for i, views in enumerate(original_eigen_split_test_list):
    original_eigen_split_test[all_seqs[i]] = views

print_split(original_eigen_split_test, "Original Eigen test split", only_size=True)

# Prepare Views where Densified Depth is available:

depth_from_single_view_benchmark_train_path = osp.join(osp.dirname(osp.realpath(__file__)), "col_K.txt")
depth_from_single_view_benchmark_train_list = open(depth_from_single_view_benchmark_train_path).readlines()
depth_from_single_view_benchmark_train_list = [desc_to_list(x[:-1]) for x in depth_from_single_view_benchmark_train_list]
depth_from_single_view_benchmark_train = {}
for i, views in enumerate(depth_from_single_view_benchmark_train_list):
    depth_from_single_view_benchmark_train[all_seqs[i]] = views
    
print_split(depth_from_single_view_benchmark_train, "Depth benchmark train", only_size=True)

depth_from_single_view_benchmark_val_path = osp.join(osp.dirname(osp.realpath(__file__)), "col_L.txt")
depth_from_single_view_benchmark_val_list = open(depth_from_single_view_benchmark_val_path).readlines()
depth_from_single_view_benchmark_val_list = [desc_to_list(x[:-1]) for x in depth_from_single_view_benchmark_val_list]
depth_from_single_view_benchmark_val = {}
for i, views in enumerate(depth_from_single_view_benchmark_val_list):
    depth_from_single_view_benchmark_val[all_seqs[i]] = views
    
print_split(depth_from_single_view_benchmark_val, "Depth benchmark val", only_size=True)

# Combine Eigen Test Split and Views where Densified Depth is available:
combined = {}
for seq in all_seqs:
    views = [x for x in original_eigen_split_test[seq] if (x in depth_from_single_view_benchmark_train[seq] or x in depth_from_single_view_benchmark_val[seq])]
    # use sth like this to only use samples where frames before/after are available:
#     views = [x for x in views if all([(x - y) in raw[seq] for y in range(1, 11)]) and all([(x + y) in raw[seq] for y in range(1, 11)])]
#     views = [x for x in views if all([(x - y) in odom_benchmark_train[seq] for y in range(1, 11)]) and all([(x + y) in odom_benchmark_train[seq] for y in range(1, 11)])]
    combined[seq] = views

print_split(combined, "Combined", only_size=True)
        
out_path = osp.join(osp.dirname(osp.realpath(__file__)), "split.txt")
with open(out_path, 'w') as file:
    for seq, views in combined.items():
        for view in views:
            file.write("%s; %s\n" % (seq, "; ".join([str(view)])))
