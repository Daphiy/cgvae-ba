import numpy as np
from collections import defaultdict
from .utils import *

def process_raw_graphs(self, raw_data, is_training_data, file_name, bucket_sizes=None):
    if bucket_sizes is None:
        bucket_sizes = dataset_info(self.params["dataset"])["bucket_sizes"]
    incremental_results, raw_data = self.calculate_incremental_results(raw_data, bucket_sizes, file_name)
    bucketed = defaultdict(list)
    x_dim = len(raw_data[0]["node_features"][0])

    for d, (
    incremental_adj_mat, distance_to_others, node_sequence, edge_type_masks, edge_type_labels, local_stop, edge_masks,
    edge_labels, overlapped_edge_features) \
            in zip(raw_data, incremental_results):
        # choose a bucket
        chosen_bucket_idx = np.argmax(bucket_sizes > max([v for e in d['graph']
                                                          for v in [e[0], e[2]]]))
        chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
        # total number of nodes in this data point
        n_active_nodes = len(d["node_features"])
        bucketed[chosen_bucket_idx].append({
            'adj_mat': graph_to_adj_mat(d['graph'], chosen_bucket_size, self.num_edge_types,
                                        self.params['tie_fwd_bkwd']),
            'incre_adj_mat': incremental_adj_mat,
            'distance_to_others': distance_to_others,
            'overlapped_edge_features': overlapped_edge_features,
            'node_sequence': node_sequence,
            'edge_type_masks': edge_type_masks,
            'edge_type_labels': edge_type_labels,
            'edge_masks': edge_masks,
            'edge_labels': edge_labels,
            'local_stop': local_stop,
            'number_iteration': len(local_stop),
            'init': d["node_features"] + [[0 for _ in range(x_dim)] for __ in
                                          range(chosen_bucket_size - n_active_nodes)],
            'labels': [d["targets"][task_id][0] for task_id in self.params['task_ids']],
            'mask': [1. for _ in range(n_active_nodes)] + [0. for _ in range(chosen_bucket_size - n_active_nodes)]
        })

    if is_training_data:
        for (bucket_idx, bucket) in bucketed.items():
            np.random.shuffle(bucket)
            for task_id in self.params['task_ids']:
                task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
                if task_sample_ratio is not None:
                    ex_to_sample = int(len(bucket) * task_sample_ratio)
                    for ex_id in range(ex_to_sample, len(bucket)):
                        bucket[ex_id]['labels'][task_id] = None

    bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.params['batch_size'])]
                      for bucket_idx, bucket_data in bucketed.items()]
    bucket_at_step = [x for y in bucket_at_step for x in y]

    return (bucketed, bucket_sizes, bucket_at_step)

def pad_annotations(self, annotations):
    return np.pad(annotations,
                  pad_width=[[0, 0], [0, 0], [0, self.params['hidden_size'] - self.params["num_symbols"]]],
                  mode='constant')

def make_batch(self, elements, maximum_vertice_num):
    # get maximum number of iterations in this batch. used to control while_loop
    max_iteration_num = -1
    for d in elements:
        max_iteration_num = max(d['number_iteration'], max_iteration_num)
    batch_data = {'adj_mat': [], 'init': [], 'labels': [], 'edge_type_masks': [], 'edge_type_labels': [],
                  'edge_masks': [],
                  'edge_labels': [], 'node_mask': [], 'task_masks': [], 'node_sequence': [],
                  'iteration_mask': [], 'local_stop': [], 'incre_adj_mat': [], 'distance_to_others': [],
                  'max_iteration_num': max_iteration_num, 'overlapped_edge_features': []}
    for d in elements:
        # sparse to dense for saving memory
        incre_adj_mat = incre_adj_mat_to_dense(d['incre_adj_mat'], self.num_edge_types, maximum_vertice_num)
        distance_to_others = distance_to_others_dense(d['distance_to_others'], maximum_vertice_num)
        overlapped_edge_features = overlapped_edge_features_to_dense(d['overlapped_edge_features'], maximum_vertice_num)
        node_sequence = node_sequence_to_dense(d['node_sequence'], maximum_vertice_num)
        edge_type_masks = edge_type_masks_to_dense(d['edge_type_masks'], maximum_vertice_num, self.num_edge_types)
        edge_type_labels = edge_type_labels_to_dense(d['edge_type_labels'], maximum_vertice_num, self.num_edge_types)
        edge_masks = edge_masks_to_dense(d['edge_masks'], maximum_vertice_num)
        edge_labels = edge_labels_to_dense(d['edge_labels'], maximum_vertice_num)

        batch_data['adj_mat'].append(d['adj_mat'])
        batch_data['init'].append(d['init'])
        batch_data['node_mask'].append(d['mask'])

        batch_data['incre_adj_mat'].append(incre_adj_mat +
                                           [np.zeros((self.num_edge_types, maximum_vertice_num, maximum_vertice_num))
                                            for _ in range(max_iteration_num - d['number_iteration'])])
        batch_data['distance_to_others'].append(distance_to_others +
                                                [np.zeros((maximum_vertice_num))
                                                 for _ in range(max_iteration_num - d['number_iteration'])])
        batch_data['overlapped_edge_features'].append(overlapped_edge_features +
                                                      [np.zeros((maximum_vertice_num))
                                                       for _ in range(max_iteration_num - d['number_iteration'])])
        batch_data['node_sequence'].append(node_sequence +
                                           [np.zeros((maximum_vertice_num))
                                            for _ in range(max_iteration_num - d['number_iteration'])])
        batch_data['edge_type_masks'].append(edge_type_masks +
                                             [np.zeros((self.num_edge_types, maximum_vertice_num))
                                              for _ in range(max_iteration_num - d['number_iteration'])])
        batch_data['edge_masks'].append(edge_masks +
                                        [np.zeros((maximum_vertice_num))
                                         for _ in range(max_iteration_num - d['number_iteration'])])
        batch_data['edge_type_labels'].append(edge_type_labels +
                                              [np.zeros((self.num_edge_types, maximum_vertice_num))
                                               for _ in range(max_iteration_num - d['number_iteration'])])
        batch_data['edge_labels'].append(edge_labels +
                                         [np.zeros((maximum_vertice_num))
                                          for _ in range(max_iteration_num - d['number_iteration'])])
        batch_data['iteration_mask'].append([1 for _ in range(d['number_iteration'])] +
                                            [0 for _ in range(max_iteration_num - d['number_iteration'])])
        batch_data['local_stop'].append([int(s) for s in d["local_stop"]] +
                                        [0 for _ in range(max_iteration_num - d['number_iteration'])])

        target_task_values = []
        target_task_mask = []
        for target_val in d['labels']:
            if target_val is None:  # This is one of the examples we didn't sample...
                target_task_values.append(0.)
                target_task_mask.append(0.)
            else:
                target_task_values.append(target_val)
                target_task_mask.append(1.)
        batch_data['labels'].append(target_task_values)
        batch_data['task_masks'].append(target_task_mask)

    return batch_data