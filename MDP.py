import numpy as np
from Constants import *
from itertools import product as comb
import pandas as pd


class Mdp:

    def test_k_order(self, index, score_arr, k, score_tup):
        if k > 1:
            t = score_tup[::-1]
        else:
            t = [score_tup]
        j = 0
        for i in range(1, k + 1):
            index_c = index - i
            d = score_arr[index_c] == t[j]

            s = np.sum(d)
            j += 1
            if s != len(index):
                exit()

    def print_info(self, heat_mat, prev_states, next_states, k_order,
                   prev_states_dist):
        if k_order < 3:
            return
        prev_states_dist_array = np.array(prev_states_dist)
        max_prev_states_ind = prev_states_dist_array.argsort()[-4:][::-1]

        max_rows = []
        prev_states = np.array(prev_states)
        for ind in max_prev_states_ind:
            max_rows.append(heat_mat[ind, :])
        max_heat_mat = np.vstack(max_rows)

        max_prev_states = prev_states[max_prev_states_ind]
        states_index = []
        for state in max_prev_states:
            states_index.append(self.key_to_label(tuple(state)))

        max_df = pd.DataFrame(max_heat_mat, columns=SCORES_LST,
                              index=states_index)

        print("4 most common states:")
        print(max_df.to_string())
        return max_df

    def keys_lst_to_labels(self, keys):
        labels = []
        for key in keys:
            labels.append(self.key_to_label(key))
        return labels

    def key_to_label(self, key):
        if type(key) != tuple:
            return SCORE_DICT[key]
        label = ""
        for score in key:
            label += SCORE_DICT[score] + "->"
        return label

    def get_k_order_keys(self, k_order, get_labels=False):
        if k_order == 1:
            keys = SCORE_DICT.copy()
        else:
            keys = comb(SCORE_DICT.keys(), repeat=k_order)
        keys = sorted(keys)
        labels = []
        if get_labels:
            for key in keys:
                labels.append(self.key_to_label(key))
            return keys, labels

        return keys

    def get_next_indx_by_score_k_order(self, score_arr, score_tup, k_order):
        # convert score tup to iterable (handle the case: k_order=0)
        if type(score_tup) != tuple:
            prev_states = [score_tup]
        else:
            prev_states = list(score_tup)

        # ignore last x trails in calculations (x= k_order+1) so return indexes
        # will be in bounds
        scores = score_arr[:len(score_arr) - (k_order + 1)]

        next_index = np.where(scores == prev_states[0])[0]
        if len(prev_states) == 1:
            # adjust index to point at the states after the previous state
            return next_index + k_order

        prev_states = prev_states[1:]
        logics_arr = []
        next_index += 1
        for i, state in enumerate(prev_states):
            cur_next_index = next_index + i
            logics_arr.append(score_arr[cur_next_index] == state)

        score_tup_index = np.ones(len(logics_arr[0]))
        for i in range(len(logics_arr)):
            score_tup_index = np.logical_and(score_tup_index, logics_arr[i])

        # adjust index to point at the states after the previous state
        return next_index[score_tup_index] + k_order - 1


    def get_percentage_k_dict(self, score_arr, next_states, next_index):
        transition_percentage_dict = {}
        row_sum = len(next_index)

        for state in next_states:
            if not row_sum:
                transition_percentage_dict[state] = 0
            else:
                transition_percentage_dict[state] = np.sum(
                    score_arr[next_index] == state) / row_sum

        return transition_percentage_dict

    def _get_orderd_values(self, row_dict, next_states):
        values = []
        for state in next_states:
            values.append(row_dict[state])
        return np.array(values)

    def get_transition_mat_k_order(self, score_arr, k_order,
                                   order_by_certainty=False):
        prev_states, labels = self.get_k_order_keys(k_order, get_labels=True)
        next_states = sorted(list(SCORE_DICT.keys()))
        row_percentage_lst = []
        prev_states_dist = []
        for score_tup in prev_states:
            next_index = self.get_next_indx_by_score_k_order(score_arr,
                                                             score_tup,
                                                             k_order)
            prev_states_dist.append(len(next_index))

            row_dict = self.get_percentage_k_dict(score_arr, next_states,
                                                  next_index)
            current_row = self._get_orderd_values(row_dict, next_states)
            row_percentage_lst.append(current_row)

        mdp_mat = np.around(np.vstack(row_percentage_lst), 3)


        if order_by_certainty:
            return self._order_by_certainty(mdp_mat, prev_states_dist, labels)

        # add last col := prev state distribution
        prev_states_dist = np.array(prev_states_dist).reshape(mdp_mat.shape[0],
                                                              1)
        mdp_mat = np.append(mdp_mat, prev_states_dist, axis=1)
        # create df from data
        row_labels = labels
        col_labels = list(SCORE_DICT.values())
        col_labels.append(STATE_DIST_LABEL)
        mdp_df = pd.DataFrame(data=mdp_mat, index=row_labels,
                              columns=col_labels)
        # order by last col- decending
        mdp_df = mdp_df.sort_values(mdp_df.columns[-1], ascending=False)
        # remove lst col
        mdp_df = mdp_df.drop(columns=[STATE_DIST_LABEL], axis=1)

        if k_order > 2:
            return mdp_df.head(K_ORDER_TO_MAT_SIZE[k_order])

        return mdp_df

    def states_hist(self, score_arr, k_order):
        prev_states = self.get_k_order_keys(k_order)
        prev_states_dist = []
        for score_tup in prev_states:
            next_index = self.get_next_indx_by_score_k_order(score_arr,
                                                             score_tup,
                                                             k_order)
            prev_states_dist.append(len(next_index))
        prev_states = self.keys_lst_to_labels(prev_states)
        return prev_states, prev_states_dist

    def _order_by_certainty(self, mdp_mat, prev_states_dist, row_labels,
                            threshold=CERTAINTY_THRESHOLD):
        row_labels = np.array(row_labels)
        prev_states_dist = np.array(prev_states_dist)
        # normalize
        prev_states_dist = prev_states_dist / np.sum(prev_states_dist)
        max_label = row_labels[np.argmax(prev_states_dist)]
        max_p = np.max(prev_states_dist)
        print("most common state: {0}, p={1}".format(max_label, max_p))
        col_labels = list(SCORE_DICT.values())
        col_labels.append(STATE_DIST_LABEL)

        # take rows where CR and Hit above threshold
        hit_threshold_ind = mdp_mat[:, 0] > threshold
        cr_threshold_ind = mdp_mat[:, 3] > threshold
        high_certainty_ind = hit_threshold_ind & cr_threshold_ind
        high_certainty_mat = mdp_mat[high_certainty_ind]

        # take relevant labels and distribution
        row_labels = row_labels[high_certainty_ind]
        prev_states_dist = prev_states_dist[high_certainty_ind].reshape(
            high_certainty_mat.shape[0], 1)

        # add last col := prev state distribution
        high_certainty_mat = np.append(high_certainty_mat, prev_states_dist,
                                       axis=1)

        # order by last col- decending
        high_certainty_df = pd.DataFrame(data=high_certainty_mat,
                                         index=row_labels, columns=col_labels)
        high_certainty_df = high_certainty_df.sort_values(
            high_certainty_df.columns[-1], ascending=False)

        # return top 20 rows
        return high_certainty_df.head(20)


