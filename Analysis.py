import numpy as np
from scipy.spatial.distance import cosine
from Constants import *
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


class Analysis:
    def __init__(self, df=None):
        self._df = df

    def plot_3d(self, all_x_y_z, train_vec_x_y_z, graph_title):
        '''
        plot points in 3D
        :param all_x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
        coordinate)
        :param graph_title: title for plotted graph
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(all_x_y_z[0], all_x_y_z[1], all_x_y_z[2], s=1, marker='.',
                   depthshade=False, color='blue', alpha=0.5)
        ax.scatter(train_vec_x_y_z[0], train_vec_x_y_z[1], train_vec_x_y_z[2],
                   s=20, marker='o', depthshade=False, color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(graph_title)
        plt.show()

    def get_X_from_scores(self, score_arr, rolling_window=True,
                          vector_len=DEFAULT_VEC_LEN):
        if rolling_window:
            X = self.scores_2_train_vec(score_arr, vector_len)
        else:
            start_index = len(score_arr) % vector_len
            X = np.array(score_arr[start_index:])
            X = X.reshape((int(X.shape[0] / vector_len), vector_len))
        return X

    def run_pca_model(self, X, train_vec, n_components=3, test_size=0.3):
        train_X, test_X = train_test_split(X, test_size=test_size)

        # train
        pca = PCA(n_components=n_components)
        pca.fit(train_X)
        # test score
        score = pca.score(test_X)
        print("pcs score: ", score)

        # get new coordinates
        principal_components = pca.fit_transform(X)
        train_vec_new_coor = pca.transform(train_vec)

        self.plot_3d(principal_components.T, train_vec_new_coor.T, "pca")
        return principal_components

    def calc_distribution(self, train_vec):
        hit = np.sum(train_vec == HIT_SCROE)
        miss = np.sum(train_vec == MISS_SCORE)
        fa = np.sum(train_vec == FA_SCORE)
        cr = np.sum(train_vec == CR_SCORE)
        dist_vec = np.array([[hit, miss, fa, cr]])
        return dist_vec / len(train_vec)

    def compare_distribution(self, train_vec_lst, score_lst):
        print("training vectors distribution")
        print("[hit,miss,FA,CR]")
        for train_vec in train_vec_lst:
            print(self.calc_distribution(train_vec))
        vec_mat = np.zeros((2000, 4))
        for i in range(2000):
            rand_index_1 = int(
                np.random.choice(score_lst.shape[0] - 1000, 1, replace=False))

            rand_vec_1 = score_lst[rand_index_1:(rand_index_1 + 1000)]

            vec_mat[i] = self.calc_distribution(rand_vec_1)
        mean_rand_vec = vec_mat.mean(axis=0)
        print(
            '2000 random vectors:\n mean hit:{0}, mean miss:{1}, mean FA {2}, mean CR {3}'.format(
                format(mean_rand_vec[0], '.3f'),
                format(mean_rand_vec[1], '.3f'),
                format(mean_rand_vec[2], '.3f'),
                format(mean_rand_vec[3], '.3f')))

    def find_runs(self, x):
        """Find runs of consecutive items in an array."""

        # ensure array
        x = np.asanyarray(x)
        if x.ndim != 1:
            raise ValueError('only 1D array supported')
        n = x.shape[0]

        # handle empty array
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        else:
            # find run starts
            loc_run_start = np.empty(n, dtype=bool)
            loc_run_start[0] = True
            np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
            run_starts = np.nonzero(loc_run_start)[0]

            # find run values
            run_values = x[loc_run_start]

            # find run lengths
            run_lengths = np.diff(np.append(run_starts, n))

            return run_values, run_starts, run_lengths

    def fit_size(self, vec1, vec2):
        max_arg = np.argmax([vec1.shape[0], vec2.shape[0]])
        if max_arg:
            return vec1, vec2[vec2.shape[0] - vec1.shape[0]:]
        return vec1[vec1.shape[0] - vec2.shape[0]:], vec2

    def random_vectors_cosin(self):
        l = []
        for i in range(1000):
            a = np.random.rand(1000)
            b = np.random.rand(1000)
            l.append(cosine(a, b))
        plt.scatter(np.arange(0, len(l)), l, alpha=0.5)
        plt.show()

    def plot_rand_cosin_from_score(self, score_lst):
        cos_lst = []
        for i in range(2000):
            rand_index_1 = int(
                np.random.choice(score_lst.shape[0] - 1000, 1, replace=False))
            rand_index_2 = int(
                np.random.choice(score_lst.shape[0] - 1000, 1, replace=False))

            rand_vec_1 = score_lst[rand_index_1:(rand_index_1 + 1000)]
            rand_vec_2 = score_lst[rand_index_2:(rand_index_2 + 1000)]

            cos_lst.append(cosine(rand_vec_1, rand_vec_2))
        plt.scatter(np.arange(0, len(cos_lst)), cos_lst, alpha=0.5)
        plt.show()

    def expand_sequences(self, start, stop, run_starts, score_arr):
        shifted_stop = min(len(run_starts) - 1, stop + ROLLING_SUM_WINDOW_SIZE)
        shifted_stop = min(run_starts[shifted_stop], len(score_arr) - 1)
        return score_arr[run_starts[start]:shifted_stop]

    def scores_2_train_vec(self, score_arr,
                           window_size=ROLLING_SUM_WINDOW_SIZE):
        shape = score_arr.shape[:-1] + (
            score_arr.shape[-1] - window_size + 1, window_size)
        strides = score_arr.strides + (score_arr.strides[-1],)
        return np.lib.stride_tricks.as_strided(score_arr, shape=shape,
                                               strides=strides)

    def calc_train_vector_cosin(self, train_vec_mat):
        cosin_lst = []
        for i in range(train_vec_mat.shape[0] - 1):
            current_cosin_lst = []
            for j in range(train_vec_mat.shape[0] - 1):
                if j == i:
                    continue
                current_cosin_lst.append(
                    1 - cosine(train_vec_mat[i], train_vec_mat[j]))
            cosin_lst.append(current_cosin_lst)
        np.save("cosin", train_vec_mat)

    def normalize_run_lengths(self, run_lengths, score_arr, run_values,
                              soft_max=False):
        """
        divide good runs(Hit & CR) by total amount of good trails
        divide bad runs(Miss & FA) by total amount of bad trails
        multiply bad runs by -1
        multiply all rund by multiple_factor (10^4) or run softmax
        """
        if soft_max:
            run_lengths = softmax(run_lengths, axis=0)
            a = np.where(run_lengths != 0)
            run_lengths[~run_values] *= -1
            return run_lengths

        good_behave_total_len = np.sum(score_arr == HIT_SCROE) + np.sum(
            score_arr == CR_SCORE)
        bad_behave_total_len = len(score_arr) - good_behave_total_len
        # check if first run is good or bad behavior
        if run_values[0]:
            d = np.ones(run_lengths.shape)
            d[::2] *= good_behave_total_len
            run_lengths = run_lengths / d

            d = np.ones(run_lengths.shape)
            d[1::2] *= bad_behave_total_len
            run_lengths = run_lengths / d
        else:
            d = np.ones(run_lengths.shape)
            d[::2] *= bad_behave_total_len
            run_lengths = run_lengths / d

            d = np.ones(run_lengths.shape)
            d[1::2] *= good_behave_total_len
            run_lengths = run_lengths / d

        # Normalize
        run_lengths *= NORMA_MUL_FACTOR
        return run_lengths

    def run_lengths_hist(self, score_arr):
        mask_go = (score_arr == HIT_SCROE) | (score_arr == CR_SCORE)
        run_values, run_starts, run_lengths = self.find_runs(mask_go)

        # check if first run is good or bad behavior
        if run_values[0]:
            pos_behav_lengths = run_lengths[::2]
            neg_behav_lengths = run_lengths[1::2]
        else:
            pos_behav_lengths = run_lengths[1::2]
            neg_behav_lengths = run_lengths[::2]
        return pos_behav_lengths, neg_behav_lengths

    def get_2D_behaviour_analysis(self, df=None, rolling_sum=False):
        plot_lst = []
        if df is None:
            df = self._df
        score_arr = (df[SCORE]).to_numpy()
        mask_go = (score_arr == HIT_SCROE) | (score_arr == CR_SCORE)   
        run_values, run_starts, run_lengths = self.find_runs(mask_go)

        # turn FA,Miss points to negative points
        run_lengths[~run_values] *= -1

        if rolling_sum:
            rolling_sum_run_lengths = np.convolve(run_lengths, np.ones(
                ROLLING_SUM_WINDOW_SIZE, dtype=int), mode='same')

            plot_lst.append((np.cumsum(run_lengths), rolling_sum_run_lengths,
                             ROLLING_SUM_PLOT_LABEL, '-', None))


            return plot_lst

        local_peaks, _ = find_peaks(run_lengths, height=(10, None))
        # peaks, _ = find_peaks(run_lengths, height=0.4)
        peaks, _ = find_peaks(run_lengths)
        # ,distance=1800
        top_peaks = np.percentile(run_lengths[peaks], 95)

        # new_peaks, _ = find_peaks(run_lengths, height=(top_peaks, None),
        #                           distance=1800)
        new_peaks, _ = find_peaks(run_lengths, height=(top_peaks, None),
                                  distance=1800)


        y = run_lengths
        x = np.cumsum(y)
        pos_run_lengths = run_lengths.copy()
        neg_run_lengths = run_lengths.copy()
        pos_run_lengths[pos_run_lengths <= 0] = 0
        neg_run_lengths[neg_run_lengths > 0] = 0
        plot_lst.append((x, pos_run_lengths, "pos", False, 'green'))
        plot_lst.append((x, neg_run_lengths, "neg", False, 'red'))

        return plot_lst
