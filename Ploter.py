from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from Constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
    NavigationToolbar2Tk
import itertools
from Analysis import *
import pandas as pd
from MDP import *
import matplotlib.animation as animation
from collections import Counter


# ------------------- docked plot connection -------------------------------

class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


# ------------------- docked plot connection -------------------------------

class Ploter:
    def __init__(self, df):
        self._df = df
        self._analysis = Analysis(df)
        self._mdp = Mdp()

    def _initPlot(self):
        self._fig = plt.figure()
        self._gcf = plt.gcf()
        self._ax = self._fig.add_subplot(111)

    def set_df(self, df):
        self._df = df

    def setPiePlot(self, title):
        self._initPlot()
        self._ax.set_title(title, fontsize=TITLE_FONT_SIZE)

    def setPlot(self, xLabel, yLabel, title, lineWidth=1.5, LineStyle='-',
                setXRange=None, setYRange=None):

        self._initPlot()
        plt.rcParams['lines.linewidth'] = lineWidth
        plt.rcParams['lines.linestyle'] = LineStyle

        self._ax.set_xlabel(xLabel)
        self._ax.set_ylabel(yLabel)
        self._ax.set_title(title, fontsize=TITLE_FONT_SIZE)

        if setYRange:
            plt.ylim(setYRange)
        if setXRange:
            plt.xlim(setXRange)

    def set_heat_map_plot(self, xLabel, yLabel, title):
        self._initPlot()

        self._ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        self._ax.xaxis.tick_top()
        self._ax.xaxis.set_label_position('top')
        self._ax.set_xlabel(xLabel)
        self._ax.set_ylabel(yLabel)

    def plotXY(self, x, y, label, add_sign=False, color=None):
        if color is None:
            if add_sign:
                plt.plot(x, y, add_sign, label=label)
            else:
                plt.plot(x, y, label=label)
        else:
            if add_sign:
                plt.plot(x, y, add_sign, label=label, color=color)
            else:
                plt.plot(x, y, label=label, color=color)

    def plot_heat_map(self, mat, labels, color_bar_label=None):
        plt.imshow(mat, cmap='summer', aspect='auto')
        if color_bar_label is not None:
            plt.colorbar(label=color_bar_label,
                         ticks=[i / 20 for i in range(1, 11)])

        # text inside cells
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                plt.text(x, y, '%.4f' % mat[y, x],
                         horizontalalignment='center',
                         verticalalignment='center', )

        # ticks labels
        row_labels = labels[ROW_LABEL_IDX]
        col_labels = labels[COL_LABEL_IDX]

        self._ax.set_xticks(np.arange(len(col_labels)))
        self._ax.set_xticklabels(col_labels)

        self._ax.set_yticks(np.arange(len(row_labels)))
        self._ax.set_yticklabels(row_labels)
        plt.tight_layout()

    def _heat_map_labels(self, k_order):
        x_lst = SCORES_LST.copy()
        keys, y_lst = self._mdp.get_k_order_keys(k_order, get_labels=True)
        return x_lst, y_lst

    def _general_rate_calc(self, rate_kind, score_lst, bin_size):
        lst = score_lst == rate_kind
        lst = lst.astype(np.int16)
        return lst.rolling(int(bin_size), win_type='triang', center=True,
                           min_periods=int(bin_size // 2)).mean()

    def _get_rate_lst(self, partial_df, rate_kind, bin_size=DEFAULT_BIN_SIZE):
        """
        :param partial_df: data frame of all selected levels, selected mice,
                            go or no go score
        """
        levels = partial_df[LEVEL].unique()
        cumulative_lst = np.array([])
        for lvl in levels:
            df_by_lvl = partial_df.loc[partial_df[LEVEL] == lvl][SCORE]
            rate_lst = self._general_rate_calc(rate_kind, df_by_lvl, bin_size)
            cumulative_lst = np.append(cumulative_lst, rate_lst)
        return cumulative_lst

    def plot_lvl_bins(self, selected_mouse_lst, selected_level_lst, df_kind):
        partial_df = self._getPartialData(selected_mouse_lst,
                                          selected_level_lst, df_kind,
                                          use_score=True)
        df_lvl = partial_df[LEVEL]

        unique_lvls = df_lvl.unique()
        lvl_bins = np.searchsorted(df_lvl, unique_lvls)
        lvl_bins = np.append(lvl_bins, len(df_lvl))
        plt.vlines(lvl_bins, MIN_LVL_BIN, MAX_LVL_BIN, linestyles="dashed",
                   label="Level Bins", colors="PURPLE", linewidth=0.7)

    def tprPlot(self, mouseNumLst, lvlNumList, do_plot=True,
                bin_size=DEFAULT_BIN_SIZE):
        partialDf = self._getPartialData(mouseNumLst, lvlNumList,
                                         self.getGoScoreLst())

        y = self._get_rate_lst(partialDf, HIT_SCROE, bin_size)

        x = np.arange(len(y))

        if do_plot:
            self.plotXY(x, y, TPR_LABEl)
        else:
            return y

    def _adjust_len_lst_of_lst(self, lst_of_lst):
        shortest_length = min(map(len, lst_of_lst))
        return [l[:shortest_length] for l in lst_of_lst]

    def _toPaddedMat(self, lst_of_lst):
        mat = np.array(
            list(itertools.zip_longest(*lst_of_lst, fillvalue=-1))).T
        return mat

    def rateMeanPlot(self, comulativeSumLst, partialDf, label):

        if partialDf.empty:
            x = []
            mean = []
            std = []

        else:
            timex = self.getTimecol(partialDf)
            x = np.arange(len(timex))
            # adjust list len for each list in comulativeSumLst
            comulativeSumMat = self._toPaddedMat(comulativeSumLst)
            # adjust x(Time) axis
            x = x[:min(np.size(comulativeSumMat, 1), len(x))]

            # calc mean of each col for relevant cells
            mask = comulativeSumMat != -1
            toMeanMat = np.multiply(mask, comulativeSumMat)
            mean = np.true_divide(toMeanMat.sum(0),
                                  (comulativeSumMat != -1).sum(0))

            # change empty cells to col mean
            inds = np.where(comulativeSumMat <= -1)
            comulativeSumMat[inds] = np.take(mean, inds[1])

            std = np.std(comulativeSumMat, axis=0)

        plt.errorbar(x, mean, label=label, yerr=std, fmt='--.', elinewidth=0.5,
                     errorevery=250, capsize=1)


    def multiple_mice_rate_plot(self, rates_lst, selected_mouse_lst,
                                selected_level_lst):

        # get partial data
        partial_df = self._getPartialData(selected_mouse_lst,
                                          selected_level_lst, [], False)

        partial_go_df = partial_df.loc[
            partial_df[SCORE].isin(self.getGoScoreLst())]

        partial_no_go_df = partial_df.loc[
            partial_df[SCORE].isin(self.getNoGoScoreLst())]

        temp_df_lst_go = []
        temp_df_lst_no_go = []

        for mouse in selected_mouse_lst:
            temp_df_lst_go.append(
                partial_go_df.loc[partial_go_df[MOUSE_ID] == mouse])
            temp_df_lst_no_go.append(
                partial_no_go_df.loc[partial_no_go_df[MOUSE_ID] == mouse])

        # call specific rate function and get comulativeSumLst
        for rate in rates_lst:
            if rate == HIT_GUI_LABEL:
                label = TPR_ERR_BAR_LABEL
                partial_df = partial_go_df
                comulativeSumLst = self._tprMeanPlot(selected_mouse_lst,
                                                     temp_df_lst_go)

            elif rate == MISS_GUI_LABEL:
                label = FN_ERR_BAR_LABEL
                partial_df = partial_go_df
                comulativeSumLst = self._fnrMeanPlot(selected_mouse_lst,
                                                     temp_df_lst_go)

            elif rate == FA_GUI_LABEL:
                label = FP_ERR_BAR_LABEL
                partial_df = partial_no_go_df
                comulativeSumLst = self._fprMeanPlot(selected_mouse_lst,
                                                     temp_df_lst_no_go)


            elif rate == CR_GUI_LABEL:
                label = TNR_ERR_BAR_LABEL
                partial_df = partial_no_go_df
                comulativeSumLst = self._tnrMeanPlot(selected_mouse_lst,
                                                     temp_df_lst_no_go)

            self.rateMeanPlot(comulativeSumLst, partial_df, label)

    def _tprMeanPlot(self, mouseNumLst, df_lst):
        comulativeSumLst = []
        for tempDf in df_lst:
            if not tempDf.empty:
                comulativeSumLst.append(self._getComulativeTPR(tempDf[SCORE]))
        return comulativeSumLst

    def _fnrMeanPlot(self, mouseNumLst, df_lst):
        comulativeSumLst = []
        for tempDf in df_lst:
            if not tempDf.empty:
                comulativeSumLst.append(self._getComulativeFnr(tempDf[SCORE]))
        return comulativeSumLst

    def _fprMeanPlot(self, mouseNumLst, df_lst):
        comulativeSumLst = []
        for tempDf in df_lst:
            if not tempDf.empty:
                comulativeSumLst.append(self._getComulativeFpr(tempDf[SCORE]))
        return comulativeSumLst

    def _tnrMeanPlot(self, mouseNumLst, df_lst):
        comulativeSumLst = []
        for tempDf in df_lst:
            if not tempDf.empty:
                comulativeSumLst.append(self._getComulativeTnr(tempDf[SCORE]))
        return comulativeSumLst

    def _getPartialData(self, mouseNumLst, lvlNumList, scoreList=None,
                        use_score=True):
        partialDf = self._df.loc[(self._df[MOUSE_ID].isin(mouseNumLst))]
        if use_score:
            partialDf = partialDf.loc[partialDf[SCORE].isin(scoreList)]
        partialDf = partialDf.loc[partialDf[LEVEL_NAME].isin(lvlNumList)]

        return partialDf

    def getTimecol(self, partialDf):
        return partialDf[TIME]

    def getAllScoreLst(self):
        return [HIT_SCROE, FA_SCORE, MISS_SCORE, CR_SCORE]

    def getGoScoreLst(self):
        return [HIT_SCROE, MISS_SCORE]

    def getNoGoScoreLst(self):
        return [FA_SCORE, CR_SCORE]

    def _calcTpr(self, tp, p):
        if not p:
            return 0
        return tp / p

    def _getComulativeTPR(self, goScoreLst):
        # hit amount
        p = INIT_SUM
        # Go amount
        tp = INIT_SUM

        comulativTpr = []
        for score in goScoreLst:
            if score == HIT_SCROE:
                tp += 1
            p += 1
            comulativTpr.append(self._calcTpr(tp, p))
        return comulativTpr

    def _getComulativeFnr(self, goScoreLst):
        tprLst = self._getComulativeTPR(goScoreLst)
        return [1 - tpr for tpr in tprLst]

    def _getComulativeFpr(self, NoGoScoreLst):
        tnrLst = self._getComulativeTnr(NoGoScoreLst)
        return [1 - tnr for tnr in tnrLst]

    def fnrPlot(self, mouseNumLst, lvlNumList, bin_size=DEFAULT_BIN_SIZE):
        partialDf = self._getPartialData(mouseNumLst, lvlNumList,
                                         self.getGoScoreLst())

        y = 1 - self._get_rate_lst(partialDf, HIT_SCROE, bin_size)
        x = np.arange(len(y))
        self.plotXY(x, y, FNR_LABEL)

    def _getComulativeTnr(self, noGoScore):
        # hit amount
        n = INIT_SUM
        # Go amount
        tn = INIT_SUM

        comulativTnr = []
        for score in noGoScore:
            if score == CR_SCORE:
                tn += 1
            n += 1
            comulativTnr.append(self._calcTpr(tn, n))
        return comulativTnr


    def tnrPlot(self, mouseNumLst, lvlNumList, bin_size=DEFAULT_BIN_SIZE):
        partialDf = self._getPartialData(mouseNumLst, lvlNumList,
                                         self.getNoGoScoreLst())

        y = self._get_rate_lst(partialDf, CR_SCORE, bin_size)

        x = np.arange(len(y))
        self.plotXY(x, y, TNR_LABEL)


    def fprPlot(self, mouseNumLst, lvlNumList, do_plot=True,
                bin_size=DEFAULT_BIN_SIZE):
        partialDf = self._getPartialData(mouseNumLst, lvlNumList,
                                         self.getNoGoScoreLst())

        y = 1 - self._get_rate_lst(partialDf, CR_SCORE, bin_size)
        x = np.arange(len(y))

        if do_plot:
            self.plotXY(x, y, FPR_LABEL)
        else:
            return y

    def _getDataByScore(self, scoreLst):
        return self._df.loc[self._df[SCORE].isin(scoreLst)]

    def _geDataByStimId(self, stimID, data):
        tempDf = data.loc[data[STIM_ID] == stimID]
        noGoLen = len(tempDf)
        return tempDf, noGoLen

    def _getNoGoScoreCountByStimID(self, stimID, score, noGoData):
        noGoDataByid, noGoLen = self._geDataByStimId(stimID, noGoData)
        if not noGoLen:
            return 0
        return len(noGoDataByid.loc[noGoDataByid[SCORE] == score]) / noGoLen

    def noGoExplorePlot(self):
        stimIdLst = self._df[STIM_ID].unique().tolist()
        noGoData = self._getDataByScore([FA_SCORE, CR_SCORE])
        x1 = []
        x2 = []
        for stimId in stimIdLst:
            x1.append(
                self._getNoGoScoreCountByStimID(stimId, FA_SCORE, noGoData))
            x2.append(
                self._getNoGoScoreCountByStimID(stimId, CR_SCORE, noGoData))

        n = np.asarray(stimIdLst)
        self._ax.bar(n, x1, color='b', width=0.25, label=FNR_LABEL)
        self._ax.bar(n + 0.25, x2, color='g', width=0.25, label=TNR_LABEL)

    def stimPiePlot(self, mouseLst=None, lvlLst=None):
        if not mouseLst:
            mouseLst = self._df[MOUSE_NUM_COL_NAME].unique().tolist()
        if not lvlLst:
            lvlLst = self._df[LEVEL].unique().tolist()

        scoreLst = self._df[SCORE].unique().tolist()
        df = self._getPartialData(mouseLst, lvlLst, scoreLst)
        pieSumLst = []
        pieLabelLst = []

        score_dict_keys = SCORE_DICT.keys()
        for score in scoreLst:
            if score not in score_dict_keys:
                continue
            pieSumLst.append(len(df.loc[df[SCORE] == score]))
            pieLabelLst.append(SCORE_DICT[score])
        self._ax.pie(pieSumLst, labels=pieLabelLst, autopct='%1.2f%%')

    def _d_prime_adjust_amount(self, rate_lst_len, replace_hit):
        if not rate_lst_len:
            return D_PRIME_ADJUSTMENT
        if replace_hit:
            return D_PRIME_ADJUSTMENT - (D_PRIME_ADJUSTMENT / rate_lst_len)
        return D_PRIME_ADJUSTMENT / rate_lst_len

    def _fix_rate_lst(self, rate_lst, replace_hit):
        # neither  hit rate or fa rate can be 0 or 1
        adjustment = self._d_prime_adjust_amount(len(rate_lst), replace_hit)
        rate_lst = np.asarray(rate_lst)
        rate_lst[rate_lst == 1] = adjustment
        rate_lst[rate_lst == 0] = adjustment
        return rate_lst

    def _adjust_lst_len(self, lst_a, lst_b):
        if len(lst_a) > len(lst_b):
            return lst_a[0: len(lst_b)], lst_b
        return lst_a, lst_b[0: len(lst_a)]

    def d_prime_plot(self, selected_mouse_lst, selected_level_lst):
        hit_rate_lst = self.tprPlot(selected_mouse_lst, selected_level_lst,
                                    False)
        fa_rate_lst = self.fprPlot(selected_mouse_lst, selected_level_lst,
                                   False)

        hit_rate_lst = self._fix_rate_lst(hit_rate_lst, True)
        fa_rate_lst = self._fix_rate_lst(fa_rate_lst, False)

        ppf_hit_lst = norm.ppf(hit_rate_lst)
        ppf_fa_lst = norm.ppf(fa_rate_lst)

        ppf_hit_lst, ppf_fa_lst = self._adjust_lst_len(ppf_hit_lst, ppf_fa_lst)

        d_prime = ppf_hit_lst - ppf_fa_lst

        x = np.arange(len(d_prime))
        self.plotXY(x, d_prime, label=None)

    def _calc_mean_dict(self, dict, n):
        for key in dict.keys():
            dict[key] = dict[key] / n
        return dict

    def behav_runs_hist(self, selected_mouse_lst, selected_level_lst):
        partialDf = self._getPartialData(selected_mouse_lst,
                                         selected_level_lst, use_score=False)
        score_arr = (partialDf[SCORE]).to_numpy()

        cur_pos_lst, cur_neg_lst = self._analysis.run_lengths_hist(score_arr)

        bin_max = max(np.max(cur_pos_lst), np.max(cur_neg_lst))
        bins = np.arange(1, bin_max, 1)

        plt.hist([cur_pos_lst, cur_neg_lst], bins=bins, alpha=0.5,
                 color=["blue", "red"], density=True,
                 label=["positive behaviour", "negative behaviour"])


    def plot_behav_analysis(self, selected_mouse_lst, selected_level_lst,
                            rolling_sum):

        partialDf = self._getPartialData(selected_mouse_lst,
                                         selected_level_lst, use_score=False)
        plot_lst = self._analysis.get_2D_behaviour_analysis(partialDf,
                                                            rolling_sum)

        for plot in plot_lst:
            self.plotXY(plot[0], plot[1], plot[2], plot[3], color=plot[4])

    def states_hist(self, selected_mouse_lst, selected_level_lst, k_order):
        partialDf = self._getPartialData(selected_mouse_lst,
                                         selected_level_lst, use_score=False)
        prev_states, prev_states_dist = self._mdp.states_hist(
            partialDf[SCORE].to_numpy(), k_order)

        self._initPlot()
        plt.hist(prev_states_dist, label="test")
        plt.tight_layout()

    def plot_MDP_heat_map(self, selected_mouse_lst, selected_level_lst,
                          k_order, range=None, do_animation=False,
                          order_by_certainty=False):
        heat_mat_lst = []
        row_labels = []
        col_labels = []
        labels_flag = True

        if do_animation:
            for i, lvl in enumerate(selected_level_lst):
                partialDf = self._getPartialData(selected_mouse_lst, [lvl],
                                                 use_score=False)
                score_arr = (partialDf[SCORE]).to_numpy()
                current_mat = self._mdp.get_transition_mat_k_order(score_arr,
                                                                   k_order)
                heat_mat_lst.append(
                    [self._ax.imshow(current_mat, animated=True)])
                plt.pause(0.1)
                if not i:
                    self._ax.imshow(current_mat)
            ani = animation.ArtistAnimation(self._fig, heat_mat_lst,
                                            interval=50, blit=True,
                                            repeat_delay=1000)  # plt.show()

        for mouse in selected_mouse_lst:
            partialDf = self._getPartialData([mouse], selected_level_lst,
                                             use_score=False)

            score_arr = (partialDf[SCORE]).to_numpy()
            np.save("_scores.npy",score_arr)

            if range is not None:
                score_arr = score_arr[range[0]:range[1]]

            df = self._mdp.get_transition_mat_k_order(score_arr, k_order,
                                                      order_by_certainty)
            if labels_flag:
                row_labels = list(df.index)
                col_labels = list(df.columns)
                labels_flag = False

            heat_mat_lst.append(df.to_numpy())

        mean_heat_map = np.mean(heat_mat_lst, axis=0)

        self.plot_heat_map(mean_heat_map, color_bar_label=MDP_COLOR_BAR_LABEL,
                           labels=[row_labels, col_labels])

    # ------------------- docked plot connection -------------------------------

    def _draw_figure_w_toolbar(self, canvas, fig, canvas_toolbar):
        if canvas.children:
            for child in canvas.winfo_children():
                child.destroy()
        if canvas_toolbar.children:
            for child in canvas_toolbar.winfo_children():
                child.destroy()
        figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
        figure_canvas_agg.draw()
        toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
        toolbar.update()
        figure_canvas_agg.get_tk_widget().pack(side='right', fill='both',
                                               expand=1)

    # ------------------- docked plot connection -------------------------------

    def showPlot(self, docked_window=None, show_legend=True, legend_loc=None):
        if show_legend:
            if legend_loc is not None:
                self._ax.legend(loc=legend_loc)
            else:
                self._ax.legend()

        if docked_window:
            DPI = self._gcf.get_dpi()
            self._gcf.set_size_inches(600 * 2 / float(DPI), 600 / float(DPI))
            self._draw_figure_w_toolbar(docked_window[FIG_KEY].TKCanvas,
                                        self._fig,
                                        docked_window[CONTROL_KEY].TKCanvas)

        else:
            plt.show()
