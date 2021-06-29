# import pandas as pd
# import os
# import numpy as np
# import itertools

from Preprocessor import *
from DataMerge import *
from Ploter import *
import PySimpleGUI as sg


class Gui:
    def __init__(self):
        self._data_merger = DataMerge()
        self._pre = Preprocessor()
        self._pre.remove_bad_behaviour(self._pre.getDF(), update_self=True)
        self._plt = Ploter(self._pre.getDF())
        self._is_data_loaded = False

        self._window = None
        self._data_dir_path = None

        self._runApp()

    def _get_mouse_lst(self):
        # return self._pre.getUniqeList(MOUSE_NUM_COL_NAME)
        return self._pre.getUniqeList(MOUSE_ID)

    def _get_lvl_lst(self):
        # return self._pre.getUniqeList(LEVEL)
        return self._pre.getUniqeList(LEVEL_NAME)

    def _change_gui_element_visibility(self, key_lst, disable):
        for key in key_lst:
            self._window[key].update(disabled=disable)

    def _handleDisabledPlot(self, values):
        if values[BEHAVIOR_PIE_CBOX_KEY]:
            # disable rate plot and pie plot
            self._change_gui_element_visibility(
                [RATE_LIST_KEY, D_PRIME_CBOX_KEY], True)
            self._change_gui_element_visibility([BEHAVIOR_PIE_CBOX_KEY], False)
        elif values[RATE_LIST_KEY]:
            self._change_gui_element_visibility(
                [D_PRIME_CBOX_KEY, BEHAVIOR_PIE_CBOX_KEY], True)
            self._change_gui_element_visibility([RATE_LIST_KEY], False)
        elif values[D_PRIME_CBOX_KEY]:
            self._change_gui_element_visibility(
                [RATE_LIST_KEY, BEHAVIOR_PIE_CBOX_KEY], True)
            self._change_gui_element_visibility([D_PRIME_CBOX_KEY], False)

        else:
            self._change_gui_element_visibility(
                [RATE_LIST_KEY, BEHAVIOR_PIE_CBOX_KEY, D_PRIME_CBOX_KEY],
                False)

        self._window.refresh()

    # def _multiple_mouse_plot(self, rates_lst, selected_mouse_lst,
    #                          selected_level_lst):
    #
    #     for rate in rates_lst:
    #         if rate == HIT_GUI_LABEL:
    #             self._plt.rateMeanPlot(selected_mouse_lst, selected_level_lst,
    #                                    TPR)
    #         elif rate == MISS_GUI_LABEL:
    #             self._plt.rateMeanPlot(selected_mouse_lst, selected_level_lst,
    #                                    FPR)
    #         elif rate == FA_GUI_LABEL:
    #             self._plt.rateMeanPlot(selected_mouse_lst, selected_level_lst,
    #                                    FNR)
    #         elif rate == CR_GUI_LABEL:
    #             self._plt.rateMeanPlot(selected_mouse_lst, selected_level_lst,
    #                                    TNR)
    def _get_lvl_df_kind(self, no_go_df_in_use, go_df_in_use):
        if no_go_df_in_use and go_df_in_use:
            return self._plt.getAllScoreLst()
        elif go_df_in_use:
            return self._plt.getGoScoreLst()
        return self._plt.getNoGoScoreLst()

    def _select_rate_plot(self, rates_lst, selected_mouse_lst,
                          selected_level_lst, add_lvl_bins=False, do_plot=True,
                          bin_size=DEFAULT_BIN_SIZE):
        # multiple mice plot
        if len(selected_mouse_lst) > MIN_MOUSE_LEN_FOR_MEAN:
            self._plt.multiple_mice_rate_plot(rates_lst, selected_mouse_lst,
                                              selected_level_lst)

        # self._multiple_mouse_plot(rates_lst, selected_mouse_lst,
        # selected_level_lst)

        # one mouse for ploting
        else:
            go_df_in_use = False
            no_go_df_in_use = False
            for rate in rates_lst:
                if rate == HIT_GUI_LABEL:
                    go_df_in_use = True
                    self._plt.tprPlot(selected_mouse_lst, selected_level_lst,
                                      bin_size=bin_size)
                elif rate == MISS_GUI_LABEL:
                    go_df_in_use = True
                    self._plt.fnrPlot(selected_mouse_lst, selected_level_lst,
                                      bin_size=bin_size)

                elif rate == FA_GUI_LABEL:
                    no_go_df_in_use = True
                    self._plt.fprPlot(selected_mouse_lst, selected_level_lst,
                                      bin_size=bin_size)
                elif rate == CR_GUI_LABEL:
                    no_go_df_in_use = True
                    self._plt.tnrPlot(selected_mouse_lst, selected_level_lst,
                                      bin_size=bin_size)

            if add_lvl_bins:
                self._plt.plot_lvl_bins(selected_mouse_lst, selected_level_lst,
                                        self._get_lvl_df_kind(no_go_df_in_use,
                                                              go_df_in_use))

    def _update_gui_lists(self):
        mouse_lst = self._get_mouse_lst()
        self._window[MOUSE_SELECT_KEY].update(values=mouse_lst)
        # self._window[MOUSE_SELECT_KEY].update(values=self._get_mouse_lst())
        self._window[LEVEL_SELECT_KEY].update(values=self._get_lvl_lst())
        self._window[RATE_LIST_KEY].update(values=RATES_LIST)

        self._window.refresh()

    def _update_status_bar(self, msg, color, do_refresh=False):
        self._window[STATUS_BAR_KEY].update(msg, background_color=color)
        if do_refresh:
            self._window.refresh()

    def _handle_load_data(self, values):
        self._data_dir_path = values[DATA_PATH_KEY]
        self._data_merger.set_data_path(self._data_dir_path)
        self._update_status_bar(LOAD_DATA_MSG, "orange", True)
        try:
            df = self._data_merger.load_data()
            self._pre.set_data(df)
            self._plt.set_df(df)
            self._update_status_bar(DATA_LOADED, "green", True)
            self._is_data_loaded = True

            self._update_gui_lists()
            self._reset_all_ckbox()


        except Exception as e:
            # err_msg="{0}:{1}".format(type(e),e)
            # er="{0}".format(e)
            self._update_status_bar(e, "red")

    def _handle_set_default(self):
        if not self._is_data_loaded:
            self._update_status_bar(ERR_ENTER_DATA_PATH, "red")
        else:
            try:
                self._pre.saveData(DF_PICKL_EXTENSION)
            except Exception as e:
                self._update_status_bar(e, "red")

            self._update_status_bar(SET_DEFAULT_SUC_MSG, "green")

    def _handle_csv_download(self):
        if self._is_data_loaded:
            msg = CSV_POPUP + self._data_merger.save_to_csv()
            sg.popup_ok(msg, title=CSV_POPUP_TITLE)
        else:
            msg = ERR_SAVE_CSV
            color = "red"
            self._update_status_bar(msg, color, True)

    def _handle_select_all_lvl(self, is_selected):
        if is_selected:
            new_values = self._get_lvl_lst()
        else:
            new_values = []

        self._window[LEVEL_SELECT_KEY].set_value(new_values)
        self._window.refresh()

    def _handle_select_all_mice(self, is_selected):
        if is_selected:
            new_values = self._get_mouse_lst()
        else:
            new_values = []

        self._window[MOUSE_SELECT_KEY].set_value(new_values)
        self._window.refresh()

    def _disable_lvl_bins_ckbox(self, values):
        selected_mouse_lst = values[MOUSE_SELECT_KEY]
        if len(selected_mouse_lst) > MIN_SELECTED_MOUSE_FOR_LVL_BINS or values[
            BEHAVIOR_PIE_CBOX_KEY]:
            self._window[ADD_LVL_BINS_KEY].update(disabled=True)
        else:
            self._window[ADD_LVL_BINS_KEY].update(disabled=False)

        self._window.refresh()

    def _reset_all_ckbox(self):
        self._window[SLCT_ALL_LVLS].update(False)
        self._window[SLCT_ALL_MICE].update(False)
        self._window[ADD_LVL_BINS_KEY].update(False)

    def _get_status_bar_msg_and_color(self):
        if self._pre.is_default_data_loaded():
            return STATUS_BAR_DEFAULT_DATA, STATUS_BAR_DEFAULT_DATA_COLOR
        return STATUS_BAR_NO_DATA, STATUS_BAR_NO_DATA_COLOR

    def _get_plot_tile(self, selected_mouse_lst, selected_level_lst,
                       add_text=None):
        if add_text:
            title = add_text
        else:
            title = ""

        lvl_lst_len = len(selected_level_lst)
        if lvl_lst_len > TITLE_MAX_LINE_LEN:
            return title + BOLD_MICE_TITLE + MOUSE_LVL_TITLE_LONG_1.format(
                selected_mouse_lst) + BOLD_LVL_TITLE + MOUSE_LVL_TITLE_LONG_2.format(
                selected_level_lst[0: lvl_lst_len // 2],
                selected_level_lst[lvl_lst_len // 2: lvl_lst_len])

        return title + BOLD_MICE_TITLE + MOUSE_LVL_TITLE_SHORT_1.format(
            selected_mouse_lst) + BOLD_LVL_TITLE + MOUSE_LVL_TITLE_SHORT_2.format(
            selected_level_lst)

    def _select_plot_validation(self, is_pie_plot, is_rate_plot,
                                is_d_prime_plot, is_behaviour_analysis, is_mdp,
                                is_seq_hist):
        return is_pie_plot and is_rate_plot and is_d_prime_plot and is_behaviour_analysis and is_mdp and is_seq_hist

    def _runApp(self):
        # get data
        mouse_lst = self._get_mouse_lst()
        level_lst = self._get_lvl_lst()
        status_bar_msg_color = self._get_status_bar_msg_and_color()

        # window sections
        selectionT_lst_sg = [sg.Column([[sg.T(SELECT_MOUSE_LABEL)], [
            sg.Checkbox(SLCT_ALL, default=False, key=SLCT_ALL_MICE,
                        enable_events=True)]]),
                             sg.Listbox(values=mouse_lst, size=(14, 10),
                                        enable_events=True,
                                        bind_return_key=True,
                                        select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
                                        key=MOUSE_SELECT_KEY), sg.VSeparator(),
                             sg.Column([[sg.T(SELECT_LEVEL_LABEL)], [
                                 sg.Checkbox(SLCT_ALL, default=False,
                                             key=SLCT_ALL_LVLS,
                                             enable_events=True)], [
                                            sg.Checkbox(ADD_LVL_BINS,
                                                        default=False,
                                                        key=ADD_LVL_BINS_KEY,
                                                        enable_events=True)]]),

                             sg.Listbox(values=level_lst, size=(20, 10),
                                        enable_events=True,
                                        bind_return_key=True,
                                        select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
                                        key=LEVEL_SELECT_KEY)]

        # mouse_lvl_lsts_sg = [, ]
        #
        # level_lst_sg = [sg.T(SELECT_LEVEL_LABEL),
        #                 sg.Listbox(values=level_lst, size=(14, 10),
        #                            enable_events=True, bind_return_key=True,
        #                            select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
        #                            key=LEVEL_SELECT_KEY)]
        status_section_sg = [sg.T(STATUS_BAR_TEXT),
                             sg.StatusBar(status_bar_msg_color[0],
                                          key=STATUS_BAR_KEY,
                                          text_color="black", background_color=
                                          status_bar_msg_color[1],
                                          size=(50, 1), auto_size_text=True),
                             sg.Button(SET_DEFAULT_BUTTON_LABEL,
                                       key=SET_DEFAULT_BUTTON_KEY,
                                       tooltip=SET_DEFAULT_BUTTON_TOOLTIP)]
        data_path_section_sg = [sg.Text(ENTER_DATA_PATH),
                                sg.In(size=(25, 1), enable_events=True,
                                      key=DATA_PATH_KEY), sg.FolderBrowse(),
                                sg.B(LOAD_BTN_LABEL, key=LOAD_BTN_KEY),
                                sg.Button(button_color=sg.TRANSPARENT_BUTTON,
                                          image_filename=DOWNLOAD_CSV_IMG_PATH,
                                          border_width=0, image_subsample=2,
                                          tooltip=DOWNLOAD_TO_CSV_TOOLTIP,
                                          key=DOWNLOAD_TO_CSV_KEY)]
        # pkl_path_section_sg = [sg.Text(ENTER_PKL_PATH),
        #                        sg.In(size=(25, 1), enable_events=True,
        #                              key=PKL_PATH_KEY), sg.FolderBrowse(), ]
        # sg.FileBrowse(file_types=((FILE_TYPE_LABEL, FILE_TYPE_VALUE),))
        # sg.FolderBrowse(enable_events=True)
        d_prime_cbox_sg = [
            sg.Checkbox(D_PRIME_SELECTION_LABEL, enable_events=True,
                        default='0', key=D_PRIME_CBOX_KEY), ]
        behav_analysis_cbox_sg = [
            sg.Checkbox(BEHV_ANALYSIS_SELECTION_LABEL, enable_events=True,
                        default='0', key=BEHV_ANALYSIS_CBOX_KEY),
            sg.Checkbox(ROLLING_SUM_CHBOX_LABEL,
                        key=ROLLING_SUM_TOGGLE_BUT_KEY),
            sg.Checkbox(RUNS_HIST_CHCBX_LABEL, key=RUN_HIST_CHCBOX_KEY), ]

        mdp_cbox_sg = [
            sg.Checkbox(MDP_CBOX_LABEL, enable_events=True, default='0',
                        key=MDP_CBOX_KEY),
            sg.Checkbox(MDP_CBOX_ORDER_LABEL, enable_events=True, default='0',
                        key=MDP_CBOX_ORDER_KEY), sg.T(K_ORDER_SLIDER_LABEL),
            sg.Slider(range=(1, 5), orientation='v', size=(5, 20),
                      default_value=1, key=K_ORDER_SLIDER_KEY,
                      enable_events=True)]

        pie_selection_sg = [
            sg.Checkbox(BEHAVIOUR_PIE_CBOX_LABEL, enable_events=True,
                        default='0', key=BEHAVIOR_PIE_CBOX_KEY), ]
        rate_selection_sg = [sg.T(RATE_CBOX_LABEL),
                             sg.Listbox(values=RATES_LIST, size=(9, 4),
                                        enable_events=True,
                                        bind_return_key=True,
                                        select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
                                        key=RATE_LIST_KEY),
                             sg.T(BIN_SIZE_LABEL),
                             sg.Slider(range=(100, 900), orientation='v',
                                       size=(5, 20), default_value=500,
                                       key=BIN_SIZE_KEY, enable_events=True)]
        control_section_sg = [sg.Canvas(key=CONTROL_KEY)]
        #
        # data_column = [status_section_sg, data_path_section_sg, mouse_lst_sg,
        #                level_lst_sg, ]

        data_column = [status_section_sg, data_path_section_sg,
                       selectionT_lst_sg, [
                           sg.Text(PLOT_OPTIONS, font="Helvetica",
                                   background_color="black")],
                       pie_selection_sg, rate_selection_sg, d_prime_cbox_sg,
                       behav_analysis_cbox_sg, mdp_cbox_sg]

        figure_column = [
            sg.Canvas(key=FIG_KEY,  # it's important to set this size
                      size=(350 * 2, 350))]

        plot_column = [[sg.B(PLOT_BTN_LABEL, font='Any 15', key=PLOT_BTN_KEY,
                             button_color=('white', '#8FBC8F'), size=(5, 3))],
                       [sg.T(CONTROL_LABEL)], control_section_sg,
                       [sg.T(FIGURE_LABEL)], figure_column]

        # ----- Full layout -----
        layout = [[sg.Column(data_column), sg.VSeparator(),
                   sg.Column(plot_column), ], ]

        self._window = sg.Window(WINDOW_LABEL, layout,
                                 resizable=True).finalize()
        self._window.maximize()

        # set toggels
        rolling_sum_toggle_push = False
        # Run the Event Loop
        while True:
            event, values = self._window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            self._handleDisabledPlot(values)

            if event is SLCT_ALL_MICE:
                self._handle_select_all_mice(values[
                                                 SLCT_ALL_MICE])  # self._disable_lvl_bins_ckbox(values)

            # check if add level bins check box should be disabled
            self._disable_lvl_bins_ckbox(values)

            if event is SLCT_ALL_LVLS:
                self._handle_select_all_lvl(values[SLCT_ALL_LVLS])

            if event is DOWNLOAD_TO_CSV_KEY:
                self._handle_csv_download()

            if event is SET_DEFAULT_BUTTON_KEY:
                self._handle_set_default()

            if event is LOAD_BTN_KEY:
                self._handle_load_data(values)

            if event is PLOT_BTN_KEY:

                # get mouse and level lists, k_order
                k_order = int(values[K_ORDER_SLIDER_KEY])

                selected_mouse_lst = values[MOUSE_SELECT_KEY]
                selected_level_lst = values[LEVEL_SELECT_KEY]
                # get desired plot
                is_pie_plot = values[BEHAVIOR_PIE_CBOX_KEY]
                is_rate_plot = values[RATE_LIST_KEY]
                is_d_prime_plot = values[D_PRIME_CBOX_KEY]
                is_behaviour_analysis = values[BEHV_ANALYSIS_CBOX_KEY]
                is_mdp = values[MDP_CBOX_KEY]
                is_order_by = values[MDP_CBOX_ORDER_KEY]
                is_seq_hist = values[RUN_HIST_CHCBOX_KEY]

                # if no plot is selected
                if self._select_plot_validation(is_pie_plot, is_rate_plot,
                                                is_d_prime_plot,
                                                is_behaviour_analysis, is_mdp,
                                                is_seq_hist):
                    sg.popup_ok(SELECT_PLOT_MSG)

                elif not selected_level_lst:
                    sg.popup_ok(SELECT_LEVEL_MSG)
                elif not selected_mouse_lst:
                    sg.popup_ok(SELECT_MICE_MSG)

                # if one of the above is null print error
                elif is_pie_plot:
                    self._plt.setPiePlot(
                        self._get_plot_tile(selected_mouse_lst,
                                            selected_level_lst))
                    self._plt.stimPiePlot(selected_mouse_lst,
                                          selected_level_lst)
                    self._plt.showPlot(self._window, show_legend=False)

                elif is_rate_plot:

                    self._plt.setPlot(TRIALS_LABEL, PERCENTAGE,
                                      self._get_plot_tile(selected_mouse_lst,
                                                          selected_level_lst),
                                      setYRange=PERCENTAGE_RANGE)
                    # self._plt.setPlot(TRIALS_LABEL, PERCENTAGE,
                    #                   MOUSE_LVL_TITLE.format(
                    #                       selected_mouse_lst,
                    #                       selected_level_lst),
                    #                   setYRange=PERCENTAGE_RANGE)
                    self._window.refresh()
                    self._select_rate_plot(is_rate_plot, selected_mouse_lst,
                                           selected_level_lst,
                                           values[ADD_LVL_BINS_KEY],
                                           bin_size=values[BIN_SIZE_KEY])
                    self._plt.showPlot(self._window)

                elif is_d_prime_plot:
                    self._plt.setPlot(TRIALS_LABEL, D_PRIME_LABEL,
                                      self._get_plot_tile(selected_mouse_lst,
                                                          selected_level_lst))
                    self._plt.d_prime_plot(selected_mouse_lst,
                                           selected_level_lst)

                    self._plt.showPlot(self._window, show_legend=False)
                elif is_behaviour_analysis:
                    if values[ROLLING_SUM_TOGGLE_BUT_KEY]:
                        y_label = BEHEV_ANALYSIS_Y_ROL_SUM
                    else:
                        y_label = BEHEV_ANALYSIS_Y
                    self._plt.setPlot(BEHEV_ANALYSIS_X, y_label,
                                      self._get_plot_tile(selected_mouse_lst,
                                                          selected_level_lst, ))
                    self._plt.plot_behav_analysis(selected_mouse_lst,
                                                  selected_level_lst, values[
                                                      ROLLING_SUM_TOGGLE_BUT_KEY])

                    self._plt.showPlot(self._window, show_legend=True,
                                       legend_loc='upper right')

                elif is_mdp:
                    # k_order = int(values[K_ORDER_SLIDER_KEY])
                    self._plt.set_heat_map_plot(MDP_X, MDP_Y,
                                                self._get_plot_tile(
                                                    selected_mouse_lst,
                                                    selected_level_lst))

                    self._plt.plot_MDP_heat_map(selected_mouse_lst,
                                                selected_level_lst,
                                                k_order=k_order, range=None,
                                                do_animation=False,
                                                order_by_certainty=is_order_by)
                    # train vector range = (6164,7978)
                    self._plt.showPlot(self._window, show_legend=False)

                elif is_seq_hist:

                    # self._plt.states_hist(selected_mouse_lst,selected_level_lst,k_order)
                    # self._plt.showPlot(self._window, show_legend=False)

                    self._plt.setPlot(SEQ_HIST_X, SEQ_HIST_Y,
                                      self._get_plot_tile(selected_mouse_lst,
                                                          selected_level_lst, ))
                    self._plt.behav_runs_hist(selected_mouse_lst,
                                              selected_level_lst)

                    self._plt.showPlot(self._window, show_legend=True,
                                       legend_loc='upper right')

        self._window.close()


if __name__ == '__main__':
    Gui()
