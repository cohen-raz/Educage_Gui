import os
from Constants import *
import pandas as pd


class DataMerge:
    def __init__(self, data_path=None):
        self._full_data = None
        self._data_path = data_path

    def set_data_path(self, data_path):
        self._data_path = data_path

    def load_data(self):
        if not self._data_path:
            raise Exception(ERR_ENTER_DATA_PATH)
        if not self.path_validation():
            raise Exception(ERR_INPUT_FOLDER)
        return self.dir_to_df()

    def path_validation(self):
        self._list_dir = os.listdir(self._data_path)
        return os.path.exists(self._data_path) and self._list_dir

    def load_data_throws(self, txt_path):
        try:
            with open(txt_path, 'r') as f:
                file_lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(ERR_INPUT_TXT_FILE)
        try:

            return pd.DataFrame([string.split(';') for string in file_lines])

        except:
            raise Exception(ERR_SPLIT_TXT_FILE)

    def _load_data(self, txt_path):
        with open(txt_path, 'r') as f:
            file_lines = f.readlines()
        return pd.DataFrame([line.split(';') for line in file_lines])

    def set_col_as_numeric(self, df, col_name):
        df[col_name] = pd.to_numeric(df[col_name])
        return df

    def _set_as_numeric(self):
        self._full_data = self.set_col_as_numeric(self._full_data, SCORE)
        self._full_data = self.set_col_as_numeric(self._full_data, LEVEL)

    def get_mouse_data(self, mouse_dict, mouse_counter):
        ir_path = mouse_dict[IR]
        lick_path = mouse_dict[LICK]
        mosix_path = mouse_dict[MOSIX]

        df_lick = self._load_data(ir_path)
        df_lick.rename(columns={0: TIME, 1: MOUSE_ID, 2: "LICK"}, inplace=True)

        df_ir = self._load_data(lick_path)
        df_ir.rename(columns={0: TIME, 1: MOUSE_ID, 2: "IR"}, inplace=True)

        df_mosix = self._load_data(mosix_path)
        df_mosix.rename(
            columns={0: MOUSE_ID, 1: LEVEL, 2: TIME, 3: "?", 4: "Level name",
                     5: SCORE}, inplace=True)

        df_mosix["LICK"] = df_lick["LICK"]
        df_mosix["IR"] = df_ir["IR"]

        df_mosix[MOUSE_NUM_COL_NAME] = mouse_counter
        return df_mosix

    def _is_log_file(self, f_name_lst):
        if not len(f_name_lst) == LOG_FILE_LEN:
            return False
        return f_name_lst[1] in LOG_FILE_LST

    def _add_file_to_dir_dict(self, dir_dict, f_name_lst, f_path):
        # at this point, f_name_lst len is 2
        if f_name_lst[1] not in LOG_FILE_LST:
            return None
        f_name = f_name_lst[0]
        f_extension = f_name_lst[1]
        if f_name in dir_dict:
            mouse_dict = dir_dict[f_name]
            mouse_dict[f_extension] = f_path
        else:
            dir_dict[f_name] = {f_extension: f_path}
        return dir_dict

    def dir_to_dict(self):
        """
        :return: dictionary: {mouse_id:[file_path.IR,file_path.Lck,file_path.mosix]
        """
        if not self._data_path:
            return None

        dir_dict = {}
        for full_name in self._list_dir:
            f_path = os.path.join(self._data_path, full_name)
            if os.path.isdir(f_path):
                # skip directories
                continue
            else:
                f_name_lst = full_name.split(sep=".")
                if not self._is_log_file(f_name_lst):
                    continue
                dir_dict = self._add_file_to_dir_dict(dir_dict, f_name_lst,
                                                      f_path)

        return dir_dict

    # def _df_lst_to_df(self, all_mice_df_lst):
    #     # complete_df = all_mice_df_lst[0]
    #     # for df_idx in range(1, len(all_mice_df_lst)):
    #     #     complete_df = complete_df.append(all_mice_df_lst[df_idx],
    #     #                                      ignore_index=True)
    #     # return complete_df
    #     return pd.concat(all_mice_df_lst)

    def _df_lst_to_df(self, all_mice_df_lst):
        if not all_mice_df_lst:
            raise Exception(ERR_NO_DATA)
        if len(all_mice_df_lst) > MIN_LEN_FOR_CONCAT:
            self._full_data = pd.concat(all_mice_df_lst)
        else:
            self._full_data = all_mice_df_lst[0]

    def dir_to_df(self):
        all_mice_df_lst = []
        dir_dict = self.dir_to_dict()
        mouse_counter = 0
        for mouse in dir_dict:
            mouse_counter += 1
            mouse_dict = dir_dict[mouse]
            m_df = self.get_mouse_data(mouse_dict, mouse_counter)
            all_mice_df_lst.append(m_df)
        self._df_lst_to_df(all_mice_df_lst)
        self._set_as_numeric()
        return self._full_data

    def _get_csv_path(self, path):
        return path  + CSV_EXTENSION

    def save_to_csv(self, path=None):
        """
        saves full df to .scv in given path. if path is None,saves to given dir
        """
        if self._full_data is None:
            return False
        if not path:
            file_path = self._get_csv_path(self._data_path)
            self._full_data.to_csv(file_path)
            return file_path
        else:
            self._full_data.to_csv(path)
            return path

