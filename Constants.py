# Default Path
DATA_PATH = ''# **<write here the path for the log files>**
DEFAULT_DF_PCKL_PATH = ''# **<write here the path for a df containing the data, saved as .pkl (not mandatory)>**
DF_PICKL_EXTENSION = "df.pkl"
PICKL_EXTENSION = ".pkl"
DOWNLOAD_CSV_IMG_PATH = ".\download_csv_img.png"

# Plot Related (Titles,Labels,Ranges,style)
# Titles
D_PRIME_LABEL = "d'"
TPR_TITLE = "Hit rate (TPR) for mouse:{0}, at level:{1}"
FNR_TITLE = "False Negative rate (FNR) for mouse:{0}, at level:{1}"
TNR_TITLE = "Hit rate (TPR) for mouse:{0}, at level:{1}"
TNR_TILE = "CR rate (TNR) for mouse:{0}, at level:{1}"
FNR_LABEL = "Miss Rate"
TRIALS_LABEL = "Trials"
TPR_LABEl = "Hit Rate"
TNR_LABEL = "CR Rate"
FPR_LABEL = "FA Rate"
HIT_MISS_LABEL = "Hit and Miss"
TPR_ERR_BAR_LABEL = "Mean Hit rate"
FP_ERR_BAR_LABEL = "Mean FA rate"
FN_ERR_BAR_LABEL = "Mean Miss Rate"
TNR_ERR_BAR_LABEL = "Mean CR Rate"
BOLD_MICE_TITLE = r"$\bf{Mice: }$"
BOLD_LVL_TITLE = r"$\bf{Levels: }$"

MOUSE_LVL_TITLE_LONG_1 = "{0}\n"
MOUSE_LVL_TITLE_LONG_2 = "{0},\n{1}"

MOUSE_LVL_TITLE_SHORT_1 = "{0}\n"
MOUSE_LVL_TITLE_SHORT_2 = "{0}"

STIM_HIST_LABEL = "StimId"
PERCENTAGE = "%"
NO_GO_HIST_TITLE = "NoGo stimuli analysis"
MOUSE_SCORE_PIE = "Score pie:\n"
PERCENTAGE_RANGE = [0, 1]
DOT_lINE = '--'
SEMI_DOT_LINE = ":"
DOWNLOAD_TO_CSV_TOOLTIP = "save as .csv"
BEHEV_ANALYSIS_Y = "Sequences Length"
BEHEV_ANALYSIS_Y_ROL_SUM = "Sequences Length Sum"
BEHEV_ANALYSIS_ROLLING = "Sum of Sequences"
BEHEV_ANALYSIS_X = TRIALS_LABEL
BEHEV_ANALYSIS_LABEL = "behavior sequences"
ROLLING_SUM_CHBOX_LABEL = "use rolling sum"
ROLLING_SUM_PLOT_LABEL = "rolling sum"
RUNS_HIST_CHCBX_LABEL = "sequences histogram"
BEHEV_ANALYSIS_MAXMIA_LABEL = "Local Max"
PERCENTILE_LABEL = "Local Max\n95 percentile"
MDP_Y = "previous state"
MDP_X = "next state"
SEQ_HIST_X = "sequence length"
SEQ_HIST_Y = "%"
MDP_COLOR_BAR_LABEL = 'Transition probability'

# Rate Coding
TPR = 0
FNR = 1
FPR = 2
TNR = 3
MAX_SCORE = 3
SCORE_DICT = {0: "Hit", 1: "FA", 2: "Miss", 3: "CR"}
SCORES_LST = ["Hit", "FA", "Miss", "CR"]
# Score Value
HIT_SCROE = 0
FA_SCORE = 1
MISS_SCORE = 2
CR_SCORE = 3
DEFAULT_BIN_SIZE = 500

# General
INIT_SUM = 0
MIN_LEN_FOR_CONCAT = 1
CSV_EXTENSION = "/data.csv"
CSV_POPUP = "saved to: "
CSV_POPUP_TITLE = "data has been saved successfully"
NORMA_MUL_FACTOR = 10 ** 4
ROLLING_SUM_WINDOW_SIZE = 100
DEFAULT_VEC_LEN = 100

# Load data
IR = "IR"
LICK = "Lck"
MOSIX = "mosix"
LOG_FILE_LST = [IR, LICK, MOSIX]
LOG_FILE_LEN = 2

# Columns names
STIM_ID = "stimID"
MOUSE_NUM_COL_NAME = "mouse_num"
MOUSE_ID = "mouse_name"
LEVEL = "level"
SCORE = "score"
FREQ = "freq_played"
TIME = "time"
LEVEL_NAME = "Level name"

# Error Messages
ERR_PICKLE_OVERRIDE = "Warning: changes override existing pickle"
ERR_INPUT_FOLDER = "Folder is empty or not exists"
ERR_INPUT_TXT_FILE = "File not exists"
ERR_SPLIT_TXT_FILE = "Converting txt data failed"
ERR_ENTER_DATA_PATH = "Please select a directory"
ERR_NO_DATA = "There are no log files"
ERR_SAVE_CSV = "Please load data, and then try again."
# Gui related constants

# Labels
ENTER_DATA_PATH = "Data Path (.csv):"
ENTER_PKL_PATH = "Fast Loading (.pkl):"
WINDOW_LABEL = "Educage Analysis "
SELECT_MOUSE_LABEL = "Select one or more mice:"
SELECT_LEVEL_LABEL = "Select one or more levels:"
PLOT_OPTIONS = "Select Desired Plot"
BEHAVIOUR_PIE_CBOX_LABEL = "Behavior Pie"
D_PRIME_SELECTION_LABEL = "d'"
BEHV_ANALYSIS_SELECTION_LABEL = "2D behavior analysis"
MDP_CBOX_LABEL = "MDP heat map"
MDP_CBOX_ORDER_LABEL = "order by\n certainty"
TOGGLE_LABEL_ON = "on"
TOGGLE_LABEL_OFF = "off"
RATE_CBOX_LABEL = "Behavior Rates"
CONTROL_LABEL = "Controls:"
FIGURE_LABEL = "Figure:"
PLOT_BTN_LABEL = "Plot"
SELECT_LEVEL_MSG = "Please select levels"
SELECT_MICE_MSG = "Please select mice"
SELECT_PLOT_MSG = "Please select a plot"
FILE_TYPE_LABEL = "CSV,PKL Files"
STATUS_BAR_TEXT = "Data Status:"
LOAD_BTN_LABEL = "Load"
SET_DEFAULT_BUTTON_LABEL = "Set Default"
SET_DEFAULT_BUTTON_TOOLTIP = "Load current data faster when reloading the GUI"
SLCT_ALL = "Select All"
ADD_LVL_BINS = "Add Level Bins"
BIN_SIZE_LABEL = "Bin size"
K_ORDER_SLIDER_LABEL = "Chain Order"

# STATUS MSG
LOAD_DATA_MSG = "Loading, please wait"
DATA_LOADED = "Data loaded successfully"
STATUS_BAR_DEFAULT_DATA = "default data loaded"
STATUS_BAR_DEFAULT_DATA_COLOR = "#8FBC8F"
STATUS_BAR_NO_DATA = "Please load data"
STATUS_BAR_NO_DATA_COLOR = "red"
SET_DEFAULT_SUC_MSG = "Current data set as default"

# Keys
DATA_PATH_KEY = "-Data Path Folder-"
PKL_PATH_KEY = "-PKL Path Folder-"
MOUSE_SELECT_KEY = "-Mouse select Key-"
LEVEL_SELECT_KEY = "-Level select Key-"
BEHAVIOR_PIE_CBOX_KEY = "-Pie cBox Key-"
D_PRIME_CBOX_KEY = "-d prime key-"
BEHV_ANALYSIS_CBOX_KEY = "-reduced behavior key-"
MDP_CBOX_KEY = "--MDP CBOX KEY"
MDP_CBOX_ORDER_KEY = "- MDP CBOX ORDER BY KEY -"
K_ORDER_SLIDER_KEY = "-MDP_K_ORDER_KEY-"
ROLLING_SUM_TOGGLE_BUT_KEY = "-rolling sum toggle key-"
RUN_HIST_CHCBOX_KEY = "-run histogram check box key-"
RATE_LIST_KEY = "-Rate cBox Key-"
CONTROL_KEY = "-Controls_Key-"
PLOT_BTN_KEY = "-Plot Btn Key-"
FIG_KEY = "-Fig Key-"
STG_KEY = "-Fig Key-"
STATUS_BAR_KEY = "-Status Bar Key-"
LOAD_BTN_KEY = "-Load Btn key-"
SET_DEFAULT_BUTTON_KEY = "-set default data key-"
DOWNLOAD_TO_CSV_KEY = "-save as .csv key-"
SLCT_ALL_LVLS = "-select all lvls key-"
SLCT_ALL_MICE = "-select all mice key-"
ADD_LVL_BINS_KEY = "-add level bins key-"
BIN_SIZE_KEY = "-bin size key-"

# Values
HIT_GUI_LABEL = "Hit Rate"
MISS_GUI_LABEL = "Miss Rate"
FA_GUI_LABEL = "FA Rate"
CR_GUI_LABEL = "CR Rate"
RATES_LIST = [HIT_GUI_LABEL, MISS_GUI_LABEL, FA_GUI_LABEL, CR_GUI_LABEL]
MIN_MOUSE_LEN_FOR_MEAN = 1
FILE_TYPE_VALUE = ["*.csv", "*.pkl"]
MIN_SELECTED_MOUSE_FOR_LVL_BINS = 1
MIN_LVL_BIN = 0
MAX_LVL_BIN = 1
TITLE_FONT_SIZE = 8.5
TITLE_MAX_LINE_LEN = 3
D_PRIME_ADJUSTMENT = 1

# MDP
DEF_STEP_BACK = 1
STATE_DIST_LABEL = "distribution"
ROW_LABEL_IDX = 0
COL_LABEL_IDX = 1
K_ORDER_TO_MAT_SIZE = {3: 25, 4: 20, 5: 15}
CERTAINTY_THRESHOLD = 0.25
