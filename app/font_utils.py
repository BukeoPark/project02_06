# requires: matplotlib
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

FONT_PATH = os.path.join(os.path.dirname(__file__), "assets", "NanumGothic-Regular.ttf")


def apply_korean_font():
    fm.fontManager.addfont(FONT_PATH)
    font_name = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
