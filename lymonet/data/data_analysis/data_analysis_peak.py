"""
医学知识：
正常及发炎淋巴结：中心部为强回声的髓质，外围是低回声
转移淋巴结：髓质回声消失，呈单一的低回声或弱回声
"""

import os, sys
from pathlib import Path
import numpy as np
import cv2
from scipy.signal import find_peaks
import scipy.signal

import json
here = Path(__file__).parent

import matplotlib.pyplot as plt


class DataAnalysis():

    def __init__(self, data_path, train_test_val="test") -> None:
        self.dp = data_path
        self.train_test_val = train_test_val
        self.names = None

    def _save_img(self, img, name="img.png"):
        cv2.imwrite(f'{here}/{name}', img)
        pass

    def restore_org_img(self, img):
        mask = np.all(img != [128, 128, 128], axis=-1)  # 必须是三个通道都不是128才是mask
        coords = np.column_stack(np.where(mask))

        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)

        org_img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]
        return org_img
    
    def analysis_echo_peak(self, **kwargs):
        """
        分析回声峰
        """
        plot = kwargs.get("plot", False)
        train_test_val = self.train_test_val

        dpp = f"{self.dp}/{train_test_val}"
        self.names = os.listdir(dpp)
        peak_dict = dict()
        peak_dict["names"] = self.names

        for i, name in enumerate(self.names):
            print(f"processing {name}...")
            img_paths = [f"{dpp}/{name}/{x}" for x in os.listdir(f"{dpp}/{name}")]

            for j, img_path in enumerate(img_paths):
                assert os.path.exists(img_path), f"{img_path} not exists"
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                org_img = self.restore_org_img(img)
                org_img_gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

                # hist = cv2.calcHist([org_img_gray], [0], None, [256], [0, 256])  # 像素直方图
                # hist = hist.ravel()/hist.max()
                # if plot:
                    # plt.plot(hist)
                    # plt.savefig(f"{here}/{train_test_val}_{name}_{j}_hist.png")
                    # self._save_img(org_img, f"{train_test_val}_{name}_{j}.png")

                x_distri = np.sum(org_img_gray, axis=0)
                y_distri = np.sum(org_img_gray, axis=1)
                x_distri = x_distri.ravel()/x_distri.max()
                y_distri = y_distri.ravel()/y_distri.max()

                peaks, FHWM = self.find_peaks(x_distri)
                if len(peaks) > 0:
                    print(f"find peaks {i} {j} : {peaks}, FHWM: {FHWM}")
                    # 医学知识：正常及发炎淋巴结：中心部为强回声的髓质，外围是低回声；转移淋巴结：髓质回声消失，呈单一的低回声或弱回声
                    if plot:
                        plt.plot(x_distri)
                        plt.savefig(f"{here}/{train_test_val}_{name}_{j}_x_distri.png")
                        # 清空plot
                        plt.clf()
                        plt.plot(y_distri)
                        plt.savefig(f"{here}/{train_test_val}_{name}_{j}_y_distri.png")
                        assert name in ['inflammatory', "normal"], f"{name} not in ['inflammatory', 'normal']"
                        exit()
                else:
                    print(f"not find peaks {i} {j}")
                    assert name in ['metastatic'], f"{name} not in ['metastatic']"

                pass
                

    def find_peaks(self, data):
        peaks, _ = scipy.signal.find_peaks(data, prominence=0.1, height=0.2)
        FHWM = scipy.signal.peak_widths(data, peaks, rel_height=0.5)
        return peaks, FHWM
        
    def analysis_aspect_ratio(self, **kwargs):
        plot = kwargs.get("plot", False)
        train_test_val = self.train_test_val

        dpp = f"{self.dp}/{train_test_val}"
        self.names = os.listdir(dpp)
        asp_ratios = dict()
        asp_ratios["names"] = self.names

        for i, name in enumerate(self.names):
            print(f"processing {name}...")
            img_paths = [f"{dpp}/{name}/{x}" for x in os.listdir(f"{dpp}/{name}")]

            asp_ratios[name] = list()
            for j, img_path in enumerate(img_paths):
                assert os.path.exists(img_path), f"{img_path} not exists"
                img = cv2.imread(img_path)
                h, w, _ = img.shape

                org_img = self.restore_org_img(img) # 从正方形图像中恢复出原始图像
                org_h, org_w, _ = org_img.shape 
                # asp_ratio = org_h/org_w if org_h >= org_w else org_w/ org_h
                # asp_ratio = org_h/org_w
                asp_ratio = org_w/org_h

                asp_ratios[name].append(asp_ratio)

                if j == 0 and plot:
                    self._save_img(org_img, f"{train_test_val}_{name}_{asp_ratio}.png")
                    self._save_img(img, f"{train_test_val}_{name}_{asp_ratio}_resized.png")
                print(f'\r{j}/{len(img_paths)}', end='')
            print()
            pass

        # 保存数据
        with open(f"{here}/{train_test_val}_aspect_ratio.json", "w") as f:
            json.dump(asp_ratios, f)


    def read_and_plot(self):
        ttv = self.train_test_val
        import matplotlib.pyplot as plt
        with open(f"{here}/{ttv}_aspect_ratio.json", "r") as f:
            asp_ratios = json.load(f)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        names = asp_ratios["names"]
        fig, axes = plt.subplots(nrows=len(names), ncols=1, figsize=(10, 5 * len(names)))
        if len(names) == 1:
            axes = [axes]

        for ax, name, color in zip(axes, asp_ratios["names"], colors):
            data = asp_ratios[name]
            ax.hist(data, bins=70, density=False, label=name, color=color)
            ax.set_title(f"{name} aspect ratio")
            ax.set_xlabel("aspect ratio")
            # ax.set_xlim(0, 1.6)
            ax.set_xlim(0.5, 8)
            ax.set_ylabel("count")
        plt.legend()
        plt.savefig(f"{here}/{ttv}_aspect_ratio.png")
    
        pass



if __name__ == "__main__":
    data_path = "/data/tml/lymonet/lymo_yolo_square1"
    da = DataAnalysis(data_path, train_test_val="val")
    da.analysis_echo_peak(plot=True)
    da.read_and_plot()
    pass