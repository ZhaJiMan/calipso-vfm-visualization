'''
https://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_summaries/vfm/
https://hdfeos.org/zoo/LaRC/CAL_LID_L2_VFM-ValStage1-V3-41.2021-11-29T23-32-39ZD.hdf.v.py
'''

import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC

class VfmReader:
    '''
    读取CALIPSO L2 VFM产品的类.

    Attributes
    ----------
    lon : (nrec,) ndarray
        激光足迹的经度.

    lat : (nrec,) ndarray
        激光足迹的纬度.

    time : (nrec,) DatetimeIndex
        激光足迹对应的UTC时间.

    height : (545,) ndarray
        廓线每个bin对应的高度, 单位为km.
        注意存在三种垂直分辨率.

    fcf : (nrec, 545, 7) ndarray
        解码后的Feature_Classification_Flags.
        7对应于文档中7个字段的值.
    '''
    def __init__(self, filepath):
        self.sd = SD(str(filepath), SDC.READ)

    def close(self):
        '''关闭文件.'''
        self.sd.end()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def lon(self):
        return self.sd.select('Longitude')[:, 0]

    @property
    def lat(self):
        return self.sd.select('Latitude')[:, 0]

    @property
    def time(self):
        # 时间用浮点型的yymmdd.ffffffff表示.
        yymmddff = self.sd.select('Profile_UTC_Time')[:, 0]
        yymmdd = (yymmddff + 2e7).astype(int).astype(str)
        yymmdd = pd.to_datetime(yymmdd, format='%Y%m%d')
        ff = pd.to_timedelta(yymmddff % 1, unit='D')
        time = yymmdd + ff

        return time

    @property
    def height(self):
        height1 = (np.arange(290) + 0.5) * 0.03 - 0.5
        height2 = (np.arange(200) + 0.5) * 0.06 + 8.2
        height3 = (np.arange(55) + 0.5) * 0.18 + 20.2
        height = np.concatenate([height1, height2, height3])

        return height

    @property
    def fcf(self):
        # 三个高度层中都只选取第一条廓线来代表5km水平分辨率的FCF.
        fcf = self.sd.select('Feature_Classification_Flags')[:]
        fcf1 = fcf[:, 1165:1455]
        fcf2 = fcf[:, 165:365]
        fcf3 = fcf[:, 0:55]
        fcf = np.hstack([fcf3, fcf2, fcf1])[:, ::-1]

        # 利用位运算进行解码.
        shifts = [0, 3, 5, 7, 9, 12, 13]
        bits = [7, 3, 3, 3, 7, 1, 7]
        fcf = fcf[:, :, None] >> shifts & bits

        return fcf