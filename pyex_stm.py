# -*- utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
import pyex_seg as ps
import pyex_lib as pl

import warnings
warnings.simplefilter('error')


# test Score model
def exp_scoredf_normal(mean=70, std=10, maxscore=100, minscore=0, samples=100000):
    return pd.DataFrame({'sf': [max(minscore, min(int(np.random.randn(1) * std + mean), maxscore), -x)
                         for x in range(samples)]})


def test_model(name='plt', df=None, fieldnames='sf', decimals=0):
    if type(df) != pd.DataFrame:
        scoredf = exp_scoredf_normal()
    else:
        scoredf = df
    if name == 'plt':
        pltmodel = PltScoreModel()
        # rawpoints = [0, 0.023, 0.169, 0.50, 0.841, 0.977, 1]   # normal ratio
        rawpoints = [0, .15, .30, .50, .70, .85, 1.00]    # ajust ratio
        # stdpoints = [40, 50, 65, 80, 95, 110, 120]  # std=15
        # stdpoints = [0, 15, 30, 50, 70, 85, 100]  # std=15
        stdpoints = [20, 25, 40, 60, 80, 95, 100]  # std=15

        pltmodel.output_score_decimals = 0
        pltmodel.set_data(scoredf, [fieldnames])
        pltmodel.set_parameters(rawpoints, stdpoints)
        pltmodel.run()
        # pltmodel.report()
        pltmodel.plot('raw')   # plot raw score figure, else 'std', 'model'
        return pltmodel
    if name == 'zt':
        zm = ZscoreByTable()
        zm.set_data(scoredf, [fieldnames])
        zm.set_parameters(stdnum=4, rawscore_max=150, rawscore_min=0)
        zm.run()
        zm.report()
        return zm
    if name == 'tt':
        tm = TscoreByTable()
        tm.set_data(scoredf, [fieldnames])
        tm.set_parameters(rawscore_max=150, rawscore_min=0)
        tm.run()
        tm.report()
        return tm
    if name == 'tzl':
        tm = TZscoreLinear()
        tm.set_data(scoredf, [fieldnames])
        tm.set_parameters(rawscore_max=100, rawscore_min=0)
        tm.run()
        tm.report()
        return tm
    if name == 'l9':
        tm = L9score()
        tm.set_data(scoredf, [fieldnames])
        tm.set_parameters(rawscore_max=100, rawscore_min=0)
        tm.run()
        tm.report()
        return tm


# Interface standard score transform model
class ScoreTransformModel(object):
    """
    转换分数是原始分数通过特定模型到预定义标准分数量表的映射结果。
    标准分常模是将原始分数与平均数的距离以标准差为单位表示出来的量表。
    因为它的基本单位是标准差，所以叫标准分数,也可以称为转换分数。
    常见的标准分常模有：z分数、Z分数、T分数、标准九分数、离差智商（IQ）等。
    标准分常模分数均是等距分数，虽然不同类型的常模其平均数和标准差不同，但均可用离均值来表示。
    标准分数可以通过线性转换，也可以通过非线性转换得到，
    由此可将标准分数分为线性转换的标准分数与非线性转换的标准分数。
    """
    def __init__(self):
        self.model_name = ''

        self.input_score_dataframe = pd.DataFrame()
        self.input_score_fields_list = []
        self.input_score_min = 0
        self.input_score_max = 100

        self.output_score_dataframe = pd.DataFrame()
        self.output_score_decimals = 0
        self.output_report_doc = ''

    def set_data(self, rawdf=None, scorefields=None):
        raise NotImplementedError()
        # define in subclass
        # self.__rawdf = rawdf
        # self.__scorefields = scorefields

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()

    def check_data(self):
        if type(self.input_score_dataframe) != pd.DataFrame:
            print('rawdf is not dataframe!')
            return False
        if (type(self.input_score_fields_list) != list) | (len(self.input_score_fields_list) == 0):
            print('no score fields assigned!')
            return False
        for sf in self.input_score_fields_list:
            if sf not in self.input_score_dataframe.columns:
                print(f'error score field {sf} !')
                return False
        return True

    def check_parameter(self):
        return True

    def run(self):
        if not self.check_data():
            print('check data find error!')
            return False
        if not self.check_parameter():
            print('check parameter find error!')
            return False
        return True

    def report(self):
        raise NotImplementedError()

    def plot(self, mode='raw'):
        # implemented plot_out, plot_raw score figure
        if mode.lower() == 'out':
            self.__plot_out_score()
        elif mode.lower() == 'raw':
            self.__plot_raw_score()
        else:
            print('error mode={}, use valid mode: out, raw'.format(mode))
            return False
        return True

    def __plot_out_score(self):
        if not self.input_score_fields_list:
            print('no field:{0} assign in {1}!'.format(self.input_score_fields_list, self.input_score_dataframe))
            return
        plt.figure(self.model_name + ' out score figure')
        labelstr = 'outscore: '
        for osf in self.ouput_score_dataframe.columns.values:
            if '_' in osf:  # find sf_outscore field
                labelstr = labelstr + ' ' + osf
                plt.plot(self.ouput_score_dataframe.groupby(osf)[osf].count())
                plt.xlabel(labelstr)
        return

    def __plot_raw_score(self):
        if not self.input_score_fields_list:
            print('no field assign in rawdf!')
            return
        plt.figure('Raw Score figure')
        for sf in self.input_score_fields_list:
            self.input_score_dataframe.groupby(sf)[sf].count().plot(label=f'{self.input_score_fields_list}')
        return


# model for linear score transform on some intervals
class PltScoreModel(ScoreTransformModel):
    __doc__ = ''' PltModel:
    use linear standardscore transform from raw-score intervals
    to united score intervals
    '''

    def __init__(self):
        # intit rawdf, scorefields, outdf, modelname
        super(PltScoreModel, self).__init__()
        self.model_name = 'plt'  # 'Pieceise Linear Transform Model'

        # new properties for linear segment stdscore
        self.__stdScorePoints = []
        self.__rawScorePercentPoints = []
        self.__rawScorePoints__ = []
        self.__pltCoeff = {}
        self.__formulastr = ''

        return

    def set_data(self, score_dataframe=None, score_fields_list=None):
        # check and set rawdf
        if type(score_dataframe) == pd.Series:
            self.input_score_dataframe = pd.DataFrame(score_dataframe)
        elif type(score_dataframe) == pd.DataFrame:
            self.input_score_dataframe = score_dataframe
        else:
            print('rawdf set fail!\n not correct data set(DataFrame or Series)!')
        # check and set scorefields
        if not score_fields_list:
            self.input_score_fields_list = [s for s in score_dataframe]
        elif type(score_fields_list) != list:
            print('scorefields set fail!\n not a list!')
            return
        elif sum([1 if sf in score_dataframe else 0 for sf in score_fields_list]) != len(score_fields_list):
            print('scorefields set fail!\n field must in rawdf.columns!')
            return
        else:
            self.input_score_fields_list = score_fields_list

    def set_parameters(self,
                       input_score_percent_list=None,
                       output_score_points_list=None,
                       input_score_min=0,
                       input_score_max=150):
        if (type(input_score_percent_list) != list) | (type(output_score_points_list) != list):
            print('rawscorepoints or stdscorepoints is not list type!')
            return
        if len(input_score_percent_list) != len(output_score_points_list):
            print('len is not same for rawscorepoints and stdscorepoint list!')
            return
        self.__rawScorePercentPoints = input_score_percent_list
        self.__stdScorePoints = output_score_points_list
        self.input_score_min = input_score_min
        self.input_score_max = input_score_max

    def check_parameter(self):
        if not self.input_score_fields_list:
            print('no score field assign in scorefields!')
            return False
        if (type(self.__rawScorePercentPoints) != list) | (type(self.__stdScorePoints) != list):
            print('rawscorepoints or stdscorepoints is not list type!')
            return False
        if (len(self.__rawScorePercentPoints) != len(self.__stdScorePoints)) | \
                len(self.__rawScorePercentPoints) == 0:
            print('len is 0 or not same for raw score percent and std score points list!')
            return False
        return True

    def run(self):
        # check valid
        if not super().run():
            return

        # transform score on each field
        df = self.input_score_dataframe.copy()
        result_dataframe = {}
        result_report_save = ''
        for i, fs in enumerate(self.input_score_fields_list):
            # create outdf
            filter = '(df.{0}>={1}) & (df.{2}<={3})'.\
                format(fs, self.input_score_min, fs, self.input_score_max)
            df2 = df[eval(filter)][[fs]]
            self.output_score_dataframe = df2  # .loc[:, self.input_score_fields_list]

            #self.__pltrun(fs)
            if not self.__preprocess(fs):
                print('fail to initializing !')
                return
            # transform score

            score_list = list(df2[fs].apply(self.__getscore, self.output_score_decimals))
            self.output_score_dataframe.loc[:, (fs + '_plt')] = score_list
            self._create_report()
            # print(self.output_score_dataframe.head())

            # result_dataframe = pd.merge(result_dataframe, self.output_score_dataframe[[fs+'_plt']], right_index=True, left_index=True)
            result_dataframe[fs] = self.output_score_dataframe
            result_report_save += self.output_report_doc

        self.output_report_doc = result_report_save
        self.output_score_dataframe = result_dataframe


    def _create_report(self):
        self.__formulastr = ['{0}*(x-{1})+{2}'.format(x[0], x[1], x[2])
                             for x in self.__pltCoeff.values()]
        self.output_report_doc = '\nraw score percent:{}'.format(self.__rawScorePercentPoints)
        self.output_report_doc += '\nraw score points :{}'.format(self.__rawScorePoints__)
        self.output_report_doc += '\nstd score points :{}'.format(self.__stdScorePoints)
        self.output_report_doc += '\nformulacoefficent:{}'.format(self.__pltCoeff)
        self.output_report_doc += '\n formula descrbes:{}'.format(self.__formulastr)

    def report(self):
        print(self.output_report_doc)

    def plot(self, mode='raw'):
        if mode == 'model':
            self.__plotmodel()
        elif not super().plot(mode):
            print('mode {} is invalid'.format(mode))

    # --------------property set end

    def __getcoeff(self):
        # the format is y = (y2-y1)/(x2 -x1) * (x - x1) + y1
        # coeff = (y2-y1)/(x2 -x1)
        # bias  = y1
        # rawStartpoint = x1
        for i in range(1, len(self.__stdScorePoints)):
            if (self.__rawScorePoints__[i] + self.__rawScorePoints__[i-1]) < 0.1**6:
                print('raw score percent is not differrentiable,{}-{}'.format(i, i-1))
                return False
            if self.__rawScorePoints__[i] - self.__rawScorePoints__[i - 1] != 0:
                coff = (self.__stdScorePoints[i] - self.__stdScorePoints[i - 1]) / \
                       (self.__rawScorePoints__[i] - self.__rawScorePoints__[i - 1])
            else:
                print('raw score points[{0} - {1}] same confilct!'.format(i-1, i))
                coff = 0
            y1 = self.__stdScorePoints[i - 1]
            x1 = self.__rawScorePoints__[i - 1]
            coff = math.floor(coff*10000)/10000
            self.__pltCoeff[i] = [coff, x1, y1]
        return True

    def __getscore(self, x):
        for i in range(1, len(self.__stdScorePoints)):
            if x <= self.__rawScorePoints__[i]:
                if self.output_score_decimals > 0:
                    return np.round(self.__pltCoeff[i][0] * (x - self.__pltCoeff[i][1]) + self.__pltCoeff[i][2],
                                    decimals=self.output_score_decimals)
                else:
                    return int(np.round(self.__pltCoeff[i][0] * (x - self.__pltCoeff[i][1]) + self.__pltCoeff[i][2]))
        return -1

    def __preprocess(self, field):
        # check format
        if type(self.input_score_dataframe) != pd.DataFrame:
            print('no dataset given!')
            return False
        if not self.__stdScorePoints:
            print('no standard score interval points given!')
            return False
        if not self.__rawScorePercentPoints:
            print('no score interval percent given!')
            return False
        if len(self.__rawScorePercentPoints) != len(self.__stdScorePoints):
            print('score interval for rawscore and stdscore is not same!')
            print(self.__stdScorePoints, self.__rawScorePercentPoints)
            return False
        if self.__stdScorePoints != sorted(self.__stdScorePoints):
            print('stdscore points is not in order!')
            return False
        if sum([0 if (x <= 1) & (x >= 0) else 1 for x in self.__rawScorePercentPoints]) > 0:
            print('raw score interval percent is not percent value !\n', self.__rawScorePercentPoints)
            return False

        # claculate _rawScorePoints
        if field in self.output_score_dataframe.columns.values:
            __rawdfdesc = self.output_score_dataframe[field].describe(self.__rawScorePercentPoints)
            # calculating _rawScorePoints
            self.__rawScorePoints__ = []
            for f in __rawdfdesc.index:
                if '%' in f:
                    self.__rawScorePoints__ += [__rawdfdesc.loc[f]]
        else:
            print('error score field name!')
            print('not in ' + self.input_score_dataframe.columns.values)
            return False
        # calculate Coefficients
        return self.__getcoeff()

    def __pltrun(self, scorefieldname):
        if not self.__preprocess(scorefieldname):
            print('fail to initializing !')
            return

        # transform score
        score_list = list(self.input_score_dataframe[scorefieldname].apply(self.__getscore, self.output_score_decimals))
        self.output_score_dataframe.loc[:, scorefieldname + '_plt'] = score_list

        self._create_report()

    def __plotmodel(self):
        plt.figure('Piecewise Linear Score Transform: {0}'.format(self.input_score_fields_list))
        plen = len(self.__rawScorePoints__)
        plt.xlim(self.__rawScorePoints__[0], self.__rawScorePoints__[plen - 1])
        plt.ylim(self.__stdScorePoints[0], self.__stdScorePoints[plen-1])
        plt.plot(self.__rawScorePoints__, self.__stdScorePoints)
        plt.plot([self.__rawScorePoints__[0], self.__rawScorePoints__[plen - 1]],
                 [self.__rawScorePoints__[0], self.__rawScorePoints__[plen - 1]],
                 )
        plt.xlabel('piecewise linear transform model')
        plt.show()
        return


class ZscoreByTable(ScoreTransformModel):
    """
    transform raw score to Z-score according to percent position on normal cdf
    input data: 
    rawdf = raw score dataframe
    stdNum = standard error numbers
    output data:
    outdf = result score with raw score field name + '_z'
    """
    HighPrecise = 0.9999999
    MinError = 0.1 ** 9

    def __init__(self):
        super(ZscoreByTable, self).__init__()
        self.model_name = 'zt'
        self.stdNum = 3
        self.maxRawscore = 150
        self.minRawscore = 0
        self._segtable = None
        self.__currentfield = None
        # create norm table
        self._samplesize = 100000    # cdf error is less than 0.0001
        self._normtable = pl.create_normaltable(self._samplesize, stdnum=4)
        self._normtable.loc[max(self._normtable.index), 'cdf'] = 1

    def set_data(self, rawdf=None, scorefields=None):
        self.rawdf = rawdf
        self.scorefields = scorefields

    def set_parameters(self, stdnum=3, rawscore_max=100, rawscore_min=0):
        self.stdNum = stdnum
        self.maxRawscore = rawscore_max
        self.minRawscore = rawscore_min

    def check_parameter(self):
        if self.maxRawscore <= self.minRawscore:
            print('max raw score or min raw score error!')
            return False
        if self.stdNum <= 0:
            print('std number is error!')
            return False
        return True

    def run(self):
        # check data and parameter in super
        if not super().run():
            return
        self.outdf = self.rawdf[self.scorefields]
        self._segtable = self.__getsegtable(self.outdf, self.maxRawscore, self.minRawscore, self.scorefields)
        for sf in self.scorefields:
            print('start run...')
            st = time.clock()
            self._calczscoretable(sf)
            # print(f'zscoretable finished with {time.clock()-st} consumed')
            self.outdf.loc[:, sf+'_zscore'] = self.outdf[sf].\
                apply(lambda x: x if x in self._segtable.seg.values else np.NaN)
            self.outdf.loc[:, sf+'_zscore'] = \
                self.outdf[sf+'_zscore'].replace(self._segtable.seg.values,
                                                 self._segtable[sf+'_zscore'].values)
            print(f'zscore transoform finished with {round(time.clock()-st,2)} consumed')

    def _calczscoretable(self, sf):
        if sf+'_percent' in self._segtable.columns.values:
            self._segtable.loc[:, sf+'_zscore'] = \
                self._segtable[sf+'_percent'].apply(self.__get_zscore_from_normtable)
        else:
            print(f'error: not found field{sf+"_percent"}!')

    def __get_zscore_from_normtable(self, p):
        df = self._normtable.loc[self._normtable.cdf >= p - ZscoreByTable.MinError][['sv']].head(1).sv
        y = df.values[0] if len(df) > 0 else None
        if y is None:
            print(f'error: cdf value[{p}] can not find zscore in normtable!')
            return y
        return max(-self.stdNum, min(y, self.stdNum))

    @staticmethod
    def __getsegtable(df, maxscore, minscore, scorefieldnamelist):
        """no sort problem in this segtable usage"""
        seg = ps.SegTable()
        seg.set_data(df, scorefieldnamelist)
        seg.set_parameters(segmax=maxscore, segmin=minscore)
        seg.run()
        return seg.segdf

    def report(self):
        if type(self.outdf) == pd.DataFrame:
            print('output score desc:\n', self.outdf.describe())
        else:
            print('output score data is not ready!')
        print(f'data fields in rawscore:{self.scorefields}')
        print('parameters:')
        print(f'\tzscore stadard diff numbers:{self.stdNum}')
        print(f'\tmax score in raw score:{self.maxRawscore}')
        print(f'\tmin score in raw score:{self.minRawscore}')

    def plot(self, mode='out'):
        if mode in 'raw,out':
            super().plot(mode)
        else:
            print('not support this mode!')


class TscoreByTable(ScoreTransformModel):
    __doc__ = '''
    T分数是一种标准分常模，平均数为50，标准差为10的分数。
    即这一词最早由麦柯尔于1939年提出，是为了纪念推孟和桑代克
    对智力测验，尤其是提出智商这一概念所作出的巨大贡献。'''

    def __init__(self):
        super().__init__()
        self.model_name = 't'
        self.rscore_max = 150
        self.rscore_min = 0
        self.tscore_std = 10
        self.tscore_mean = 50
        self.tscore_stdnum = 4

    def set_data(self, rawdf=None, scorefields=None):
        self.rawdf = rawdf
        self.scorefields = scorefields

    def set_parameters(self, rawscore_max=150, rawscore_min=0, tscore_mean=50, tscore_std=10, tscore_stdnum=4):
        self.rscore_max = rawscore_max
        self.rscore_min = rawscore_min
        self.tscore_mean = tscore_mean
        self.tscore_std = tscore_std
        self.tscore_stdnum = tscore_stdnum

    def run(self):
        zm = ZscoreByTable()
        zm.set_data(self.rawdf, self.scorefields)
        zm.set_parameters(stdnum=self.tscore_stdnum, rawscore_min=self.rscore_min,
                          rawscore_max=self.rscore_max)
        zm.run()
        self.outdf = zm.outdf
        namelist = self.outdf.columns
        for sf in namelist:
            if '_zscore' in sf:
                newsf = sf.replace('_zscore', '_tscore')
                self.outdf.loc[:, newsf] = \
                    self.outdf[sf].apply(lambda x: x * self.tscore_std + self.tscore_mean)

    def report(self):
        print('T-score by normal table transform report')
        print('-' * 50)
        if type(self.rawdf) == pd.DataFrame:
            print('raw score desc:')
            pl.report_stats_describe(self.rawdf)
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.outdf) == pd.DataFrame:
            print('T-score desc:')
            pl.report_stats_describe(self.outdf)
            print('-'*50)
        else:
            print('output score data is not ready!')
        print(f'data fields in rawscore:{self.scorefields}')
        print('-' * 50)
        print('parameters:')
        print(f'\tzscore stadard deviation numbers:{self.tscore_std}')
        print(f'\tmax score in raw score:{self.rscore_max}')
        print(f'\tmin score in raw score:{self.rscore_min}')
        print('-' * 50)

    def plot(self, mode='raw'):
        super().plot(mode)


class TZscoreLinear(ScoreTransformModel):
    """Get Zscore by linear formula: (x-mean)/std"""
    def __init__(self):
        #super().__init__()
        self.model_name = 'tzl'
        self.rawscore_max = 150
        self.rawscore_min = 0
        self.tscore_mean = 50
        self.tscore_std = 10
        self.tscore_stdnum = 4

    def set_data(self, rawdf=None, scorefields=None):
        self.rawdf = rawdf
        self.scorefields = scorefields

    def set_parameters(self, rawscore_max=150, rawscore_min=0, tscore_std=10, tscore_mean=50, tscore_stdnum=4):
        self.rawscore_max = rawscore_max
        self.rawscore_min = rawscore_min
        self.tscore_mean = tscore_mean
        self.tscore_std = tscore_std
        self.tscore_stdnum = tscore_stdnum

    def check_data(self):
        super().check_data()
        return True

    def check_parameter(self):
        if self.rawscore_max <= self.rawscore_min:
            print('raw score max and min error!')
            return False
        if self.tscore_std <= 0 | self.tscore_stdnum <= 0:
            print(f't score std number error:std={self.tscore_std}, stdnum={self.tscore_stdnum}')
            return False
        return True

    def run(self):
        super().run()
        self.outdf = self.rawdf[self.scorefields]
        for sf in self.scorefields:
            rmean, rstd = self.outdf[[sf]].describe().loc[['mean', 'std']].values[:, 0]
            self.outdf[sf+'_zscore'] = \
                self.outdf[sf].apply(lambda x:
                                     min(max((x - rmean) / rstd, -self.tscore_stdnum), self.tscore_stdnum))
            self.outdf.loc[:, sf+'_tscore'] = self.outdf[sf+'_zscore'].\
                apply(lambda x: x * self.tscore_std + self.tscore_mean)

    def report(self):
        print('TZ-score by linear transform report')
        print('-' * 50)
        if type(self.rawdf) == pd.DataFrame:
            print('raw score desc:')
            pl.report_stats_describe(self.rawdf)
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.outdf) == pd.DataFrame:
            print('raw,T,Z score desc:')
            pl.report_stats_describe(self.outdf)
            print('-'*50)
        else:
            print('output score data is not ready!')
        print(f'data fields in rawscore:{self.scorefields}')
        print('-' * 50)
        print('parameters:')
        print(f'\tzscore stadard deviation numbers:{self.tscore_std}')
        print(f'\tmax score in raw score:{self.rawscore_max}')
        print(f'\tmin score in raw score:{self.rawscore_min}')
        print('-' * 50)

    def plot(self, mode='raw'):
        super().plot(mode)


class L9score(ScoreTransformModel):
    """
    level 9 score transform model
    procedure: rawscore -> percent score(by segtable) -> 9 levels according cdf values:
    [0, 4%)->1, [4%, 11%)->2, [11%. 23%)->3, [23%, 40%)->4, [40%, 60%)->5
    [ 60%, 77%)->6, [77%, 89%)->7, [89%, 96%)->8, [97%, 100%]->9

    """
    def __init__(self):
        # super().__init__('l9')
        self.model_name = 'l9'
        self.rawscore_max = 100
        self.rawscore_min = 0
        self.levelscoretable = {1: [0, 0.04], 2: [0.04, 0.11], 3: [0.11, 0.23], 4: [0.23, 0.4], 5: [0.4, 0.6],
                                6: [0.6, 0.77], 7: [0.77, 0.89], 8: [0.89, 0.96], 9: [0.96, 1]}
        self.segtable = None

    def set_data(self, rawdf=None, scorefields=None):
        self.rawdf = rawdf
        self.scorefields = scorefields

    def set_parameters(self, rawscore_max=100, rawscore_min=0):
        self.rawscore_max = rawscore_max
        self.rawscore_min = rawscore_min

    def run(self):
        # import py2ee_lib as pl
        seg = ps.SegTable()
        seg.set_data(self.rawdf, self.scorefields)
        seg.set_parameters(segmax=self.rawscore_max, segmin=self.rawscore_min, segsort='ascending')
        seg.run()
        self.segtable = seg.segdf
        self.__calcscoretable()
        self.outdf = self.rawdf[self.scorefields]
        for sf in self.scorefields:
            self.outdf[sf+'_percent'] = self.outdf[sf].\
                replace(self.segtable['seg'].values, self.segtable[sf+'_percent'].values)
            self.outdf[sf+'_l9score'] = self.outdf[sf].\
                replace(self.segtable['seg'].values, self.segtable[sf+'_l9score'].values)

    def __calcscoretable(self):
        for sf in self.scorefields:
            self.segtable.loc[:, sf+'_l9score'] = self.segtable[sf+'_percent'].\
                apply(lambda x: self.__percentmaplevel(x))

    def __percentmaplevel(self, p):
        for k in self.levelscoretable:
            if p < self.levelscoretable.get(k)[1]:
                return k
        return 9

    def report(self):
        print('L9-score transform report')
        print('=' * 50)
        if type(self.rawdf) == pd.DataFrame:
            print('raw score desc:')
            pl.report_stats_describe(self.rawdf)
            print('-'*50)
        else:
            print('output score data is not ready!')
        if type(self.outdf) == pd.DataFrame:
            print('raw,L9 score desc:')
            pl.report_stats_describe(self.outdf)
            print('-'*50)
        else:
            print('output score data is not ready!')
        print(f'data fields in rawscore:{self.scorefields}')
        print('-' * 50)
        print('parameters:')
        print(f'\tmax score in raw score:{self.rawscore_max}')
        print(f'\tmin score in raw score:{self.rawscore_min}')
        print('-' * 50)

    def plot(self, mode='raw'):
        super().plot(mode)

    def check_parameter(self):
        if self.rawscore_max > self.rawscore_min:
            return True
        else:
            print('raw score max value is less than min value!')
        return False

    def check_data(self):
        return super().check_data()
