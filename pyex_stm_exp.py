# -*- utf-8 -*-

import pyex_stm as stm
import pandas as pd

# test Score model
def exp_scoredf_normal(mean=70, std=10, maxscore=100, minscore=0, samples=100000):
    return pd.DataFrame({'sf': [max(minscore, min(int(np.random.randn(1) * std + mean), maxscore), -x)
                                for x in range(samples)]})


def test_model(name='plt-sd20',
               df=None,
               fields_list=[],
               decimals=0,
               min_score=0,
               max_score=150):
    if type(df) != pd.DataFrame:
        print('no score df given!')
        return
    # else:
    #    scoredf = df

    if name == 'plt-sd20':
        pltmodel = stm.PltScoreModel()
        rawpoints = [0, .03, .10, .26, .50, .74, .90, .97, 1.00]    # ajust ratio
        stdpoints = [20, 30, 40, 50, 60, 70, 80, 90, 100]  # std=15

        pltmodel.output_score_decimals = decimals
        pltmodel.set_data(score_dataframe=df, score_fields_list=fields_list)
        pltmodel.set_parameters(rawpoints, stdpoints,
                                input_score_min=min_score,
                                input_score_max=max_score)
        pltmodel.run()

        pltmodel.report()
        # pltmodel.plot('raw')   # plot raw score figure, else 'std', 'model'
        # pltmodel.plot('out')

        return pltmodel

    if name == 'plt':
        pltmodel = stm.PltScoreModel()
        # rawpoints = [0, 0.023, 0.169, 0.50, 0.841, 0.977, 1]   # normal ratio
        rawpoints = [0, .15, .30, .50, .70, .85, 1.00]    # ajust ratio
        # stdpoints = [40, 50, 65, 80, 95, 110, 120]  # std=15
        # stdpoints = [0, 15, 30, 50, 70, 85, 100]  # std=15
        stdpoints = [20, 25, 40, 60, 80, 95, 100]  # std=15

        pltmodel.set_data(score_dataframe=df, score_fields_list=fields_list)
        pltmodel.set_parameters(rawpoints, stdpoints)
        pltmodel.run()
        pltmodel.report()
        pltmodel.plot('raw')   # plot raw score figure, else 'std', 'model'
        return pltmodel

    if name == 'zt':
        zm = stm.ZscoreByTable()
        zm.set_data(scoredf, fields_list)
        zm.set_parameters(stdnum=4, rawscore_max=150, rawscore_min=0)
        zm.run()
        zm.report()
        return zm
    if name == 'tt':
        tm = stm.TscoreByTable()
        tm.set_data(scoredf, fields_list)
        tm.set_parameters(rawscore_max=150, rawscore_min=0)
        tm.run()
        tm.report()
        return tm
    if name == 'tzl':
        tm = stm.TZscoreLinear()
        tm.set_data(scoredf, fields_list)
        tm.set_parameters(rawscore_max=100, rawscore_min=0)
        tm.run()
        tm.report()
        return tm
    if name == 'l9':
        tm = stm.L9score()
        tm.set_data(scoredf, fields_list)
        tm.set_parameters(rawscore_max=100, rawscore_min=0)
        tm.run()
        tm.report()
        return tm
