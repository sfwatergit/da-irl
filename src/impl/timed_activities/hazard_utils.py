import os.path as osp
from collections import defaultdict

import lifelines as sa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from swlcommon import TraceLoader, Persona


class PersonaLoader:
    def __init__(self, config_file, source_df_path):
        self.config_file = config_file
        self.df = pd.read_parquet(source_df_path)

    def tp(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, len(self.df))
        groups = self.df.groupby('uid')
        group = list(groups.values())[idx]
        uid_df = TraceLoader.load_traces_from_df(self.df.iloc[group])
        return Persona(traces=uid_df, build_profile=True,
                       config_file=self.config_file)

    def create_persona_traces(self, n):
        datasets = []
        for i in range(n):
            df1 = self.make_seq_df(i)
            if df1 is not None:
                df1 = df1[df1.apply(lambda x: (
                        (x.iloc[0] == 'H' and x.iloc[-1] == 'H') and not np.all(
                    x.values == 'H')), axis=1)]
                df1['persona_id'] = i
                datasets.append(df1)
        df1 = pd.concat(datasets)

        return df1

    def make_seq_df(self, idx=None):
        try:
            p = self.tp(idx)
            if (len(p.works) > 0) and \
                    (p.homes[0].confidence >
                     p.profile_builder.config.WORK_CONFIDENCE_THRESHOLD):
                f = lambda x: \
                    dict(
                        (k, s.type.symbol) for k, s in p.habitat.sites.items())[
                        x] if x is not -1 else u'-'
                seq_df = pd.DataFrame(p.sequences).T
                seq_df.index = pd.to_datetime(seq_df.index)
                seq_df = seq_df.applymap(f)
                seq_df = seq_df.apply(lambda x: x.astype('category'))
                return seq_df
        except ValueError:
            self.make_seq_df(idx)
        except IndexError:
            self.make_seq_df(idx)


def symbol_type(s):
    if s == '-':
        stype = 'travel'
    else:
        stype = 'activity'
    return stype


def row_counts(row, interval_length=15):
    row_data = []
    duration = 0
    for i in range(1, len(row) - 1):
        if row[i - 1] == row[i]:
            duration += 1
        else:
            row_data.append(
                {'symbol': row[i - 1], 'stype': symbol_type(row[i - 1]),
                 'duration': duration, 'persona_id': row.persona_id})

            duration = 1
    row_data.append({'symbol': 'H', 'stype': symbol_type(row[i - 1]),
                     'duration': duration + 1, 'persona_id': row.persona_id})
    row_data.append(
        {'symbol': 'H', 'stype': symbol_type(row[i - 1]), 'duration': 0,
         'persona_id': row.persona_id})
    out = pd.DataFrame(row_data)

    out['duration'] *= interval_length
    out['duration_next'] = np.concatenate(
        [np.roll(out.duration.values, -1)[:-1], [0]])
    out['duration_prev'] = np.concatenate(
        [np.concatenate([[0], np.roll(out.duration.values, 1)[1:-1]]), [0]])

    assert out['duration'].sum() == 1440
    return out


def create_hazard_df(df):
    data_arr = []
    data_num = 1
    for date, row in df.iterrows():

        row_df = row_counts(row)

        if (len(row_df[row_df.stype == 'activity']) == len(
                row_df[row_df.stype == 'travel']) + 2) and max(
            row_df.index) < 8:
            row_df['next_state'], row_df['next_act'], row_df[
                'prev_act'] = '', '', ''
            row_df['next_act'] = np.concatenate(
                [np.roll(row_df.symbol.values, -1)[:-1], ['H']])
            row_df['prev_act'] = np.concatenate(
                [['H'], np.roll(row_df.symbol.values, 1)[1:]])
            row_df['date'] = date
            cond = row_df.stype == "activity"
            row_df.loc[cond, 'episode'] = np.array(
                ["EP {}".format(i) for i in range(1, len(row_df[cond]) + 1)])
            row_df.loc[cond, 'state'] = np.concatenate([["{} {}".format(
                str(b + 1), a) for a, b in zip(
                row_df.loc[cond, 'symbol'].values,
                range(0, len(row_df.loc[cond]) - 1))], ['F H']])
            row_df.loc[(row_df.index == 0), 'state'] = 'S H'
            cond = row_df.stype == "travel"
            row_df.loc[row_df.stype == "travel", 'state'] = np.array(
                ["Trip {}".format(i) for i in range(1, len(row_df[cond]) + 1)])
            row_df.loc[row_df.stype == "travel", 'episode'] = np.array(
                ["EP {}".format(i) for i in range(2, len(row_df[cond]) + 2)])
            row_df.loc[:, 'next_state'] = np.concatenate(
                [np.roll(row_df.state.values, -1)[:-1], ['F H']])

            if row_df['duration'].sum() != 1440:
                continue
            row_df.date = row_df.persona_id + data_num

            tb = 1440 - row_df.duration_prev.cumsum()
            tb.iloc[-1] = 0
            row_df['time_budget'] = tb
            row_df['time_entry'] = 1440 - row_df.time_budget

            data_num += 1
            data_arr.append(row_df)

    sdf = pd.concat(data_arr)

    sdf.rename(
        columns={'next_state': 'to', 'state': 'from', 'duration': 'time'},
        inplace=True)
    sdf['state'] = sdf['from']
    return sdf


def plot_hazard(cx_ep, df_ep, fig_name, fig_dir='hazard_plots'):
    ax1 = plt.subplot(222)
    cx_ep.baseline_cumulative_hazard_.plot(ax=ax1, legend=False,
                                           title='Baseline cumulative hazard '
                                                 'rate')
    ax2 = plt.subplot(221)
    kmf = sa.KaplanMeierFitter()
    T = df_ep['time']
    kmf.fit(T)
    kmf.plot(ax=ax2)
    cx_ep.baseline_survival_.plot(ax=ax2, legend=False,
                                  title='Baseline survival rate')
    ax3 = plt.subplot(212)
    cx_ep.plot(ax=ax3)

    plt.savefig(osp.join(fig_dir, fig_name), dpi=150)
    plt.show()


def decimal_hour_to_time_of_day(dec_hr):
    hours = int(dec_hr)
    minutes = (dec_hr * 60) % 60
    seconds = (dec_hr * 3600) % 60
    return "%d:%02d:%02d" % (hours, minutes, seconds)


def prepare_hazard_analytic_df(sdf, travel=False):
    if not travel:
        pdf = sdf.loc[sdf.stype == 'activity']
        pdf.duration_prev = np.roll(pdf.time.values, 1)
        pdf.duration_next = np.roll(pdf.time.values, -1)
        pdf['prev_act'] = np.roll(pdf.symbol.values, 1)
        pdf['next_act'] = np.roll(pdf.symbol.values, -1)
    else:
        pdf = sdf.loc[sdf.stype == 'travel']
    pdf.reset_index(inplace=True)
    return pdf


def prepare_design_matrices(df, model_spec, travel=False):
    df_eps = {}
    if travel:
        group_cols = ['episode', 'next_act']
    else:
        group_cols = ['episode', 'symbol']
    for ep_id, group in sorted(df.groupby(group_cols).groups.items()):
        if travel:
            ep_id = (ep_id, 'Trip')

        df_ep = pt.dmatrix(model_spec, df.iloc[group], return_type='dataframe')
        df_ep.fillna(0, axis=1, inplace=True)
        if not travel:
            if ep_id == ('EP 1', 'H'):
                df_ep.drop(['time_budget', 'duration_prev'], axis=1,
                           inplace=True)

        del df_ep['Intercept']

        df_eps[ep_id] = df_ep
    return df_eps


def run_hazard_analysis(train_df, test_df=None, travel=False, test=False,
                        penalty=1.0, plot=True, summary=True, xval=False,
                        verbose=False):
    if test and test_df is None:
        raise ValueError("You need to provide a test dataframe to run tests!")
    outcome = defaultdict(list)
    model_spec = 'time + duration_prev + time_budget + duration_next + ' \
                 'next_act + prev_act'
    train_df = prepare_hazard_analytic_df(train_df, travel)
    train_dmatrices = prepare_design_matrices(train_df, model_spec, travel)

    if test:
        test_df = prepare_hazard_analytic_df(test_df, travel)
        test_dmatrices = prepare_design_matrices(test_df, model_spec, travel)
    else:
        test_dmatrices = None

    cx_eps = {}

    for ep_id, train_dmatrix in train_dmatrices.items():
        if travel:
            ep_data = ep_id[0]
            ep_num = ep_data[0].replace('EP ', '')
            next_act = ep_data[1]
            print("Episode: {}, Activity: Trip->{}".format(ep_num, next_act))
            fig_name = 'EP {}_Trip_{}_HazPlots'.format(ep_num, next_act)
        else:
            print("Episode: {}, Activity: {}".format(*ep_id))
            fig_name = '{}_{}_HazPlots'.format(*ep_id)

        try:
            if test:
                test_dmatrix = test_dmatrices[ep_id]
            train_dmatrix.fillna(0, axis=1, inplace=True)
            cx_ep = sa.CoxPHFitter(penalizer=penalty)
            cx_ep.fit(df=train_dmatrix, duration_col='time',
                      show_progress=verbose)

            if xval:
                scores = sa.utils.k_fold_cross_validation(cx_ep, train_dmatrix,
                                                          'time', k=3)
                print(scores)
                print(np.mean(scores))
                print(np.std(scores))
            if plot:
                plot_hazard(cx_ep, train_dmatrix, fig_name)

            if summary:
                cx_ep.print_summary()

            cx_eps[ep_id] = cx_ep

            if test and len(test_dmatrix) > 1:
                test_expected = cx_ep.predict_expectation(test_dmatrix).values

                outcome['activity'].append(ep_id[1])
                outcome['episode'].append(ep_id[0])

                outcome['pred mean duration'].append(
                    np.round(test_expected.mean() / 60., 3))
                outcome['obs mean duration'].append(
                    np.round(train_dmatrix.time.values.mean() / 60., 3))
                outcome['pred std duration'].append(
                    np.round(test_expected.std() / 60., 3))
                outcome['obs std duration'].append(
                    np.round(train_dmatrix.time.values.std() / 60., 3))


        except LinAlgError:
            print("\tLinAlgError: {}".format(ep_id))
        except ValueError:
            print("\tValueError: {}".format(ep_id))
        except ZeroDivisionError:
            pass
        except KeyError:
            print('Key Error! {}'.format(ep_id))
    if test:
        outcome_df = pd.DataFrame(outcome)
        outcome_df.set_index(['episode', 'activity'], inplace=True)
        print(outcome_df)
        return cx_eps, outcome_df
    else:
        return cx_eps, {}


def build_persona_dataset(expert):
    p = expert.persona

    f = lambda x: \
        dict((k, s.type.symbol) for k, s in p.habitat.sites.items())[
            x] if x is not -1 else u'-'
    seq_df = pd.DataFrame(p.sequences).T
    seq_df.index = pd.to_datetime(seq_df.index)
    seq_df = seq_df.applymap(f)
    seq_df = seq_df.apply(lambda x: x.astype('category'))
    df1 = seq_df

    df1 = df1[df1.apply(lambda x: (
            (x.iloc[0] == 'H' and x.iloc[-1] == 'H') and not np.all(
        x.values == 'H')), axis=1)]

    df1['persona_id'] = 0
    edf = df1

    return edf


def fix_dm(dm):
    if 'next_act[T.w]' not in dm.columns:
        dm['next_act[T.w]'] = 0
    if 'next_act[T.W]' not in dm.columns:
        dm['next_act[T.W]'] = 0
    if 'next_act[T.H]' not in dm.columns:
        dm['next_act[T.H]'] = 0
    if 'next_act[T.h]' not in dm.columns:
        dm['next_act[T.h]'] = 0
    if 'next_act[T.o]' not in dm.columns:
        dm['next_act[T.o]'] = 0
    if 'prev_act[T.w]' not in dm.columns:
        dm['prev_act[T.w]'] = 0
    if 'prev_act[T.W]' not in dm.columns:
        dm['prev_act[T.W]'] = 0
    if 'prev_act[T.H]' not in dm.columns:
        dm['prev_act[T.H]'] = 0
    if 'prev_act[T.h]' not in dm.columns:
        dm['prev_act[T.h]'] = 0
    if 'prev_act[T.o]' not in dm.columns:
        dm['prev_act[T.o]'] = 0
    return dm


def sample_duration(key, expert_cxs, dm, activity=False):
    dm = fix_dm(dm)
    if activity:
        cx = expert_cxs['activity'][0][key]
    else:
        cx = expert_cxs['travel'][0][key]
    res = -np.log(np.random.uniform(size=1)) * cx.predict_partial_hazard(dm[0:])
    cum_haz = cx.baseline_cumulative_hazard_
    vals = [find_nearest(cum_haz.values, i) for i in res.values]

    return [cum_haz.iloc[i].name for i in vals]

def build_expert_test_dms(persona, travel=False):
    edf = build_persona_dataset(persona)
    test_df = create_hazard_df(edf)
    test_df.reset_index(inplace=True)

    model_spec = 'time + duration_prev + time_budget + duration_next + ' \
                 'next_act + prev_act'
    test_df = prepare_hazard_analytic_df(test_df, travel)
    test_dmatrices = prepare_design_matrices(test_df, model_spec, travel)
    return test_dmatrices


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx
