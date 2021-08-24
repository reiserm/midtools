import numpy as np
import pandas as pd
import re
import os
from glob import glob
from datetime import datetime

from .plotting import AnaFile
from .scheduler import make_jobtable, get_completed_analysis_jobs
from . import Dataset as MidData


def update_analysis_database(jobdir, filename='analysis-table.pkl'):
    jobs_df = make_jobtable(jobdir)
    completed_df = get_completed_analysis_jobs(jobs_df)
    if not len(completed_df):
        return completed_df

    dfc = completed_df
    dfc['finished'] = pd.to_datetime(dfc['runtime'])
    ana_df = pd.read_pickle(filename)
    ana_df['finished'] = pd.to_datetime(ana_df['finished'])
    ana_df = pd.merge(ana_df, dfc.drop(
        columns=['status', 'file-id', 'runtime']), how='outer')
    ana_df['datdir'] = ana_df['datdir'].apply(
        lambda x: os.path.abspath(re.sub("\\r\d{4}$", "", x)))
    ana_df = ana_df.sort_values(by=['proposal', 'run', 'idx'])
    columns = ['proposal', 'run', 'idx', 'analysis',
               'xpcs-norm', 'filename', 'finished']
    columns.extend([x for x in ana_df.columns if x not in columns])
    ana_df = ana_df.reindex(columns=columns)
    ana_df.to_pickle(filename)
    return ana_df


def get_counter(s):
    return int(re.search('\d{3}(?=\.h5)', s).group(0))


def check_stale_entries(ana_df, drop=False):
    for index, row in ana_df.iterrows():
        f = AnaFile(row['filename'])
        if not os.path.isfile(str(f)) or f.setupfile is None:
            print('File not found', index, row['idx'], f)
            ana_df.loc[index, 'filename'] = ana_df.loc[index, 'filename'].replace('_000.h5', '_001.h5')
            if drop:
                ana_df = ana_df.drop(index)
    return ana_df


def check_stale_anafiles(ana_df, directory=None, remove=False):
    if directory is None:
        directory = "/gpfs/exfel/data/scratch/reiserm/mid-proteins/analysis-20*"
    files = sorted(glob(f"{directory}/**/*.h5"))
    for f in files:
        f = AnaFile(f)
        if not ana_df['filename'].str.match(str(f)).any() or f.setupfile is None:
            print('Stale file ', f)
            if remove:
                f.remove()


def rename_files(ana_df):
    for (proposal, run), group in ana_df[ana_df.duplicated(subset=['proposal','run'], keep=False)].groupby(['proposal','run']):
        idx = group['idx']
        inc = np.unique(np.diff(idx))
        if len(inc) == 1 and inc[0] == 1 and idx.iloc[0] == 0:
            continue
        else:
            new_indices = np.arange(len(idx))
            tmp_indices = np.arange(max(idx)+1, max(idx)+1+len(idx))
            display(group)
            l = []
            for (index, row), tmp_idx in zip(group.iterrows(), tmp_indices):
                f = AnaFile(row['filename'])
                f.rename(counter=tmp_idx)
                l.append((index, str(f.fullname)))

            for (index, old_name), new_idx in zip(l, new_indices):
                f = AnaFile(old_name)
                f.rename(counter=new_idx)
                ana_df.loc[index, ['idx', 'filename']] = new_idx, f.fullname
    return ana_df


def reset_index(ana_df):
    """This function resets the index of single files.
    """
    counts = ana_df.groupby(['proposal', 'run']).size()
    counts[counts==1]
    for (proposal, run) in counts.index.values:
        index = ana_df[(ana_df['proposal']==proposal)&(ana_df['run']==run)].index.values[0]
        if ana_df.loc[index, 'idx'] > 0:
            filename = ana_df.loc[index, 'filename']
            f = AnaFile(filename)
            f.rename(counter=0)
            ana_df.loc[index, ['idx', 'filename']] = 0, f.fullname
    return ana_df


def add_ana_infos(ana_df):
    for index, row in ana_df.iterrows():
        f = AnaFile(row['filename'])
        if f.setupfile is None:
            continue
        setupdict = MidData._read_setup(f.setupfile)
        ana_df.loc[index, 'xpcs-norm'] = setupdict.get('xpcs_opt', None).get('norm', None)
    return ana_df

def sanitize_database(database_file=None, return_df=False):
    if database_file is None:
        database_file = './analysis-table.pkl'

    ana_df = pd.read_pickle(database_file)
    ana_df['idx'] = ana_df['filename'].apply(lambda x: get_counter(x))

    ana_df = check_stale_entries(ana_df, drop=True)
    check_stale_anafiles(ana_df, remove=False)

    ana_df['finished'] = ana_df['filename'].apply(lambda x: datetime.fromtimestamp(os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M'))
    ana_df = rename_files(ana_df)
    ana_df = reset_index(ana_df)
    ana_df = add_ana_infos(ana_df)
    ana_df['finished'] = ana_df['finished'].apply(pd.to_datetime)
    ana_df.to_pickle(database_file)
    if return_df:
        return ana_df



