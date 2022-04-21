import os
import stat
import re
import pdb
from pathlib import Path
import getpass

import numpy as np
import h5py as h5
import pandas as pd
import subprocess
from glob import glob
from tqdm import tqdm
from .dataset import _submit_slurm_job

import shutil
import yaml
import threading
import time
from datetime import timedelta, datetime

import IPython.display
from IPython.display import display
from ipywidgets import Layout
import ipywidgets as widgets

from .plotting import AnaFile


def get_running_jobs():
    output = subprocess.check_output(
        [f"squeue -u {getpass.getuser()} | awk '{{print $1, $2, $3, $5, $6, $8}}'"],
        shell=True,
    )
    jobs = str(output, "utf-8").split("\n")[1:-1]
    columns = ["id", "partition", "name", "status", "runtime", "node"]
    jobs = np.array(list(map(lambda x: x.split(), jobs)))
    if len(jobs):
        jobs = pd.DataFrame(columns=columns, data=jobs)
        jobs["id"] = pd.to_numeric(jobs["id"])
        jobs = jobs.set_index("id")
    else:
        jobs = pd.DataFrame(columns=columns)
    return jobs


def find_jobid(output):
    sres = re.search("(?<=SLURM_JOB_ID)\s*\d{8}", output)
    return int(sres.group(0)) if bool(sres) else 0


def get_fileid(output):
    filename = get_h5filename(output)
    if bool(filename):
        sres = re.search("\d{3}(?=\.h5)", filename)
        if bool(sres):
            return int(sres.group(0))
    return np.nan


def get_datdir(output):
    filename = re.search("(?<=Analyzing ).*", output)
    return filename.group(0) if bool(filename) else ""


def is_running(jobid, jobs):
    return True if jobid in jobs.index.values else False


def is_saved(s):
    return True if re.search("Results saved", s) else False


def get_parser_args(s):
    cmd = re.search("midtools .*", s).group(0).split()[1:]
    cmd.insert(0, "setupfile")
    cmd.insert(2, "analysis")
    cmd[6::2] = list(map(lambda x: x.lstrip("-"), cmd[6::2]))
    args = dict(zip(cmd[0::2], cmd[1::2]))
    return int(args.pop("-r")), args


def get_status(df):
    jobs = get_running_jobs()
    for idx, row in df.iterrows():
        outfile = row["outfile"]
        if outfile.is_file():
            with open(outfile) as f:
                jobc = f.read()
            if is_running(row["slurm-id"], jobs):
                status = "running"
            elif is_saved(jobc):
                status = "complete"
            else:
                status = "error"
        else:
            status = "unknown"
        df.loc[idx, "status"] = status
    return df


def get_h5filename(s):
    search_strings = ["(?<=Filename: ).*", "(?<=Results saved under ).*"]
    for search_string in search_strings:
        out = re.search(search_string, s)
        if bool(out):
            return out.group(0)
    return None


def get_proposal(s):
    out = re.search("(?<=/p00)\d{4}(?=/)", s)
    return int(out.group(0)) if bool(out) else None


def get_walltime(s):
    time = re.search("(?<=Finished: elapsed time: )\d{1,}\.\d{1,}(?=min)", s)
    return timedelta(minutes=round(float(time.group(0)), 1)) if bool(time) else None


def make_jobtable(jobdir):
    if not os.path.isdir(jobdir):
        raise FileNotFoundError(f"Directory {jobdir} does not exist")

    if not len(os.listdir(jobdir)):
        print("Jobdir is empty.")
        return pd.DataFrame()

    entries = {}
    entries["run"] = []
    entries["file-id"] = []
    entries["slurm-id"] = []
    entries["runtime"] = []
    entries["datdir"] = []

    jobf = list(filter(lambda x: x.endswith("job"), os.listdir(jobdir)))
    jobf = [Path(jobdir).joinpath(x) for x in jobf]
    jobf = sorted(jobf)

    entries.update(
        {
            "jobfile": jobf,
            "outfile": [Path(str(s) + ".out") for s in jobf],
            "errfile": [Path(str(s) + ".err") for s in jobf],
        }
    )

    jobs = get_running_jobs()
    for jobfile, outfile in zip(entries["jobfile"], entries["outfile"]):
        with open(jobfile) as f:
            jobc = f.read()
        run, args = get_parser_args(jobc)
        entries["run"].append(run)
        slurm_id = 0
        t = "PD"
        datdir = ""
        file_id = -1
        if outfile.is_file():
            with open(outfile) as f:
                outc = f.read()
            file_id = args.get("file-identifier", get_fileid(outc))
            datdir = get_datdir(outc)
            slurm_id = find_jobid(outc)
            if slurm_id in jobs.index:
                t = jobs.loc[slurm_id, "runtime"]
            else:
                mtime = os.path.getmtime(outfile)
                t = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        entries["file-id"].append(file_id)
        entries["datdir"].append(datdir)
        entries["slurm-id"].append(slurm_id)
        entries["runtime"].append(t)

    df = pd.DataFrame(entries)
    df = get_status(df)
    df = df.reindex(columns=["status", *df.drop(columns=["status"]).columns])

    return df.sort_values(by=["status", "jobfile", "run", "file-id"])


def get_tcp(s):
    out = re.search("tcp://(\d{1,3}\.?){4}:\d{1,6}", s)
    if bool(out):
        return out.group(0)
    return None


from shutil import SameFileError
def log_error(df, jobdir):
    failed_jobs_file = (
        Path(jobdir).parent.joinpath("failed-jobs").joinpath("failed-jobs.yml")
    )
    if not failed_jobs_file.is_file():
        failed_jobs_file.touch()
    with open(failed_jobs_file, "r+") as f:
        failed = yaml.load(f, Loader=yaml.FullLoader)
    if not bool(failed):
        failed = {}

    status = "error"
    df = df[(df["status"] == status)]
    for index, row in df.iterrows():
        fcontent = {}
        for key in ["job", "out", "err"]:
            fname = row[key + "file"]
            if os.path.isfile(fname):
                with open(fname) as f:
                    fcontent[key] = f.read()
            else:
                fcontent[key] = ""

            if key == "err":
                shutil.copy(fname, failed_jobs_file.parent.joinpath(row[key + "file"].name))

        failed[find_jobid(fcontent["out"])] = {
            "tcp": get_tcp(fcontent["out"]),
            "errlog": row[key + "file"],
            "time": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        }

        with open(failed_jobs_file, "w") as f:
            yaml.dump(failed, f)


def handle_failed(df, remove=True, resubmit=True, run=None, subset="error"):
    if run is not None:
        df = df[df["run"] == run]
    for i, row in (
        df[df["status"] == subset].drop_duplicates(subset=["run", "file-id"]).iterrows()
    ):
        with open(row["jobfile"]) as f:
            jobc = f.read()
        run_number, args = get_parser_args(jobc)
        args["job_dir"] = str(row['jobfile'].parent)
        if remove:
            if not np.isnan(row["file-id"]):
                try:
                    anafile = AnaFile(
                        (int(run_number), int(row["file-id"])), dirname=args["out-dir"]
                    )
                    anafile.remove()
                except FileNotFoundError:
                    print(f"Tried to delete analysis result for run"
                           " {int(run_number)}, but file does not exist.")
        if resubmit:
            _submit_slurm_job(run_number, args)
        if not remove and not resubmit:
            print(f"Set either 'remove' or 'resubmit' to True. Otherwise "
                    "handle_failed will do nothing.")


def get_completed_analysis_jobs(df):
    if not len(df):
        return df
    df = df[df["status"] == "complete"]
    for i, row in (
        df[df["status"] == "complete"]
        .drop_duplicates(subset=["run", "file-id"])
        .iterrows()
    ):
        with open(row["jobfile"]) as f:
            jobc = f.read()
        run_number, args = get_parser_args(jobc)
        with open(row["outfile"]) as f:
            outc = f.read()
        filename = AnaFile(get_h5filename(outc))
        df.loc[i, "filename"] = filename.fullname
        df.loc[i, "idx"] = int(filename.counter)
        df.loc[i, "analysis"] = args["analysis"]
        df.loc[i, "proposal"] = get_proposal(outc)
        df.loc[i, "walltime"] = get_walltime(outc)
    if len(df):
        df["analysis"] = df["analysis"].astype(str)
        df["idx"] = df["idx"].astype("uint16")
        df["proposal"] = df["proposal"].astype("uint16")
        cols = ["run", "idx", "analysis"]
        cols.extend(df.drop(columns=cols).columns)
        df = df.reindex(columns=cols)
    df.drop(columns=["jobfile", "outfile", "errfile"], inplace=True)
    return df


def stop_running(df):
    for jobid in df.loc[df["status"] == "running", "slurm-id"]:
        subprocess.run(["scancel", str(jobid)])


def clean_jobdir(df, jobdir, subset=None, run=None):
    if run is not None:
        df = df[df["run"] == run]
    if subset is None:
        subset = [
            "error",
        ]

    for status in subset:
        df2 = df[(df["status"] == status)]
        for col in ["jobfile", "outfile", "errfile"]:
            files = df2[col].values
            for f in files:
                if os.path.isfile(f):
                    os.remove(f)


def merge_files(outfile, filenames, h5_structure, delete_file=False):
    """merge existing HDF5 files for a run"""

    keys = [x for y in h5_structure.values() for x in y.keys()]
    with h5.File(outfile, "w") as F:
        for filename in filenames:
            with h5.File(filename, "r") as f:
                for method in h5_structure:
                    for key, value in h5_structure[method].items():
                        fixed_size = bool(value[0])
                        if key in f:
                            data = f[key]
                            s = data.shape
                            if not key in F:
                                # check scalars and fixed_size
                                if len(s) == 0:
                                    F[key] = np.array(data)
                                else:
                                    F.create_dataset(
                                        key,
                                        data=data,
                                        compression="gzip",
                                        chunks=True,
                                        maxshape=(None, *s[1:]),
                                    )
                            else:
                                if not fixed_size:
                                    F[key].resize((F[key].shape[0] + s[0]), axis=0)
                                    F[key][-s[0] :] = data
            if delete_file:
                anafile = AnaFile(filename)
                anafile.remove()


def load_yaml_to_dict(filename):
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_closest_entry(d, proposal, run, verbose=True):
    proposals = np.asarray(list(d.keys()))
    proposal = proposals[np.argmin(np.abs(proposals-proposal))]
    runs = np.asarray(list(d[proposal].keys()))
    run = runs[np.argmin(np.abs(runs-run))]
    if verbose:
        print(
            f"Closest database entry found for proposal: {proposal}, run:{run}")
    return d[proposal][run]


def get_setupfile_content(proposal, run, masksfile=None, quadposfile=None):
    if masksfile is None:
        masksfile = r'/home/reiserm/scratch/mid-proteins/masks/masks.yml'
    if quadposfile is None:
        quadposfile = r'/home/reiserm/scratch/mid-proteins/geoms/quadrant-positions.yml'
    out = {}
    keys = ['quadrant_positions', 'mask']
    files = [quadposfile, masksfile]
    for key, f in zip(keys, files):
        print(f"Searching for '{key}' in database")
        d = load_yaml_to_dict(f)
        out[key] = get_closest_entry(d, proposal, run)[key]
    return out


def write_tmp_setupfile(d, tmpdir='./tmp/'):
    tmpdir = os.path.abspath(tmpdir)
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)
    tmpname = "tmp_setup-config_" + \
        time.strftime("%Y-%m-%d_T%H-%M-%S") + ".yml"
    tmpname = os.path.join(tmpdir, tmpname)
    with open(tmpname, 'w') as f:
        yaml.dump(d, f)
    return tmpname


def create_setupfile(proposal, run, basefile=None, tmpdir='./tmp/', search_database=False):
    if basefile is None:
        basefile = "/".join(__file__.split('/')[:-1]) + "/setup_config/setup.yml"
    d = load_yaml_to_dict(basefile)
    if search_database:
        d.update(get_setupfile_content(proposal, run))
    return write_tmp_setupfile(d, tmpdir)

def start_midtools(
        proposal,
        run_number,
        setupfile=None,
        pulses_per_train=None,
        datdir=None,
        first_train=0,
        last_train=1000,
        first_cell=0,
        pulse_step=1,
        test=False,
        jobdir='./jobs',
        base_dir=f'./analyzed_runs/',
        saxs=True,
        xpcs=True,
        ):

    out_dir = os.path.join(base_dir, f'p{proposal:06d}')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    setupfile = create_setupfile(
        proposal, run_number, basefile=setupfile, tmpdir='./tmp/')

    analysis = f'1{int(saxs)}{int(xpcs)}0'
    args = {'setupfile': setupfile,
            'analysis': analysis,
            'job_dir': jobdir,
            'out-dir': out_dir,
            'pulses_per_train': pulses_per_train,
            'pulse_step': pulse_step,
            'datdir': datdir,
            'first-train': first_train,
            'last-train': last_train,
            'first-cell': first_cell,
            }

    _submit_slurm_job(run_number, args, test=test)


class Scheduler:
    def __init__(self, jobdir):
        jobdir = os.path.abspath(jobdir)

        if not os.path.exists(jobdir):
            os.mkdir(jobdir)

        self.jobdir = jobdir

        self.df = None
        self.sel_run = None

    def gui(
        self,
    ):
        def on_button_clicked_update_jobtable(b):
            with output:
                IPython.display.clear_output()
                #                 print(f"Run Number is {run_number_IntText.value}")
                self.df = make_jobtable(self.jobdir)
                display(self.df.tail(table_length_IntText.value))

        def on_button_clicked_clean_jobdir(b):
            run = run_number_IntText.value if run_number_IntText.value >= 0 else None
            clean_jobdir(self.df, self.jobdir, subset=[subset_Dropdown.value], run=run)

        def on_button_clicked_handle_failed(b):
            run = run_number_IntText.value if run_number_IntText.value >= 0 else None
            with output:
                handle_failed(
                    self.df,
                    remove=remove_checkbox.value,
                    resubmit=resubmit_checkbox.value,
                    run=run,
                    subset=subset_Dropdown.value,
                )

        def on_button_clicked_print_file(b):
            index = table_index_IntText.value
            s = "errfile"
            if "out" in print_file_Dropdown.value:
                s = "outfile"
            elif "job" in print_file_Dropdown.value:
                s = "jobfile"
            with open(self.df.loc[table_index_IntText.value, s]) as f:
                jobc = f.read()
                #                 print(f"Job_ID: {find_jobid(jobc)}")
                # lastline = list(filter(lambda x: len(x), jobc.split("\n")))[-10:]
                lastline = list(filter(lambda x: "[" not in x, jobc.split("\n")))
                # jobc = re.sub("(^\[\#{0,39}\s*\].*$)*", "", jobc, re.MULTILINE)

                with output:
                    IPython.display.clear_output()
                    # print(jobc)
                    print("\n".join(lastline))

        def on_button_clicked_stop_running_job(b):
            table_index = int(table_index_IntText.value)
            jobid = int(self.df.loc[table_index, 'slurm-id'])
            subprocess.run(["scancel", str(jobid)])

        def on_button_clicked_list_jobs(b):
            with output:
                IPython.display.clear_output()
                subout = subprocess.check_output(f"squeue -u {getpass.getuser()}", shell=True)
                print(subout.decode('utf-8'))


        jobtable_button = widgets.Button(
            description="Update Job Table / Clear Output", layout=Layout(flex="1 0 auto", width="auto")
        )
        clean_jobdir_button = widgets.Button(
            description="Clean Jobdir", layout=Layout(flex="1 0 auto", width="auto")
        )
        print_file_button = widgets.Button(
            description="Print File", layout=Layout(flex="1 0 auto", width="auto")
        )
        handle_failed_button = widgets.Button(
            description="Handle Failed", layout=Layout(flex="1 0 auto", width="auto")
        )
        stop_running_job_button = widgets.Button(
            description="stop slurm job", layout=Layout(flex="1 0 auto", width="auto")
        )
        list_jobs_button = widgets.Button(
            description="list jobs", layout=Layout(flex="1 0 auto", width="auto")
        )

        remove_checkbox = widgets.Checkbox(
            value=False,
            description="del files",
            disabled=False,
            indent=False,
            layout=Layout(flex="1 0 auto", width="auto"),
        )
        resubmit_checkbox = widgets.Checkbox(
            value=False,
            description="resubmit",
            disabled=False,
            indent=False,
            layout=Layout(flex="1 0 auto", width="auto"),
        )

        run_number_IntText = widgets.IntText(
            description="Run",
            continuous_update=False,
            value=-1,
            layout=Layout(flex="1 1 1px", width="auto"),
        )
        table_index_IntText = widgets.IntText(
            description="Table Index",
            continuous_update=False,
            value=len(os.listdir(self.jobdir)) // 3 - 1,
            layout=Layout(flex="1 1 0%", width="auto"),
        )
        table_length_IntText = widgets.IntText(
            description="Output Length",
            continuous_update=False,
            value=8,
            layout=Layout(flex="1 1 0%", width="auto"),
        )
        subset_Dropdown = widgets.Dropdown(
            options=["error", "complete", "unknown"],
            value="error",
            description="subset:",
            continuous_update=False,
            layout=Layout(flex="1 1 auto", width="4cm"),
        )
        print_file_Dropdown = widgets.Dropdown(
            description="file",
            options=["errfile", "outfile", "jobfile"],
            value="errfile",
            continuous_update=False,
            layout=Layout(flex="1 1 auto", width="1cm"),
        )
        box_layout = Layout(
            display="flex-start",
            flex_flow="flex-start",
            align_items="flex-start",
            justify_content="flex-start",
            width="80%",
        )
        v_checkboxes = widgets.VBox([remove_checkbox, resubmit_checkbox])

        hbox1 = widgets.HBox(
            [
                jobtable_button,
                table_length_IntText,
                run_number_IntText,
            ],
            layout=box_layout,
        )
        hbox2 = widgets.HBox(
            [
                subset_Dropdown,
                clean_jobdir_button,
                list_jobs_button,
                stop_running_job_button,
            ],
            layout=box_layout,
        )
        hbox3 = widgets.HBox(
           [remove_checkbox,
            resubmit_checkbox,
            handle_failed_button,
            ],
           layout=box_layout,
        )
        hbox4 = widgets.HBox(
            [table_index_IntText, print_file_Dropdown, print_file_button],
            layout=box_layout,
        )
        vbox = widgets.VBox([hbox1, hbox2, hbox3, hbox4])

        output = widgets.Output()
        display(vbox, output)

        jobtable_button.on_click(on_button_clicked_update_jobtable)
        clean_jobdir_button.on_click(on_button_clicked_clean_jobdir)
        handle_failed_button.on_click(on_button_clicked_handle_failed)
        print_file_button.on_click(on_button_clicked_print_file)
        stop_running_job_button.on_click(on_button_clicked_stop_running_job)
        list_jobs_button.on_click(on_button_clicked_list_jobs)
