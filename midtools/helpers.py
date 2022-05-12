from dask.distributed import Client, progress, LocalCluster
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar
from extra_data.components import AGIPD1M
from matplotlib import pyplot as plt


def start_slurm_cluster(njobs=4, nprocs=1, use_local_cluster=False, partition='upex'):

    if use_local_cluster:
        cluster = LocalCluster(
            n_workers=nprocs,
        )
    else:
        print(f"\nSubmitting {njobs} jobs using {nprocs} processes per job.")
        cluster = SLURMCluster(
            queue=partition,
            processes=nprocs,
            cores=40,
            memory="400GB",
            log_directory="./dask_log/",
            local_directory="/scratch/",
            nanny=True,
            death_timeout=60 * 60,
            walltime="24:00:00",
            #       interface="ib0",
            name="XPCS",
        )
        cluster.scale(nprocs * njobs)
        # self._cluster.adapt(maximum_jobs=njobs)

    print(cluster)
    client = Client(cluster)
    print("Cluster dashboard link:", cluster.dashboard_link)
    return cluster, client


def get_pulse_intensity(run, number_of_trains=200):
    """Integrate AGIPD run and return average intensity per pulse.
    """
    agipd = AGIPD1M(run, min_modules=16)
    images = agipd.get_dask_array('image.data')
    images = (images
          .unstack()
          .isel(trainId=slice(0,number_of_trains)))

    if "dim_2" in images.dims:
        images = images.isel(dim_0=0)

    avr_dims = list(images.dims)
    del avr_dims[avr_dims.index('pulseId')]

    pulse_intensity = images.mean(avr_dims)
    return pulse_intensity


def get_agipd_pulse_pattern(pulse_intensity, threshold=.1, plot=False):
    """Compute the AGIPD pulse pattern based on the average intensity."""
    xray_pulse_indices = pulse_intensity.pulseId[pulse_intensity>threshold]
    xray_pulses = pulse_intensity.sel(pulseId=xray_pulse_indices)
    empty_pulses = pulse_intensity.drop_sel(pulseId=xray_pulse_indices)
    number_of_pulses = len(xray_pulses)

    first_cell = xray_pulse_indices[0].values[()]

    cell_step = 1
    itercells = pulse_intensity.pulseId[pulse_intensity.pulseId>first_cell].values
    while itercells[0] in empty_pulses.pulseId.values:
        cell_step += 1
        itercells = itercells[1:]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(xray_pulses.pulseId, xray_pulses, '.', color='red', label='X-rays')
        ax.plot(empty_pulses.pulseId, empty_pulses, '.', color='gray', label='empty')
        ax.hlines(threshold, pulse_intensity.pulseId.min(), pulse_intensity.pulseId.max(), color='k', ls='--')
        ax.grid(ls=':')
        ax.set_xlabel('pulse ID')
        ax.set_ylabel('average pulse intensity / pixel / frame / train')
        ax.legend(loc='upper right')
        ax.set_title(f"{first_cell =}, {cell_step = }, {number_of_pulses = }")
        ax.minorticks_on()

    return first_cell, cell_step, number_of_pulses
