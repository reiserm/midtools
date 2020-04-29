# standard python packages
import numpy as np
import scipy.integrate as integrate
from time import time
import pickle
import xarray
import copy

# main analysis software can be installed by: pip install Xana
from Xana import Xana

# reading AGIPD data provided by XFEL
from extra_data import RunDirectory, stack_detector_data, open_run
from extra_geom import AGIPD_1MGeometry
from extra_data.components import AGIPD1M

# for plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.colors import LogNorm
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# DASK
import dask.array as da
from dask.distributed import Client, progress
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar


def azimuthal_integration(run, method='average', partition="upex", quad_pos=None, verbose=True,
    mask=None, to_counts=False, apply_internal_mask=True, setupfile=None,
    geom=None, savname='azimuthal_integration',  adu_per_photon=65, **kwargs):
  """Calculate the azimuthally integrated intensity of a run using dask.
  
  Args:
    run (DataCollection): the run objection, e.g., created by ``RunDirectory``.
  
    method (str, optional): how to integrate. Defaults to 'average'. Use
      'average' to average all trains and pulses. Use 'single' to calculate the
      azimuthally integrated intensity per pulse.
  
    partition (str, optional): Maxwell partition. Defaults 'upex'. 'upex' for users, 'exfel' for
      XFEL employees.
  
    quad_pos (list, optional): list of the four quadrant positions. If not
      provided, the latest configuration will be used.
  
    verbose (bool, optional): whether to print output. Defaults to True.
  
    mask (np.ndarray, optional): Good pixels are 1 bad pixels are 0. If None,
      no pixel is masked.
  
    to_count (bool, optional): Convert the adus to photon counts based on
      thresholding and the value of adu_per_photon.
  
    apply_internal_mask (bool, optional): Read and apply the mask calculated
      from the calibration pipeline. Defaults to True.
  
    setupfile (str, optional): Xana setupfile. Defaults to None. Include path
      in the filename.
  
    geom (geometry, optional):
  
    savname (str, optional): Prefix of filename under which the results are
      saved automatically. Defaults to 'azimuthal_integration'. An incrementing
      number is added not to overwrite previous results.
  
    adu_per_photon ((int,float), optional): number of ADUs per photon.
  
  
  Returns:
    dict: Dictionary containing the results depending on the method it
      contains the keys:
      - for average:
        - 'soq': the azimuthal intensity
        - 'avr': the average image (16,512,128)
        - 'img2d': the repositioned image in 2D
      - for single:
        - 'soq-pr': azimuthal intensity pulse resolved
        - 'q(nm-1)': the q-values in inverse nanometers
  
  """

  def integrate_azimuthally(data, returnq=True):
    # Check data type
    if isinstance(data, np.ndarray):
      pass
    elif isinstance(data, xarray.DataArray):
      data = data.data
    else:
      raise(ValueError(f"Data type {type(data)} not understood."))

    # do azimuthal integration
    wnan = np.isnan(data)
    q, I = ai.integrate1d(data.reshape(8192,128), 
                            500, 
                            mask=~(mask.reshape(8192,128).astype('bool')) 
                                  | wnan.reshape(8192,128), 
                            unit='q_nm^-1', dummy=np.nan)
    if returnq:
      return q, I
    else:
      return I 

  t_start = time()
  xana = Xana(detector='agipd1m', setupfile=setupfile)

  if setupfile is None:
    xana.setup.make()

  if mask is None:
    mask = np.ones((16,512,128), "bool")

  if verbose:
    # run.info()
    print(f"Calculating azimuthal intensity for {method} trains.")
    
  cluster = SLURMCluster(
    queue=partition,
    processes=5, 
    cores=25, 
    memory='500GB',
    log_directory='./dask_log/',
    local_directory='./dask_tmp/',
    nanny=True,
    death_timeout=100,
    walltime="24:00:00",
  )

  cluster.scale(50)
  client = Client(cluster)

  # define corners of quadrants. 
  if quad_pos is None:
    dx = -18
    dy = -15
    quad_pos = [(-500+dx, 650+dy),
               (-550+dx, -30+dy),
               (570+dx, -216+dy),
               (620+dx, 500+dy)] 

  # generate the detector geometry
  if geom is None:
    geom = AGIPD_1MGeometry.from_quad_positions(quad_pos)

  # update the azimuthal integrator
  dist = geom.to_distortion_array()
  xana.setup.detector.IS_CONTIGUOUS = False
  xana.setup.detector.set_pixel_corners(dist)
  xana.setup.update_ai()

  agp = AGIPD1M(run, min_modules=16)
  # data = agp.get_dask_array('image.data')
  arr = agp.get_dask_array('image.data')
  if to_counts:
    arr.data = np.floor((arr.data + 0.5*adu_per_photon) / adu_per_photon)
    arr.data[arr.data<0] = 0
    # arr.data.astype('float32')

  if apply_internal_mask:
    mask_int = agp.get_dask_array('image.mask')
    arr = arr.where((mask_int.data <= 0) | (mask_int.data>=8))
    # mask[(mask_int.data > 0) & (mask_int.data<8)] = 0

  if verbose:
    print(f"masked {np.sum(mask==0)/np.sum(mask==1)*100:.2f}% of the pixels")

  arr = arr.where(mask[:,None,:,:])

  if verbose:
    print("Cluster dashboard link:", cluster.dashboard_link)

  ai = copy.copy(xana.setup.ai)

  # do the actual calculation
  if method == 'average':
    avr_img = arr.mean('train_pulse', skipna=True).compute()

    # aziumthal integration
    q, I = integrate_azimuthally(avr_img)

    img2d = geom.position_modules_fast(avr_img.data)[0]
    savname = f'./{savname}_Iq_{int(time())}.pkl'
    pickle.dump({"soq":(q,I), "avr":avr_img, "avr2d":img2d}, open(savname, 'wb'))
  
  elif method == 'single':
    arr = arr.stack(pixels=('module','dim_0','dim_1')).chunk({'train_pulse':1000})
    q = integrate_azimuthally(arr[0])[0]

    print("Start calculation...")
    dim = arr.get_axis_num("pixels")
    saxs_pulse_resolved = da.apply_along_axis(integrate_azimuthally, dim, \
        arr.data, dtype='float32', shape=(500,), returnq=False)
    saxs_arr = saxs_pulse_resolved.compute()

    savname = f'./{savname}_Iqpr_{int(time())}.pkl'
    pickle.dump({"soq-pr":saxs_arr, "q(nm-1)":q}, open(savname, 'wb'))
  else:
    raise(ValueError(f"Method {method} not understood. \
                       Choose between 'average' and 'single'."))
  
  elapsed_time = time() - t_start
  cluster.close()
  client.close()
  print(f"Finished: elapsed time: {elapsed_time/60:.2f}min")
  print(f"Filename is {savname}")
  return
