"""
  /*
  Copyright (c) 2014, The Australian National University
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  
  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
  */
  
  /*
  Please acknowledge use of this code by citing the following works in any publications:
  
  Myers, G, Kingston, A, Varslot, T et al 2011, 'Extending reference scan drift correction to high-magnification high-cone-angle tomography', Optics Letters, vol. 36, no. 24, pp. 4809-4811.
  
  Sheppard, A, Latham, S, Middleton, J et al 2014, 'Techniques in helical scanning, dynamic imaging and image segmentation for improved quantitative analysis with X-ray micro-CT', Nuclear Instruments and Methods in Physics Research: Section B, vol. 324, pp. 49-56.
  
  Kingston, A, Sakellariou, A, Varslot, T et al 2011, 'Reliable automatic alignment of tomographic projection data by passive auto-focus', Medical Physics, vol. 38, no. 9, pp. 4934-4945.
  
  */
"""


# This is a module which interfaces with a CPU-accellerated CUDA backprojection function.
# To-do:
#	Control the data types better, helping with memory management.
#	Make function rampfilter local
#	It's possible for there to be confusion about the float length, between python and c.

import numpy as np
import ctypes
import math
import sys

#Data types compatible with c_float and c_int
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

def project(vol, proj, angle, sourceDistance, zOrigin, volStart=np.array([0,0,0], dtype=INT_TYPE), volDims=None, projWOffset=None, projHOffset=None, projDims=None, norm=None, GPUID=0, misAlignments=np.array([0,0,0,0,0,0], dtype=FLOAT_TYPE), pixelWidth=1.0):
  """
  Assumed geometry:
  Cone beam. We assume the rotation axis and projections are correctly centred. The projection data is NOT zeroed before projection occurs; the projected volume is added to whatever data exists there already.

  Necessary parameters:
  vol[z,y,x] - The volume region to be reconstructed/projected. If no vol**** optional parameters are specified, this is assumed to be the entire sample volume.
  proj[theta, height, width] - The projection region to be calculated/backprojected. If no proj**** optional parameters are specified, this is assumed to be the entire projection data set.
  projAngles = 1D array of projection angles in radians i.e. proj(n,*,*) is at angle projAngles(n). Do not have to be sequential.
  sourceDistance - the distance from the source to the centre of the object, measured in pixels. 
  zOrigin - the z index (measured from element [0,0,0] of the volume) of the source AND detector centre. Typically volDims[0]/2 for circular.
  Optional parameters:
  volStart - 3 element array. If vol is only a subset of the total volume (e.g. due to splitting the operation across many GPUs), then this specifies the offset of vol[0,0,0] from the [0,0,0]-indexed pixel of the total volume. E.g. if I'm doing the bottom half of a 256 cubed sample, then vol.shape = [128,256,256],
  volStart = [128,0,0], and volDims = [256,256,256]. Defaults to zero offset.
  volEnd - 3 element array. If vol is only a subset of the total volume, (e.g. due to splitting the operation across many GPUs), then this specifies the dimensions of the total sample area. This is necessary so that the algorithm knows where to put the rotation axis. Defaults to volStart + vol.shape, but it is recommended to specify this explicitly if a sub-volume is being reconstructed.
  projDims - A 3-element array, used if the array proj only covers a sub-region of the total projection data set (e.g. due to MPI). Specifies the total dimensions of the projection data set. The total number of projections is used for normalisation purposes when backprojecting, but is irrelevant to projection operations.
  projWOffset[theta] - The offset of the element proj[theta,0,0] from the [:,0,0]-index of the total projection data set. May be different for every theta. Assumed to be zero.
  projHOffset[theta] - As above, but along the h axis.
  norm[theta] - A list of normalization factors to be applied to projection theta. Defaults to 1/projDim[0] for backprojection, and 1 for projection.i
  misAlignments - Optional misalignment parameters. Parameters are as per Andrew's scheme, and are applied in the order they are specified: Dw, Dh, Dl, Dphi, Dtheta, Dpsi. First three terms are in units of "pixels". Last 3 are in units of radians. Dphi is a rotation about the w axis, Dtheta a rotation about the h axis, and Dpsi a rotation in the w-h plane.
  pixelWidth - The width and height of a demagnified detector pixel.
  """
  #Check for exceptions - only things that can cause the c interface to crash.
  if (vol.dtype != FLOAT_TYPE):
    sys.exit("Volume array supplied to function 'project' does not have data type np.float32")
  if (proj.dtype != FLOAT_TYPE):
    sys.exit("Projection array supplied to function 'project' does not have data type np.float32")
  if (angle.size != proj.shape[0]):
    sys.exit("Number of angles supplied does not match number of projections in proj array")
  if (zOrigin.size != proj.shape[0]):
    sys.exit("Number of zOrigin elements does not match number of projections in proj array")
  #Assign default values appropriate variables
  volEnd = np.array([vol.shape[0]+volStart[0], vol.shape[1]+volStart[1], vol.shape[2]+volStart[2]], dtype=INT_TYPE)
  if (volDims == None):
    volDims = np.array([vol.shape[0], vol.shape[1], vol.shape[2]], dtype=INT_TYPE)
  if ( projWOffset == None ):
    projWOffset = np.zeros(angle.size, dtype=INT_TYPE)
  if ( projHOffset == None ):
    projHOffset = np.zeros(angle.size, dtype=INT_TYPE)
  if (projDims == None):
    projDims = np.array([proj.shape[0], proj.shape[1]+projHOffset.max(), proj.shape[2]+projWOffset.max()], dtype=INT_TYPE)
  if (norm == None):
    norm = np.ones(angle.size, dtype=FLOAT_TYPE)
  projDims = projDims.copy()
  projDims[0] = proj.shape[0] #projDims[0] means something different to the c code.
  projWHSize = np.array([proj.shape[1], proj.shape[2]], dtype=INT_TYPE)
  hMidplane = zOrigin.astype(FLOAT_TYPE)
  #Fix type incompatibilities
  if (angle.dtype != FLOAT_TYPE):
    angle = angle.astype(FLOAT_TYPE)
  if (volStart.dtype != INT_TYPE):
    volStart = volStart.astype(INT_TYPE)
  if (volDims.dtype != INT_TYPE):
    volDims = volDims.astype(INT_TYPE)
  if (projWOffset.dtype != INT_TYPE):
    projWOffset = projWOffset.astype(INT_TYPE)
  if (projHOffset.dtype != INT_TYPE):
    projHOffset = projHOffset.astype(INT_TYPE)
  if (projDims.dtype != INT_TYPE):
    projDims = projDims.astype(INT_TYPE)
  if (norm.dtype != FLOAT_TYPE):
    norm = norm.astype(FLOAT_TYPE)
  if (misAlignments.dtype != FLOAT_TYPE):
    misAlignments = misAlignments.astype(FLOAT_TYPE)
  #Reverse dimension variables, so that they're appropriate for the c code.
  volStart = volStart[::-1].copy()
  volEnd = volEnd[::-1].copy()
  volDims = volDims[::-1].copy()
  projDims = projDims[::-1].copy()
  projWHSize = projWHSize[::-1].copy()
  #Load, use, and unload the c library
  libc = ctypes.CDLL("libFastcone.so")
  libc.GPU_CB_misalign_project( ctypes.c_int( GPUID ),
    vol.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
    angle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    hMidplane.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_float( sourceDistance ),
    ctypes.c_float( pixelWidth ),
    volDims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    volStart.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    volEnd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    projDims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    projWHSize.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    projWOffset.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    projHOffset.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    misAlignments.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    #The projection code assumes the centre of the projection is at pixel Nh/2, when it should be Nh/2 - 1/2. For this reason, we zero the 0th column of the array.
  for tt in range(0, proj.shape[0], 1):
    if (projWOffset[tt] <= 0):
      proj[tt, :, 0:(1+projWOffset[tt])] = 0.0
  return 1


def backproject(vol, proj, angle, sourceDistance, zOrigin, volStart=np.array([0,0,0], dtype=INT_TYPE), volDims=None, projWOffset=None, projHOffset=None, projDims=None, norm=None, GPUID=0, kernelMode=2, misAlignments=None, pixelWidth=1.0):
  """
  Assumed geometry:
  Cone beam. We assume the rotation axis and projections are correctly centred. The volume data is not zeroed; the backprojected projection data is just added to whatever is there already.

  Necessary parameters:
  vol[z,y,x] - The volume region to be reconstructed/projected. If no vol**** optional parameters are specified, this is assumed to be the entire sample volume.
  proj[theta, height, width] - The projection region to be calculated/backprojected. If no proj**** optional parameters are specified, this is assumed to be the entire projection data set.
  projAngles = 1D array of projection angles in radians i.e. proj(n,*,*) is at angle projAngles(n). Do not have to be sequential.
  sourceDistance - the distance from the source to the centre of the object, measured in pixels.
  zOrigin - the z index (measured from element [0,0,0] of the volume) of the source AND detector centre. Typically volDims[0]/2 for circular.

  Optional parameters:
  volStart - 3 element array. If vol is only a subset of the total volume (e.g. due to splitting the operation across many GPUs), then this specifies the offset of vol[0,0,0] from the [0,0,0]-indexed pixel of the total volume. E.g. if I'm doing the bottom half of a 256 cubed sample, then vol.shape = [128,256,256], volStart = [128,0,0], and volDims = [256,256,256]. Defaults to zero offset.
  volEnd - 3 element array. If vol is only a subset of the total volume, (e.g. due to splitting the operation across many GPUs), then this specifies the dimensions of the total sample area. This is necessary so that the algorithm knows where to put the rotation axis. Defaults to volStart + vol.shape, but it is recommended to specify this explicitly if a sub-volume is being reconstructed.
  projDims - A 3-element array, used if the array proj only covers a sub-region of the total projection data set (e.g. due to MPI). Specifies the total dimensions of the projection data set. The total number of projections is used for normalisation purposes when backprojecting, but is irrelevant to projection operations.
  projWOffset[theta] - The offset of the element proj[theta,0,0] from the [:,0,0]-index of the total projection data set. May be different for every theta. Assumed to be zero.
  projHOffset[theta] - As above, but along the h axis.
  norm[theta] - A list of normalization factors to be applied to projection theta. Defaults to 1/projDim[0] for backprojection, and 1 for projection.
  kernelMode - The type of kernel used to do the backprojection. Mode 2 is a Feldkamp-type kernel, Mode 1 is a Katsevich-type kernel, Mode 0 is an unweighted-type kernel, suitable for iterative reconstruction like SIRT.
  pixelWidth - The width and height of a demagnified detector pixel, in voxels.
  """
  #Check for exceptions - only things that can cause the c interface to crash.
  if (vol.dtype != FLOAT_TYPE):
    sys.exit("Volume array supplied to function 'project' does not have data type np.float32")
  if (proj.dtype != FLOAT_TYPE):
    sys.exit("Projection array supplied to function 'project' does not have data type np.float32")
  if (angle.size != proj.shape[0]):
    sys.exit("Number of angles supplied does not match number of projections in proj array")
  if (zOrigin.size != proj.shape[0]):
    sys.exit("Number of zOrigin elements does not match number of projections in proj array")
  if (misAlignments != None) and (kernelMode != 0):
    sys.exit("Cannot conduct misaligned backprojection unless kernel mode is set to 0 (i.e. unweighted, not Feldkamp or Katsevich).")
  #Assign default values appropriate variables
  volEnd = np.array([vol.shape[0]+volStart[0], vol.shape[1]+volStart[1], vol.shape[2]+volStart[2]], dtype=INT_TYPE)
  if (volDims == None):
    volDims = volEnd.copy()
  if ( projWOffset == None ):
    projWOffset = np.zeros(angle.size, dtype=INT_TYPE)
  if ( projHOffset == None ):
    projHOffset = np.zeros(angle.size, dtype=INT_TYPE)
  if (projDims == None):
    projDims = np.array([proj.shape[0], proj.shape[1]+projHOffset.max(), proj.shape[2]+projWOffset.max()], dtype=INT_TYPE)
  if (norm == None):
    norm =  np.ones(angle.size, dtype=FLOAT_TYPE) / projDims[0]
  projDims = projDims.copy() #projDim[0] means different things to the c and python code, which is why this is necessary.
  projDims[0] = proj.shape[0] #projDim[0] means different things to the c and python code, which is why this is necessary.
  projWHSize = np.array([proj.shape[1], proj.shape[2]], dtype=INT_TYPE)
  hMidplane = zOrigin.astype(FLOAT_TYPE)
  #Fix type incompatibilities
  if (angle.dtype != FLOAT_TYPE):
    angle = angle.astype(FLOAT_TYPE)
  if (volStart.dtype != INT_TYPE):
    volStart = volStart.astype(INT_TYPE)
  if (volDims.dtype != INT_TYPE):
    volDims = volDims.astype(INT_TYPE)
  if (projWOffset.dtype != INT_TYPE):
    projWOffset = projWOffset.astype(INT_TYPE)
  if (projHOffset.dtype != INT_TYPE):
    projHOffset = projHOffset.astype(INT_TYPE)
  if (projDims.dtype != INT_TYPE):
    projDims = projDims.astype(INT_TYPE)
  if (norm.dtype != FLOAT_TYPE):
    norm = norm.astype(FLOAT_TYPE)
  #Reverse dimension variables, so that they're appropriate for the c code.
  volStart = volStart[::-1].copy()
  volEnd = volEnd[::-1].copy()
  volDims = volDims[::-1].copy()
  projDims = projDims[::-1].copy()
  projWHSize = projWHSize[::-1].copy()
  #Load, use, and unload the c library
  libc = ctypes.CDLL("libFastcone.so")
  if (misAlignments == None):
    libc.GPU_CB_K_backproject( ctypes.c_int( GPUID ),
      vol.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
      proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
      angle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
      hMidplane.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
      ctypes.c_float(sourceDistance),
      ctypes.c_float( pixelWidth ),
      volDims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
      volStart.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
      volEnd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
      projDims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
      projWHSize.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
      projWOffset.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
      projHOffset.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
      norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
      ctypes.c_int(kernelMode))
  else:
    if (misAlignments.dtype != FLOAT_TYPE):
      misAlignments = misAlignments.astype(FLOAT_TYPE)
    libc.GPU_CB_misalign_backproject(ctypes.c_int( GPUID ),
                              vol.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              angle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              hMidplane.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              ctypes.c_float(sourceDistance),
                              ctypes.c_float( pixelWidth ),
                              volDims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                              volStart.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                              volEnd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                              projDims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                              projWHSize.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                              projWOffset.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                              projHOffset.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                              norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              misAlignments.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
  #Return success signal.
  return 1



def fbp(vol, proj, projAngles, specimenLength, zOrigin, volStart=np.array([0,0,0], dtype=INT_TYPE), volDims=None, GPUID=0, pixelWidth=1.0):
  ww = np.arange(0.0, proj.shape[2], 1.0) - proj.shape[2]/2.0
  ww.resize([1,proj.shape[2]])
  for pp in range(proj.shape[0]):
    hh = np.arange(0.0, proj.shape[1], 1.0) - zOrigin[pp]
    hh.resize([proj.shape[1],1])
    reScale = specimenLength / np.sqrt(specimenLength*specimenLength + hh*hh + ww*ww)
    proj[pp,:,:] = proj[pp,:,:] * reScale[:,:]    
  rampfilter( proj )
  backproject(vol, proj, projAngles, specimenLength, zOrigin, volStart, volDims, GPUID=GPUID, pixelWidth=pixelWidth)
  return reScale
	
def rampfilter(proj):
  #Define the padding parameters
  projPadded = np.zeros([proj.shape[1], proj.shape[2]*2.0])
  projBegin_w = math.floor((projPadded.shape[1] - proj.shape[2]) / 2.0)
  projEnd_w = projBegin_w + proj.shape[2]
  #Define the filter
  filter = np.abs(np.fft.fftfreq(projPadded.shape[1]))
  filter[0] = 0.125 * filter[1]
  filter *= np.pi
  filter = np.resize(filter, [1, 1+math.floor(projPadded.shape[1]/2.0)])
  #For loop over all angles. This is probably incredibly wasteful re: memory, we're creating new arrays all the time.
  for tt in range(0,proj.shape[0]):
    projPadded[:,:] = 0.0
    projPadded[:, 0:proj.shape[2]] = proj[tt,:,:]
    temp = np.fft.rfft(projPadded, axis = 1) 
    #This is the last axis in the truncated 2D proj; they count from zero.
    temp *= filter 
    #Due to the broadcasting rules, above command will iterate over the first dimension of temp; which is h.
    projPadded[:,:] = np.fft.irfft(temp, axis = 1)
    proj[tt,:,:] = projPadded[:, 0:proj.shape[2]]
  return

def cosrampfilter( proj ):
  # Define the temporary arrays
  cfilter = np.abs( np.fft.fftfreq( proj.shape[2] ) )
  cfilter = np.resize( cfilter , [ 1 + math.floor( proj.shape[2] / 2.0 ) ] )
  cfilter *= np.cos(0.5*cfilter*np.pi/proj.shape[2])
  # For loop over all angles. This is probably incredibly wasteful re: memory, we're creating new arrays all the time.
  for tt in range(0,proj.shape[0]):
    temp = np.fft.rfft( proj[tt,:,:] , axis=1 )
    temp *= cfilter
  proj[tt,:,:] = np.fft.irfft( temp , axis = 1 )
  return

def gaussrampfilter( proj , scale=0.125):
  # scale is sigma as fraction of Nx (default is 1/8th)
  # Define the temporary arrays
  gfilter = np.abs( np.fft.fftfreq( proj.shape[2] ) )
  gaussianWindow =  scipy.signal.gaussian( proj.shape[2], scale*float(proj.shape[2]))
  gaussianWindow = scipy.ndimage.shift(gaussianWindow,-0.5*float(proj.shape[2]))
  gfilter *= gaussianWindow
  gfilter = np.resize( gfilter , [ 1 + math.floor( proj.shape[2] / 2.0 ) ] )
  # For loop over all angles. This is probably incredibly wasteful re: memory, we're creating new arrays all the time.
  for tt in range(0,proj.shape[0]):
    temp = np.fft.rfft( proj[tt,:,:] , axis=1 )
    temp *= gfilter
  proj[tt,:,:] = np.fft.irfft( temp , axis = 1 )
  return
