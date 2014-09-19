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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <algorithm>

#include "coneBeamGPU.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>

//****************************************
//Definitions
//****************************************

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8
#define BLOCK_SIZE_W 16
#define BLOCK_SIZE_H 16
#define FLOAT_TEX_OFFSET 0.5f
#define OVERSAMPLING 0.14285714f

//The 0.5f offset is necessary when dealing with 3D linear texture filtering. See section E.2 of the CUDA programming guide, v2.3.

//****************************************
//Variables with File scope (e.g. textures)
//****************************************

static texture<float, 3, cudaReadModeElementType> texRef;

//****************************************
//Local class/structure prototypes
//****************************************




//****************************************
//Local function prototypes
//****************************************



//****************************************
//Kernel prototypes
//****************************************


//****************************************
//Device Kernels (may only be called from device)
//****************************************



//****************************************
//Global Kernels (may be called from anywhere)
//****************************************

__global__ void GPU_new_project_volume(
  float* proj_d,
  //Geometry vectors. Not using built-in vector types at the moment, to simplify coding. It doesn't actually matter where the detector
  // is centred; that will only change the values in ssStart ssNumSteps.
  float* sourcePosition_xyz_d,
  float* detectorCentre_xyz_d,
  float* hStep_xyz_d, //The xyz vector corresponding to a single, scaled, detector pixel step in the h direction.
  float* wStep_xyz_d, //The xyz vector corresponding to a single, scaled, detector pixel step in the w direction.
  int* sStart_d, //The number of steps along the line integrals that should be taken to get from the detector plane, to the start of the integration problem.
  int* sNumSteps_d, //The number of steps that should be taken along the line integral, until we've left the supplied subVol data. This should NOT take into the accound of
    //OVERSAMPLING; that is done in the kernel.
  //Offset variables are the offset of the (0,0,0) element of the sub-proj and/or sub-volume w.r.t. the centre of the overall CT reconstruction problem.
  //More importantly, it's the number required to convert device array indicies to geometric coordinates. Index + Offset = Coordinate
  int* NwOffset_d,
  int* NhOffset_d,
  int NxOffset,
  int NyOffset,
  int NzOffset,
  //The dimensions of the sub-proj and/or sub-vol arrays.
  int Nw,
  int Nh
  )
{
  //Declare shared variables
  __shared__ int tt_s;
  __shared__ int Nh_s;
  __shared__ int sNumSteps_s;
  __shared__ float3 volOffset_s, detectorCentre_s, sourcePos_s, wStep_s, hStep_s;
  __shared__ float proj_s[BLOCK_SIZE_W * BLOCK_SIZE_H];
  //Declare thread variables
  int thread, ww, hh, ss;
  float3 volPos, volStep;
  //Calculate thread number
  thread = threadIdx.x + threadIdx.y * BLOCK_SIZE_W;
  //Zero shared memory
  proj_s[thread] = 0.0f;
  //Populate shared memory variables from device memory
  #warning Figure out how to do coallesced memory access for float3 variables
  if(thread == 0){
    tt_s = (int)blockIdx.y / (int)(Nh / BLOCK_SIZE_H);
//    Nw_s = Nw;
    Nh_s = Nh;
    sNumSteps_s = sNumSteps_d[tt_s];
    volOffset_s.x = NxOffset;
    volOffset_s.y = NyOffset;
    volOffset_s.z = NzOffset;
    detectorCentre_s.x = detectorCentre_xyz_d[tt_s*3];
    detectorCentre_s.y = detectorCentre_xyz_d[tt_s*3 + 1];
    detectorCentre_s.z = detectorCentre_xyz_d[tt_s*3 + 2];
    wStep_s.x = wStep_xyz_d[tt_s*3];
    wStep_s.y = wStep_xyz_d[tt_s*3+1];
    wStep_s.z = wStep_xyz_d[tt_s*3+2];
    hStep_s.x = hStep_xyz_d[tt_s*3];
    hStep_s.y = hStep_xyz_d[tt_s*3+1];
    hStep_s.z = hStep_xyz_d[tt_s*3+2];
    sourcePos_s.x = sourcePosition_xyz_d[tt_s*3];
    sourcePos_s.y = sourcePosition_xyz_d[tt_s*3 + 1];
    sourcePos_s.z = sourcePosition_xyz_d[tt_s*3 + 2];
  }
  __syncthreads();
  //Calculate w and h indicies for this detector pixel
  ww = threadIdx.x + BLOCK_SIZE_W * blockIdx.x;
  hh = threadIdx.y + BLOCK_SIZE_H * (blockIdx.y % (Nh_s / BLOCK_SIZE_H));
  //Calculate xyz coordinates (in detector plane) of this detector pixel
  volPos = detectorCentre_s + (ww + NwOffset_d[tt_s]) * wStep_s + (hh + NhOffset_d[tt_s]) * hStep_s;
  //Calculate xyz step vector for this detector pixel
  volStep = volPos - sourcePos_s;
  volStep *= rsqrt(dot(volStep, volStep)); //(rsqrt is reciprocal square root, with precision 2 ulp).
  //Step back to the start of the integration
  volPos = sourcePos_s + (sStart_d[tt_s] * volStep);
  //Convert xyz coordinates to xyz indicies
  volPos -= volOffset_s - FLOAT_TEX_OFFSET;
  //Do integration
  for(ss=0; ss<sNumSteps_s; ss++){
    proj_s[thread] += tex3D(texRef, volPos.x, volPos.y, volPos.z);
    volPos += volStep;
  }
  //Store the shared memory to device memory
  proj_d[ww + hh * Nw + tt_s * Nw * Nh_s] = proj_s[thread];
  return;
}

__global__ void GPU_unweighted_backproject_sinogram(
  float* vol_d,
  //Geometry vectors. Not using built-in vector types at the moment, to simplify coding. It doesn't actually matter where the detector
  // is centred; that will only change the values in ssStart ssNumSteps.
  float* sourcePosition_xyz_d,
  float* detectorCentre_xyz_d,
  float* hStep_xyz_d, //The xyz vector corresponding to a single, scaled, detector pixel step in the h direction.
  float* wStep_xyz_d, //The xyz vector corresponding to a single, scaled, detector pixel step in the w direction.
  //Offset variables are the offset of the (0,0,0) element of the sub-proj and/or sub-volume w.r.t. the centre of the overall CT reconstruction problem.
  //More importantly, it's the number required to convert device array indicies to geometric coordinates. Index + Offset = Coordinate
  int* NwOffset_d,
  int* NhOffset_d,
  int NxOffset,
  int NyOffset,
  int NzOffset,
  //The dimensions of the sub-proj and/or sub-vol arrays.
  int Nx ,
  int Ny ,
  int Nt )
{
  //Declare variables in shared memory
  __shared__ float volume[BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z];
  __shared__ float3 sourcePos_s, detectorPos_s, hStep_s, wStep_s, whPerp_s;
  //Declare variables in local memory
  int thread, tt;
  float ww, hh;
  float wDotW, hDotH, wDotH, iDotW, iDotH;
  float distanceFromSource;
  float3 objectPos, intersection;//, sourcePos_s, detectorPos_s, hStep_s, wStep_s, whPerp_s;
  int3 objectCoords;
  //Calculate thread number
  thread = threadIdx.x + BLOCK_SIZE_X * threadIdx.y + BLOCK_SIZE_X * BLOCK_SIZE_Y * threadIdx.z;
  //Zero volume
  volume[thread] = 0.0;
#warning try this out with just promoting the int3 in the relevant calculations there's only one of them (objectPos-sourcePos).
  //Work out position (relative to origin of total problem)
  objectCoords.x = threadIdx.x + BLOCK_SIZE_X * blockIdx.x;// + NxOffset;
  objectCoords.y = threadIdx.y + BLOCK_SIZE_Y * (blockIdx.y % (Ny / BLOCK_SIZE_Y));// + NyOffset;
  objectCoords.z = threadIdx.z + BLOCK_SIZE_Z * ((int)blockIdx.y / (int)(Ny / BLOCK_SIZE_Y));// + NzOffset;
  objectPos.x = (float)objectCoords.x + (float)NxOffset;
  objectPos.y = (float)objectCoords.y + (float)NyOffset;
  objectPos.z = (float)objectCoords.z + (float)NzOffset;
  __syncthreads();
  //Loop through all projections
  for( tt = 0 ; tt < Nt ; tt++ ){
    if(thread == 0){
      sourcePos_s = make_float3(sourcePosition_xyz_d[3*tt], sourcePosition_xyz_d[3*tt + 1], sourcePosition_xyz_d[3*tt + 2]);
      detectorPos_s = make_float3(detectorCentre_xyz_d[3*tt], detectorCentre_xyz_d[3*tt + 1], detectorCentre_xyz_d[3*tt + 2]);
      hStep_s = make_float3(hStep_xyz_d[3*tt], hStep_xyz_d[3*tt + 1], hStep_xyz_d[3*tt + 2]);
      wStep_s = make_float3(wStep_xyz_d[3*tt], wStep_xyz_d[3*tt + 1], wStep_xyz_d[3*tt + 2]);
      whPerp_s = cross(wStep_s, hStep_s);
    }
    __syncthreads();
    //Create 3-vectors for geometry calculations. Convert xc, yc, and zc to coordinates in the process.
    //detectorNormal = normalize(cross(wStep, hStep))
    //preCalculated1 = (detectorPos - sourcePos) dot detectorNormal
    //preCalculated2 = (detectorPos - sourcePos)
    distanceFromSource = dot(detectorPos_s - sourcePos_s, whPerp_s) / dot(objectPos - sourcePos_s, whPerp_s);
    intersection = distanceFromSource * (objectPos - sourcePos_s) + sourcePos_s - detectorPos_s;
    wDotW = dot(wStep_s, wStep_s);
    wDotH = dot(wStep_s, hStep_s);
    hDotH = dot(hStep_s, hStep_s);
    iDotW = dot(intersection, wStep_s);
    iDotH = dot(intersection, hStep_s);
    ww = (iDotW * hDotH - iDotH * wDotH) / (wDotW * hDotH - wDotH * wDotH);
    hh = (iDotH * wDotW - iDotW * wDotH) / (wDotW * hDotH - wDotH * wDotH);
    volume[thread] += tex3D(texRef, ww - (float)NwOffset_d[tt] + FLOAT_TEX_OFFSET, hh - (float)NhOffset_d[tt] + FLOAT_TEX_OFFSET, tt + FLOAT_TEX_OFFSET);
    __syncthreads();
  }
  
  //Convert xx, yy, and zz back into array indicies.
  //objectPos.x -= NxOffset;
  //objectPos.y -= NyOffset;
  //objectPos.z -= NzOffset;
  //Assign to volume array
  vol_d[objectCoords.x + Nx * objectCoords.y + Nx * Ny * objectCoords.z] = volume[thread];
  return;
}

//Boundaries of the subProjection need to be padded, so that we get zero for out-of-bounds texture fetches
__global__ void GPU_Katsevich_backproject_sinogram( 
  float* vol_d , 
  float* hMid_d , //hMid here is in geometric coordinates; an hMid of zero corresponds to a source and detector in the plane zz=0, ie. in the middle of the volume.
  float* sina_d , 
  float* cosa_d ,
  float sample_d ,
  float pixelWidth_d,
  //Offset variables are the offset of the (0,0,0) element of the sub-proj and/or sub-volume w.r.t. the centre of the overall CT reconstruction problem.
  //More importantly, it's the number required to convert device array indicies to geometric coordinates (i.e. + N?Offset converts index to coords).
  int* NwOffset_d , 
  int* NhOffset_d ,
  int NxOffset_d , 
  int NyOffset_d , 
  int NzOffset_d ,
  //The dimensions of the sub-proj and/or sub-vol arrays.
  int Nx_d , 
  int Ny_d , 
  int Nt_d )
{
  //Declare variables.
  __shared__ float volume[BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 4];
  __shared__ float sina_s , cosa_s , hMidplane_s , NwOffset_s , NhOffset_s, sample_s, pixelsFromVoxel_s;
  float temp;
  float hh , ww;
  int xx , yy , zz;
  int tt;
  int thread;
  //Calculate the index for this thread
  thread = threadIdx.x + BLOCK_SIZE_X * threadIdx.y + BLOCK_SIZE_X * BLOCK_SIZE_Y * threadIdx.z;
  //Put sample into shared memory
  if(thread==0){
    sample_s = sample_d;
    pixelsFromVoxel_s = 1.0f / pixelWidth_d;
  }
  __syncthreads();
  //Initialize volume array in shared memory
  volume[thread] = 0.0;
  volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z] = 0.0;
  volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 2] = 0.0;
  volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 3] = 0.0;
  //Work out xyz coordinates (relative to origin of total problem)
  xx = threadIdx.x + BLOCK_SIZE_X * blockIdx.x + NxOffset_d;
  yy = threadIdx.y + BLOCK_SIZE_Y * (blockIdx.y % (Ny_d / BLOCK_SIZE_Y)) + NyOffset_d;
  zz = threadIdx.z + BLOCK_SIZE_Z * 4 * ((int)blockIdx.y / (int)(Ny_d / BLOCK_SIZE_Y)) + NzOffset_d;
  //Loop over all projections.
  for( tt = 0 ; tt < Nt_d ; tt++ ){
    //Doing this is a bit inelegant, but it cuts execution time approximately in half.
    __syncthreads();
    if( thread == 0 ){
      sina_s = sina_d[tt];
      cosa_s = cosa_d[tt];
      NwOffset_s = (float)NwOffset_d[tt];
      NhOffset_s = (float)NhOffset_d[tt];
      hMidplane_s = hMid_d[tt];
    }
    __syncthreads();
    //Figure out proj_d array indicies.
    temp = sample_s / (sample_s + (float)xx * cosa_s + (float)yy * sina_s);
                
    ww = ((float)yy * cosa_s - (float)xx * sina_s ) * temp * pixelsFromVoxel_s - NwOffset_s + FLOAT_TEX_OFFSET;
    hh = ((float)zz - hMidplane_s) * temp * pixelsFromVoxel_s - NhOffset_s + FLOAT_TEX_OFFSET; //Adjusted so that the detector centre is locked to the source
    //Do the assignments
    volume[thread] += temp * tex3D(texRef, ww, hh, tt  + FLOAT_TEX_OFFSET);
    hh += ( BLOCK_SIZE_Z * temp * pixelsFromVoxel_s);
    volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 1] += temp * tex3D(texRef, ww, hh, tt  + FLOAT_TEX_OFFSET);
    hh += ( BLOCK_SIZE_Z * temp * pixelsFromVoxel_s);
    volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 2] += temp * tex3D(texRef, ww, hh, tt + FLOAT_TEX_OFFSET);       
    hh += ( BLOCK_SIZE_Z * temp * pixelsFromVoxel_s);
    volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 3] += temp * tex3D(texRef, ww, hh, tt + FLOAT_TEX_OFFSET);
  }
  //Convert xx, yy, and zz back into array indicies.
  xx -= NxOffset_d;
  yy -= NyOffset_d;
  zz -= NzOffset_d;
  //Assign to volume array
  vol_d[xx + Nx_d * yy + Nx_d * Ny_d * zz] = volume[thread];
  vol_d[xx + Nx_d * yy + Nx_d * Ny_d * ( zz + BLOCK_SIZE_Z * 1 )] = volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 1];
  vol_d[xx + Nx_d * yy + Nx_d * Ny_d * ( zz + BLOCK_SIZE_Z * 2 )] = volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 2];
  vol_d[xx + Nx_d * yy + Nx_d * Ny_d * ( zz + BLOCK_SIZE_Z * 3 )] = volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 3];
  return;
}

//Boundaries of the subProjection need to be padded, so that we get zero for out-of-bounds texture fetches
__global__ void GPU_Feldkamp_backproject_sinogram( 
  float* vol_d , 
  float* hMid_d , //hMid here is in geometric coordinates; an hMid of zero corresponds to a source in the plane hh=0, ie. in the middle of the volume.
  float* sina_d , 
  float* cosa_d , 
  float sample_d ,
  float pixelWidth_d,
  //Offset variables are the offset of the (0,0,0) element of the sub-proj and/or sub-volume w.r.t. the centre of the overall CT reconstruction problem.
  //More importantly, it's the number required to convert device array indicies to geometric coordinates.
  int* NwOffset_d , 
  int* NhOffset_d ,
  int NxOffset_d , 
  int NyOffset_d , 
  int NzOffset_d ,
  //The dimensions of the sub-proj and/or sub-vol arrays.
  int Nx_d , 
  int Ny_d , 
  int Nt_d )
{
  //Declare variables.
  __shared__ float volume[BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 4];
  __shared__ float sina_s , cosa_s , hMidplane_s , NwOffset_s , NhOffset_s, sample_s, pixelsFromVoxel_s;
  float temp;
  float hh , ww;
  int xx , yy , zz;
  int tt;
  int thread;
  //Calculate the index for this thread
  thread = threadIdx.x + BLOCK_SIZE_X * threadIdx.y + BLOCK_SIZE_X * BLOCK_SIZE_Y * threadIdx.z;
  //Put sample into shared memory
  if(thread==0){
    sample_s = sample_d;
    pixelsFromVoxel_s = 1.0f / pixelWidth_d;
  }
  __syncthreads();
  //Initialize volume array in shared memory
  volume[thread] = 0.0;
  volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z] = 0.0;
  volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 2] = 0.0;
  volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 3] = 0.0;
  //Work out position (relative to origin of total problem)
  xx = threadIdx.x + BLOCK_SIZE_X * blockIdx.x + NxOffset_d;
  yy = threadIdx.y + BLOCK_SIZE_Y * (blockIdx.y % (Ny_d / BLOCK_SIZE_Y)) + NyOffset_d;
  zz = threadIdx.z + BLOCK_SIZE_Z * 4 * ((int)blockIdx.y / (int)(Ny_d / BLOCK_SIZE_Y)) + NzOffset_d;
  //Loop over all projections.
  for( tt = 0 ; tt < Nt_d ; tt++ ){
    //Doing this is a bit inelegant, but it cuts execution time approximately in half.
    __syncthreads();
    if( thread == 0 ){
      sina_s = sina_d[tt];
      cosa_s = cosa_d[tt];
      NwOffset_s = (float)NwOffset_d[tt];
      NhOffset_s = (float)NhOffset_d[tt];
      hMidplane_s = hMid_d[tt];
    }
    __syncthreads();
    //Figure out proj_d array indicies.
    temp = sample_s / (sample_s + (float)xx * cosa_s + (float)yy * sina_s);
    ww = ((float)yy * cosa_s - (float)xx * sina_s ) * temp * pixelsFromVoxel_s - NwOffset_s + FLOAT_TEX_OFFSET;
    hh = ((float)zz - hMidplane_s) * temp * pixelsFromVoxel_s - NhOffset_s + FLOAT_TEX_OFFSET; //Adjusted so that the detector centre is locked to the source plane.
    //Do the assignments
    volume[thread] += temp * temp * tex3D(texRef, ww, hh, tt  + FLOAT_TEX_OFFSET);
    hh += ( BLOCK_SIZE_Z * temp * pixelsFromVoxel_s);
    volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 1] += temp * temp * tex3D(texRef, ww, hh, tt  + FLOAT_TEX_OFFSET);
    hh += ( BLOCK_SIZE_Z * temp * pixelsFromVoxel_s);
    volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 2] += temp * temp * tex3D(texRef, ww, hh, tt + FLOAT_TEX_OFFSET);
    hh += ( BLOCK_SIZE_Z * temp * pixelsFromVoxel_s);
    volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 3] += temp * temp * tex3D(texRef, ww, hh, tt + FLOAT_TEX_OFFSET);
  }
  //Convert xx, yy, and zz back into array indicies.
  xx -= NxOffset_d;
  yy -= NyOffset_d;
  zz -= NzOffset_d;
  //Assign to volume array
  vol_d[xx + Nx_d * yy + Nx_d * Ny_d * zz] = volume[thread];
  vol_d[xx + Nx_d * yy + Nx_d * Ny_d * ( zz + BLOCK_SIZE_Z * 1 )] = volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 1];
  vol_d[xx + Nx_d * yy + Nx_d * Ny_d * ( zz + BLOCK_SIZE_Z * 2 )] = volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 2];
  vol_d[xx + Nx_d * yy + Nx_d * Ny_d * ( zz + BLOCK_SIZE_Z * 3 )] = volume[thread + BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * 3];
  return;
}


//****************************************
//Host functions
//****************************************


//Returns memory of current CUDA device in MB.
int getDeviceRAM( )
{
  //declarations
  int deviceNumber = 0;
  int memMB = 0;
  cudaDeviceProp prop;
  //Call appropriate CUDA commands
  cudaGetDevice( &deviceNumber );
  cudaGetDeviceProperties( &prop , deviceNumber );
  memMB = prop.totalGlobalMem / (1024 * 1024 );
  return memMB;
}


//Returns the number of CUDA capable devices.
int getNumDevices( )
{
  int count = 0;
  int CUDAcount = 0;
  int ii = 0;
  cudaDeviceProp prop;
  //Get number of devices
  cudaGetDeviceCount(&count);
  for(ii = 0; ii < count; ii++) {
    if(cudaGetDeviceProperties( &prop, ii ) == cudaSuccess) {
      if(prop.major >= 1) {
        CUDAcount++;
      }
    }
  }
  return CUDAcount;
}


//Sets this thread up to use the Nth CUDA-capable device.
void setDeviceNum( int deviceNumber )
{ 
  int count = 0;
  int CUDAcount = 0;
  int ii = 0;
  cudaDeviceProp prop;
  CUDAcount = getNumDevices( );
  //Check that the device number is valid.
  if( deviceNumber >= CUDAcount ){
    //cout << "Tried to call device number " << decviceNumber << ", when only " << CUDAcount << "devices exist." << endl << flush;
    exit(0);
  }
  //Count through devices until we find the right one. Necessary because not all devices are CUDA capable.
  cudaGetDeviceCount(&count);
  for(ii = 0; ii < count; ii++) {
    if(cudaGetDeviceProperties( &prop, ii ) == cudaSuccess) {
      if(prop.major >= 1) {
        if( deviceNumber == 0 ){
          cudaSetDevice(ii);
          break;
        }
        deviceNumber--;
      }
    }
  }
  return;
}


void newProjectSubVolume(
  int ss , //The index of this sub-problem
  coneBeamGeometry* geometry ,
  //These arrays need to be allocated, and filled with data.
  gpuTexArray* deviceVolume ,
  gpuFltArray* deviceProj ,
  gpuFltArray* sourcePosition_xyz_d ,
  gpuFltArray* detectorCentre_xyz_d ,
  gpuFltArray* hStep_xyz_d ,
  gpuFltArray* wStep_xyz_d ,
  gpuIntArray* sStart_d ,
  gpuIntArray* sNumSteps_d ,
  gpuIntArray* NwOffset_d ,
  gpuIntArray* NhOffset_d ,
  int executionMode //Either __SYNC__ or __ASYNC__. 
)
{
  //Declare host variables
  int ttLocal;
  int xxOffset, yyOffset, zzOffset;
  int *NwOffset, *NhOffset;
  //Allocate host memory
  NwOffset = new int [geometry->subProjDim[2]];
  NhOffset = new int [geometry->subProjDim[2]];
  //Declare block and grid sizes on Device
  int sharedMemSize = 10*1024;
  dim3 Block(BLOCK_SIZE_W, BLOCK_SIZE_H);
  dim3 Grid(geometry->subProjDim[0] / BLOCK_SIZE_W, geometry->subProjDim[1] * geometry->subProjDim[2] / BLOCK_SIZE_H);
  //Populate host variables
  for(ttLocal = 0; ttLocal < geometry->subProjDim[2]; ttLocal++){
    NwOffset[ttLocal] = geometry->wwStart[ss][ttLocal] - geometry->projDim[0] / 2;
    NhOffset[ttLocal] = geometry->hhStart[ss][ttLocal] - geometry->projDim[1] / 2;
  }
  xxOffset = geometry->xxStart[ss] - geometry->volDim[0] / 2 - 1;
  yyOffset = geometry->yyStart[ss] - geometry->volDim[1] / 2 - 1;
  zzOffset = geometry->zzStart[ss] - geometry->volDim[2] / 2 - 1;
  //Populate device variables. Note that _xyz variables have 3 elements per viewing angle, hence the multiplication of the index by 3.
  sourcePosition_xyz_d->sendDataToGPU(geometry->sourcePosition_xyz + 3 * geometry->ttStart[ss]);
  detectorCentre_xyz_d->sendDataToGPU(geometry->detectorCentre_xyz + 3 * geometry->ttStart[ss]);
  hStep_xyz_d->sendDataToGPU(geometry->hStep_xyz + 3 * geometry->ttStart[ss]);
  wStep_xyz_d->sendDataToGPU(geometry->wStep_xyz + 3 * geometry->ttStart[ss]);
  sStart_d->sendDataToGPU(geometry->sweepStart[ss]);
  sNumSteps_d->sendDataToGPU(geometry->sweepNumSteps[ss]);
  NwOffset_d->sendDataToGPU(NwOffset);
  NhOffset_d->sendDataToGPU(NhOffset);
  //Get pointers to device arrays
  float* projData_d_ptr = deviceProj->getDevicePtr();
  float* sourcePosition_xyz_d_ptr = sourcePosition_xyz_d->getDevicePtr();
  float* detectorCentre_xyz_d_ptr = detectorCentre_xyz_d->getDevicePtr();
  float* hStep_xyz_d_ptr = hStep_xyz_d->getDevicePtr();
  float* wStep_xyz_d_ptr = wStep_xyz_d->getDevicePtr();
  int* sStart_d_ptr = sStart_d->getDevicePtr();
  int* sNumSteps_d_ptr = sNumSteps_d->getDevicePtr();
  int* NwOffset_d_ptr = NwOffset_d->getDevicePtr();
  int* NhOffset_d_ptr = NhOffset_d->getDevicePtr();
  //Bind to texture
  deviceVolume->bindToProjTexture();
  //Call projection kernel
  GPU_new_project_volume<<<Grid,Block,sharedMemSize>>>(projData_d_ptr, sourcePosition_xyz_d_ptr, detectorCentre_xyz_d_ptr, hStep_xyz_d_ptr, wStep_xyz_d_ptr, sStart_d_ptr, sNumSteps_d_ptr, NwOffset_d_ptr, NhOffset_d_ptr, xxOffset, yyOffset, zzOffset, geometry->subProjDim[0], geometry->subProjDim[1]);
  //If necessary, wait for the GPU to be done before returning.
  if( executionMode == __SYNC__ ){
    cudaThreadSynchronize();
  } 
  //Unbind the texture
  //deviceVolume->unbindProjTexture();
  //Free memory allocated to host arrays
  delete [] NwOffset;
  delete [] NhOffset;
  //Return
  return; 
}

void raytraceBackprojectSubVolume(
  int ss , //The index of this sub-problem
  coneBeamGeometry* geometry ,
  //These arrays need to be allocated, and filled with data.
  gpuFltArray* deviceVolume ,
  gpuTexArray* deviceProj ,
  //The following arrays need to be allocated.
  gpuFltArray* sourcePosition_xyz_d ,
  gpuFltArray* detectorCentre_xyz_d ,
  gpuFltArray* hStep_xyz_d ,
  gpuFltArray* wStep_xyz_d ,
  gpuIntArray* NwOffset_d ,
  gpuIntArray* NhOffset_d ,
  int executionMode //Either __SYNC__ or __ASYNC__. 
)
{
  //Declare host variables
  int ttLocal;
  int xxOffset, yyOffset, zzOffset;
  int *NwOffset, *NhOffset;
  //Allocate host memory
  NwOffset = new int [geometry->subProjDim[2]];
  NhOffset = new int [geometry->subProjDim[2]];
  //Declare block and grid sizes on Device
  int sharedMemSize = 10*1024;
  dim3 Block( BLOCK_SIZE_X , BLOCK_SIZE_Y , BLOCK_SIZE_Z );
  dim3 Grid( geometry->subVolDim[0] / BLOCK_SIZE_X , geometry->subVolDim[1] * geometry->subVolDim[2] / ( BLOCK_SIZE_Z * BLOCK_SIZE_Y ) );
  //Populate host variables
  for(ttLocal = 0; ttLocal < geometry->subProjDim[2]; ttLocal++){
    NwOffset[ttLocal] = geometry->wwStart[ss][ttLocal] - geometry->projDim[0] / 2;
    NhOffset[ttLocal] = geometry->hhStart[ss][ttLocal] - geometry->projDim[1] / 2;
  }
  xxOffset = geometry->xxStart[ss] - geometry->volDim[0] / 2;
  yyOffset = geometry->yyStart[ss] - geometry->volDim[1] / 2;
  zzOffset = geometry->zzStart[ss] - geometry->volDim[2] / 2;
  //Populate device variables. Note that _xyz variables have 3 elements per viewing angle, hence the multiplication of the index by 3.
  sourcePosition_xyz_d->sendDataToGPU(geometry->sourcePosition_xyz + 3 * geometry->ttStart[ss]);
  detectorCentre_xyz_d->sendDataToGPU(geometry->detectorCentre_xyz + 3 * geometry->ttStart[ss]);
  hStep_xyz_d->sendDataToGPU(geometry->hStep_xyz + 3 * geometry->ttStart[ss]);
  wStep_xyz_d->sendDataToGPU(geometry->wStep_xyz + 3 * geometry->ttStart[ss]);
  NwOffset_d->sendDataToGPU(NwOffset);
  NhOffset_d->sendDataToGPU(NhOffset);
  //Get pointers to device arrays
  float* volData_d_ptr = deviceVolume->getDevicePtr();
  float* sourcePosition_xyz_d_ptr = sourcePosition_xyz_d->getDevicePtr();
  float* detectorCentre_xyz_d_ptr = detectorCentre_xyz_d->getDevicePtr();
  float* hStep_xyz_d_ptr = hStep_xyz_d->getDevicePtr();
  float* wStep_xyz_d_ptr = wStep_xyz_d->getDevicePtr();
  int* NwOffset_d_ptr = NwOffset_d->getDevicePtr();
  int* NhOffset_d_ptr = NhOffset_d->getDevicePtr();
  //Bind to texture
  deviceProj->bindToProjTexture();
  //Call projection kernel
  GPU_unweighted_backproject_sinogram<<<Grid,Block,sharedMemSize>>>(volData_d_ptr, sourcePosition_xyz_d_ptr, detectorCentre_xyz_d_ptr, hStep_xyz_d_ptr, wStep_xyz_d_ptr, NwOffset_d_ptr, NhOffset_d_ptr, xxOffset, yyOffset, zzOffset, geometry->subVolDim[0], geometry->subVolDim[1], geometry->subProjDim[2]);
  //If necessary, wait for the GPU to be done before returning.
  if( executionMode == __SYNC__ ){
    cudaThreadSynchronize();
  }
  //Unbind the texture
  //deviceProj->unbindProjTexture();
  //Free memory allocated to host arrays
  delete [] NwOffset;
  delete [] NhOffset;
  //Return
  return;
}

void backprojectSubVolume(
  int ss , //The index of this sub-problem
  coneBeamGeometry* geometry ,
  //These arrays need to be allocated, and filled with data.
  gpuFltArray* deviceVolume ,
  gpuTexArray* deviceProj ,
  //These arrays need to be allocated, but not filled with data.
  gpuFltArray* deviceHmid ,
  gpuFltArray* deviceSina ,
  gpuFltArray* deviceCosa ,
  gpuIntArray* devicewwOffset ,
  gpuIntArray* devicehhOffset ,
  int executionMode , //Either __SYNC__ or __ASYNC__.
  int kernelMode //Either __KATSEVICH__ or __FELDKAMP__
)
{
  //Declare host variables
  int xxOffset;
  int yyOffset;
  int zzOffset;
  int* hhOffset;
  int* wwOffset;
  int tt;
  //Declare block and grid sizes on Device
  dim3 Block( BLOCK_SIZE_X , BLOCK_SIZE_Y , BLOCK_SIZE_Z );
  dim3 Grid( geometry->subVolDim[0] / BLOCK_SIZE_X , geometry->subVolDim[1] * geometry->subVolDim[2] / ( 4 * BLOCK_SIZE_Z * BLOCK_SIZE_Y ) );
  //Get pointers from device.
  float* d_volData = deviceVolume->getDevicePtr( );
  float* d_hMidPlane = deviceHmid->getDevicePtr( );
  float* d_sina = deviceSina->getDevicePtr( );
  float* d_cosa = deviceCosa->getDevicePtr( );
  int* d_wwOffset = devicewwOffset->getDevicePtr( );
  int* d_hhOffset = devicehhOffset->getDevicePtr( );
  //Allocate memory for arrays on host.
  wwOffset = new int [geometry->subProjDim[2]];
  hhOffset = new int [geometry->subProjDim[2]];
  //Calculate offsets (i.e. the starting point of each projection, relative to the centre).
  xxOffset = geometry->xxStart[ss] - geometry->volDim[0] / 2;
  yyOffset = geometry->yyStart[ss] - geometry->volDim[1] / 2;
  zzOffset = geometry->zzStart[ss] - geometry->volDim[2] / 2;
  for( tt = 0 ; tt < geometry->subProjDim[2] ; tt++ ){
    wwOffset[tt] = geometry->wwStart[ss][tt] - geometry->projDim[0] / 2;
    hhOffset[tt] = geometry->hhStart[ss][tt] - geometry->projDim[1] / 2;
  }
  //Copy misc. arrays to device
  deviceHmid->sendDataToGPU( geometry->hMidPlane + geometry->ttStart[ss] );
  deviceSina->sendDataToGPU( geometry->sina + geometry->ttStart[ss] );
  deviceCosa->sendDataToGPU( geometry->cosa + geometry->ttStart[ss] );
  devicewwOffset->sendDataToGPU( wwOffset );
  devicehhOffset->sendDataToGPU( hhOffset );
  //Bind to texture
  deviceProj->bindToProjTexture();
  //Call Kernel
  if (kernelMode == __KATSEVICH__){
    GPU_Katsevich_backproject_sinogram<<<Grid,Block>>>(d_volData, d_hMidPlane, d_sina, d_cosa, geometry->sourceSampleDistance, geometry->pixelWidth,
                d_wwOffset, d_hhOffset, xxOffset, yyOffset, zzOffset,
                geometry->subVolDim[0], geometry->subVolDim[1], geometry->subProjDim[2]);
  }else{
    GPU_Feldkamp_backproject_sinogram<<<Grid,Block>>>(d_volData, d_hMidPlane, d_sina, d_cosa, geometry->sourceSampleDistance, geometry->pixelWidth,
                d_wwOffset, d_hhOffset, xxOffset, yyOffset, zzOffset,
                geometry->subVolDim[0], geometry->subVolDim[1], geometry->subProjDim[2]);
  }
  //If necessary, wait for the GPU to be done before returning.
  if( executionMode == __SYNC__ ){
    cudaThreadSynchronize( );
  } 
  //Unbind the texture
  //deviceProj->unbindProjTexture();  
  //Free memory allocated to host arrays
  delete [] wwOffset;
  delete [] hhOffset;   
  return; 
}


/******************************************************************************************************

ROUTINES FOR GPU TEX ARRAY CLASS.

class gpuTexArray : public gpuCachedArray
{
  public:
    gpuTexArray( int initDims[3] );
    ~gpuTexArray( );
    void sendDataToGPU( float* hostMemory );
    void sendDataToGPU( float* hostMemory , int hostDims[3] , int xOffset , int yOffset , int zOffset);
    void bindToProjTexture( );
    void unbindProjTexture( );
    void sendCacheToGPU( );
 
  private:
    cudaArray*          deviceData_ ,
    cudaChannelFormatDesc    channelDesc_ ,
    cudaExtent          extent_ ,
    int             isTexture_
};

*******************************************************************************************************/

gpuTexArray::gpuTexArray( int initDims[3] )
{

  _dims[0] = initDims[0];
  _dims[1] = initDims[1];
  _dims[2] = initDims[2];

  extent_.width = _dims[0];
  extent_.height = _dims[1];
  extent_.depth = _dims[2];

  channelDesc_ = cudaCreateChannelDesc<float>();
  
  cudaMalloc3DArray(&deviceData_, &channelDesc_, extent_);
  
  isTexture_ = 0;

}

gpuTexArray::~gpuTexArray( )
{

  if( isTexture_ != 0 ) {
    cudaUnbindTexture( texRef );
    isTexture_ = 0;
  }

  cudaFreeArray(deviceData_);

}

void gpuTexArray::sendDataToGPU( float* hostMemory )
{

  int wasTexture = 0;

  //Unbind if necessary - wierd things happen when we write to textures.
  if( isTexture_ != 0 ) {
    unbindProjTexture( );
    wasTexture = 1;
  }
  

  //set up copy parameters
  cudaMemcpy3DParms copyParams = {0};
  copyParams.extent = make_cudaExtent( _dims[0] , _dims[1] , _dims[2] );
  copyParams.kind = cudaMemcpyHostToDevice;
  copyParams.dstArray = deviceData_;
  
  
  // The pitched pointer is really tricky to get right. We give the
  // pointer, then the pitch of a row, then the number of elements in a row, then the
  // height, and we omit the 3rd dimension.
  copyParams.srcPtr = make_cudaPitchedPtr( (void*) hostMemory , _dims[0] * sizeof(float) , _dims[0] , _dims[1] );
  
  cudaMemcpy3D( &copyParams );
  
  //Re-bind if necessary
  if( wasTexture == 1 ){
    bindToProjTexture( );
  }
  
  return;
}

void gpuTexArray::sendDataToGPU( float* hostMemory , int hostDims[3] , int xOffset , int yOffset , int zOffset  )
{

  int wasTexture = 0;

  //Unbind if necessary - wierd things happen when we write to textures.
  if( isTexture_ != 0 ) {
    unbindProjTexture( );
    wasTexture = 1;
  }

  //set up copy parameters
  cudaMemcpy3DParms copyParams = {0};
  copyParams.extent = make_cudaExtent( hostDims[0] , hostDims[1] , hostDims[2] );
  copyParams.kind = cudaMemcpyHostToDevice;
  
  copyParams.dstArray = deviceData_;
  copyParams.dstPos.x = xOffset;
  copyParams.dstPos.y = yOffset;
  copyParams.dstPos.z = zOffset;
  
  // The pitched pointer is really tricky to get right. We give the
  // pointer, then the pitch of a row, then the number of elements in a row, then the
  // height, and we omit the 3rd dimension.
  copyParams.srcPtr = make_cudaPitchedPtr( (void*) hostMemory , hostDims[0] * sizeof(float) , hostDims[0] , hostDims[1] );
  
  //Perform the copy
  cudaMemcpy3D( &copyParams );
  
  //Re-bind if necessary
  if( wasTexture == 1 ){
    bindToProjTexture( );
  }
  
  return;
}

void gpuTexArray::sendCacheToGPU( )
{
  sendDataToGPU(_hostCache);
  return;
}

void gpuTexArray::bindToProjTexture( )
{
  
  //Set texture toggles 
  texRef.addressMode[0] = cudaAddressModeClamp; 
  texRef.addressMode[1] = cudaAddressModeClamp;
  texRef.addressMode[2] = cudaAddressModeClamp; 
  texRef.filterMode = cudaFilterModeLinear; 
  texRef.normalized = 0;

  //Bind texture
  cudaBindTextureToArray( texRef , deviceData_ , channelDesc_ );

  //Set flag and return
  isTexture_ = 1;
  return;
}

void gpuTexArray::unbindProjTexture( )
{

  if( isTexture_ != 0 ) {
    cudaUnbindTexture( texRef );
    isTexture_ = 0;
  }

  return;
}

/*******************************************************************************************************
ROUTINES FOR GPU FLOAT ARRAY CLASS

class gpuFltArray : public gpuCachedArray
{
public:
        gpuFltArray( int initDims[3] );
        gpuFltArray( int singleDim );
        ~gpuFltArray( );

        void sendDataToGPU( float* hostMemory );
        void getDataFromGPU( float* hostMemory );
        void sendCacheToGPU( );
        void getCacheFromGPU( );

        float* getDevicePtr( );

private:
        float*  _deviceData;

        void _init( );
};

*******************************************************************************************************/


gpuFltArray::gpuFltArray( int initDims[3] )
{
  //Store the dimensions to the internal class variable.
  _dims[0] = initDims[0];
  _dims[1] = initDims[1];
  _dims[2] = initDims[2];
  //  printf("File: %s.   Line: %d.\n", __FILE__ , __LINE__ );
  //fflush(stdout);
  //Allocate the Memory on the device
  _init();
}

gpuFltArray::gpuFltArray( int singleDim )
{
  //Store the dimensions appropriately.
  _dims[0] = singleDim;
  _dims[1] = 1;
  _dims[2] = 1;
  //      printf("File: %s.   Line: %d.\n", __FILE__ , __LINE__ );
  //fflush(stdout);
  //Allocate the memory on the device
  _init();
}

gpuFltArray::~gpuFltArray( )
{
  //De-allocate the device memory.
  cudaFree( _deviceData );
}

void gpuFltArray::sendDataToGPU( float* hostMemory )
{
  //Copy the matched memory to the GPU.
  cudaMemcpy( _deviceData , hostMemory , sizeof(float) * _dims[0] * _dims[1] * _dims[2] , cudaMemcpyHostToDevice );
  return;
}

void gpuFltArray::getDataFromGPU( float* hostMemory )
{
  cudaMemcpy( hostMemory , _deviceData , sizeof(float) * _dims[0] * _dims[1] * _dims[2] , cudaMemcpyDeviceToHost );
  return;
}

void gpuFltArray::sendCacheToGPU( )
{
  _initCache();
  sendDataToGPU(_hostCache);
  return;
}

void gpuFltArray::getCacheFromGPU( )
{
  _initCache();
  _zeroCache();
  getDataFromGPU(_hostCache);
  return;
}

float* gpuFltArray::getDevicePtr( )
{
  return _deviceData;
}

void gpuFltArray::_init()
{
  cudaMalloc( (void**) &_deviceData , sizeof(float) * _dims[0] * _dims[1] * _dims[2] );
  return;
}



/******************************************************************************************************
 
 ROUTINES FOR CACHED ARRAY SUPER CLASS
 
 class gpuCachedArray {
 public:
 gpuCachedArray( );
 ~gpuCachedArray( );
 
 void fillCache(float* source, int sourceStart[3], int sourceEnd[3], int xStart, int yStart, int zStart);
 void fillCache(float* source, int sourceWHDims[2], int sourcePDim, int* sourceWOffset, int* sourceHOffset, float* sourceNorms, int* wStart, int* hStart, int pStart);
 void addCache(float* dest, int destStart[3], int destEnd[3], int xStart, int yStart, int zStart);
 void addCache(float* dest, int destWHDims[2], int destPDim, int* destWOffset, int* destHOffset, float* destNorms, int* wStart, int* hStart, int pStart);
 
 int getDim( int dimension );
 
 protected:
 int     _dims[3];
 float*  _hostCache;
 
 void _initCache( );
 void _zeroCache( );
 void _freeCache( );
 };
 
 ******************************************************************************************************/

gpuCachedArray::gpuCachedArray()
{
  _dims[0] = 1;
  _dims[1] = 1;
  _dims[2] = 1;
  _hostCache = NULL;
}

gpuCachedArray::~gpuCachedArray()
{
  _freeCache();
}

void gpuCachedArray::fillCache(float* source, int sourceStart[3], int sourceEnd[3], int xStart, int yStart, int zStart)
{
  int sourceWHDims[2];
  int sourcePDim;
  int sourcePOffset;
  int* sourceWOffset;
  int* sourceHOffset;
  float* sourceNorms;
  int* wStart;
  int* hStart;
  int pStart;
  int pp;
  //Converting volume arguments to projection arguments
  sourceWHDims[0] = sourceEnd[0] - sourceStart[0];
  sourceWHDims[1] = sourceEnd[1] - sourceStart[1];
  sourcePDim = sourceEnd[2] - sourceStart[2];
  sourcePOffset = sourceStart[2];
  pStart = zStart;
  //Allocate arrays
  sourceWOffset = (int*) calloc(sourcePDim, sizeof(int));
  sourceHOffset = (int*) calloc(sourcePDim, sizeof(int));
  sourceNorms = (float*) calloc(sourcePDim, sizeof(float));
  wStart = (int*) calloc(_dims[2], sizeof(int));
  hStart = (int*) calloc(_dims[2], sizeof(int));
  //Populate arrays
  for(pp=0; pp<sourcePDim; pp++){
    sourceWOffset[pp] = sourceStart[0];
    sourceHOffset[pp] = sourceStart[1];
    sourceNorms[pp] = 1.0f;
  }
  for(pp=0; pp<_dims[2]; pp++){
    wStart[pp] = xStart;
    hStart[pp] = yStart;
  }
  //Call more general routine
  fillCache(source, sourceWHDims, sourcePDim, sourceWOffset, sourceHOffset, sourcePOffset, sourceNorms, wStart, hStart, pStart);
  //Free arrays
  delete [] sourceWOffset;
  delete [] sourceHOffset;
  delete [] wStart;
  delete [] hStart;
  return;
}

void gpuCachedArray::fillCache(float* source, int sourceWHDims[2], int sourcePDim, int* sourceWOffset, int* sourceHOffset, int sourcePOffset, float* sourceNorms, int* wStart, int* hStart, int pStart)
{
  //Allocating variables. ww, hh and pp are array indexes in the Cache, offset from zero by wStart[pp], hStart[pp], and pStart.
  //wwSource, hhSource, and ppSource are array indexes in the source array, offset from zero by sourceWOffset, and sourceHOffset.
  int pp, hh, ww;
  int ppSource, hhSource, wwSource;
  int ppMax, ppMin, hhMax, hhMin, wwMax, wwMin;
  long long Nw, Nwh, NwSource, NwhSource;
  long long index, sourceIndex; 
  //Calculate parameters for array indicies
  Nw = _dims[0];
  NwSource = sourceWHDims[0];
  Nwh = _dims[1] * Nw;
  NwhSource = sourceWHDims[1] * NwSource;
  //Ensure Cache is available
  _initCache();
  _zeroCache();
  //Loop over all necessary
  ppMin = max(0, 0 - pStart + sourcePOffset);
  ppMax = min(_dims[2], sourcePDim - pStart + sourcePOffset);
  for(pp=ppMin; pp<ppMax; pp++){
    ppSource = pp + pStart - sourcePOffset;
    hhMin = max(0, 0 - hStart[pp] + sourceHOffset[ppSource]);
    hhMax = min(_dims[1], sourceWHDims[1] - hStart[pp] + sourceHOffset[ppSource]);
    for(hh=hhMin; hh<hhMax; hh++){
      hhSource = hh + hStart[pp] - sourceHOffset[ppSource];
      wwMin = max(0, 0 - wStart[pp] + sourceWOffset[ppSource]);
      wwMax = min(_dims[0], sourceWHDims[0] - wStart[pp] + sourceWOffset[ppSource]);
      index = wwMin + hh * Nw + pp * Nwh;
      sourceIndex = (wwMin + wStart[pp] - sourceWOffset[ppSource]) + hhSource * NwSource + ppSource * NwhSource;
      for(ww=wwMin; ww<wwMax; ww++){
        //wwSource = ww + wStart[pp] - sourceWOffset[ppSource];
        _hostCache[index] = sourceNorms[ppSource] * source[sourceIndex];
        index += 1;
        sourceIndex += 1;
      }
    }
  }
  return;
}

void gpuCachedArray::addCache(float* dest, int destStart[3], int destEnd[3], int xStart, int yStart, int zStart)
{
  int   destWHDims[2];
  int   destPDim;
  int destPOffset; 
  int*   destWOffset;
  int*   destHOffset;
  float*   destNorms;
  int* wStart;
  int* hStart;
  int pStart;
  int pp;
  //Converting volume arguments to projection arguments
  destWHDims[0] =   destEnd[0] -   destStart[0];
  destWHDims[1] =   destEnd[1] -   destStart[1];
  destPDim =   destEnd[2] -   destStart[2];
  destPOffset = destStart[2];
  pStart = zStart;
  //Allocate arrays
  destWOffset = (int*) calloc(  destPDim, sizeof(int));
  destHOffset = (int*) calloc(  destPDim, sizeof(int));
  destNorms = (float*) calloc(  destPDim, sizeof(float));
  wStart = (int*) calloc(_dims[2], sizeof(int));
  hStart = (int*) calloc(_dims[2], sizeof(int));
  //Populate arrays
  for(pp=0; pp<destPDim; pp++){
    destWOffset[pp] = destStart[0];
    destHOffset[pp] = destStart[1];
    destNorms[pp] = 1.0f;
  }
  for(pp=0; pp<_dims[2]; pp++){
    wStart[pp] = xStart;
    hStart[pp] = yStart;
  }
  //Call more general routine
  addCache(dest, destWHDims, destPDim, destWOffset, destHOffset, destPOffset, destNorms, wStart, hStart, pStart);
  //Free arrays
  delete [] destWOffset;
  delete [] destHOffset;
  delete [] wStart;
  delete [] hStart;
  return;
}

void gpuCachedArray::addCache(float* dest, int destWHDims[2], int destPDim, int* destWOffset, int* destHOffset, int destPOffset, float* destNorms, int* wStart, int* hStart, int pStart)
{
  //Allocating variables. ww, hh and pp are array indexes in the Cache, offset from zero by wStart[pp], hStart[pp], and pStart.
  //wwDest, hhDest, and ppDest are array indexes in the dest array, offset from zero by destWOffset, and destHOffset.
  int pp, hh, ww;
  int ppDest, hhDest, wwDest;
  int ppMax, ppMin, hhMax, hhMin, wwMax, wwMin;
  long long Nw, Nwh, NwDest, NwhDest;
  long long index, destIndex;
  //Create long long array indicies, to avoid problems when indexing absurdly large arrays.
  Nw = _dims[0];
  Nwh = Nw * _dims[1];
  NwDest = destWHDims[0];
  NwhDest = NwDest * destWHDims[1];
  //Ensure Cache is available
  _initCache();
  //Loop over all necessary
  ppMin = max(0, 0 - pStart + destPOffset);
  ppMax = min(_dims[2], destPDim - pStart + destPOffset);
  for(pp=ppMin; pp<ppMax; pp++){
    ppDest = pp + pStart - destPOffset;
    hhMin = max(0, 0 - hStart[pp] + destHOffset[ppDest]);
    hhMax = min(_dims[1], destWHDims[1] - hStart[pp] + destHOffset[ppDest]);
    for(hh=hhMin; hh<hhMax; hh++){
      hhDest = hh + hStart[pp] - destHOffset[ppDest];
      wwMin = max(0, 0 - wStart[pp] + destWOffset[ppDest]);
      wwMax = min(_dims[0], destWHDims[0] - wStart[pp] + destWOffset[ppDest]);
      index = wwMin + hh*Nw + pp*Nwh;
      destIndex = wwMin + wStart[pp] - destWOffset[ppDest] + hhDest*NwDest + ppDest*NwhDest;
      for(ww=wwMin; ww<wwMax; ww++){
        //wwDest = ww + wStart[pp] - destWOffset[ppDest];
        dest[destIndex] += destNorms[ppDest] * _hostCache[index];
        index += 1;
        destIndex += 1;
      }
    }
  }
  return;
}

int gpuCachedArray::getDim( int dimension )
{
  return _dims[dimension];
}

void gpuCachedArray::zeroCacheBoundary( )
{
  _initCache();
  int xx, yy, zz;
    for( zz = 0; zz < _dims[2]; zz++ ) {
      for( yy = 0 ; yy < _dims[1] ; yy++ ){
        xx = 0;
        _hostCache[ xx + yy * _dims[0] + zz * _dims[1] * _dims[0]] = 0.0f;
        xx = _dims[0] - 1;
        _hostCache[ xx + yy * _dims[0] + zz * _dims[1] * _dims[0] ] = 0.0f;
      }
    }
  for( zz = 0 ; zz < _dims[2] ; zz++ ){
    for( xx = 0 ; xx < _dims[0] ; xx++ ){
      yy = 0;
      _hostCache[ xx + yy * _dims[0] + zz * _dims[1] * _dims[0] ] = 0.0f;
      yy = _dims[1] - 1;
      _hostCache[ xx + yy * _dims[0] + zz * _dims[1] * _dims[0] ] = 0.0f;
    }
  }
  for( yy = 0 ; yy < _dims[1] ; yy++ ){
    for( xx = 0 ; xx < _dims[0] ; xx++ ){
      zz = 0;
      _hostCache[ xx + yy * _dims[0] + zz * _dims[1] * _dims[0] ] = 0.0f;
      zz = _dims[2] - 1;
      _hostCache[ xx + yy * _dims[0] + zz * _dims[1] * _dims[0] ] = 0.0f;
    }
  }
  return;
}

void gpuCachedArray::_initCache()
{
  if(! _hostCache){
    _hostCache = (float*) calloc(_dims[0] * _dims[1] * _dims[2], sizeof(float));
    _zeroCache();
  }
  return;
}

void gpuCachedArray::_zeroCache()
{
  memset(_hostCache, 0, _dims[0] * _dims[1] * _dims[2] * sizeof(float));
  return;
}

void gpuCachedArray::_freeCache()
{
  if (_hostCache){delete [] _hostCache;}
  _hostCache = NULL;
  return;
}

/******************************************************************************************************

ROUTINES FOR GPU INT ARRAY CLASS

******************************************************************************************************/


gpuIntArray::gpuIntArray( int initDims[3] )
{

  dims_[0] = initDims[0];
  dims_[1] = initDims[1];
  dims_[2] = initDims[2];

  cudaMalloc( (void**) &deviceData_ , sizeof(int) * dims_[0] * dims_[1] * dims_[2] );

}

gpuIntArray::gpuIntArray(int singleDim)
{
        dims_[0] = singleDim;
        dims_[1] = 1;
        dims_[2] = 2;

        cudaMalloc( (void**) &deviceData_ , sizeof(int) * dims_[0] * dims_[1] * dims_[2] );
}

gpuIntArray::~gpuIntArray( )
{

  cudaFree( deviceData_ );

}

void gpuIntArray::sendDataToGPU( int* hostMemory )
{
  
  cudaMemcpy( deviceData_ , hostMemory , sizeof(int) * dims_[0] * dims_[1] * dims_[2] , cudaMemcpyHostToDevice );
  
  return;
}

void gpuIntArray::getDataFromGPU( int* hostMemory )
{

  cudaMemcpy( hostMemory , deviceData_ , sizeof(int) * dims_[0] * dims_[1] * dims_[2] , cudaMemcpyDeviceToHost );
  
  return;
}

int* gpuIntArray::getDevicePtr( )
{
  return deviceData_;
}
    
int gpuIntArray::getDim( int dimension )
{
  return dims_[dimension];
}

/*******************************************************************************************************

ROUTINES FOR CONE BEAM GEOMETRY CLASS.

 class coneBeamGeometry {
 
 public:
 
 coneBeamGeometry(int volD[3], int projD[3],  int startV[3], int endV[3], int projDataSize[2], int* projDataWStart, int* projDataHStart, float* angles, float* hMid, float sourceSampleDist);
 coneBeamGeometry(int volD[3], int projD[3],  int startV[3], int endV[3], int projDataSize[2], int* projDataWStart, int* projDataHStart, float* angles, float* hMid, float sourceSampleDist, float* misAlignments);
 ~coneBeamGeometry( );
 
 //Problem-level data descriptors
 int volDim[3];
 int volStart[3];
 int volEnd[3];
 int projDim[3];
 int projWHSize[2];
 int* projWOffset;
 int* projHOffset;
 
 //Problem-level geometry variables
 float* sourcePosition_xyz; //This array gives the source position in coordinates for each projection, [x1,y1,z1,x2,y2,z2,etc]
 float* detectorCentre_xyz; //This array gives the detector position in coordinates for each projection, [x1,y1,z1,x2,y2,z2,etc]
 float* hStep_xyz; //This array gives a single-pixel step in the h direction for each projection, [x1,y1,z1,x2,y2,z2,etc]
 float* wStep_xyz; //This array gives a single-pixel step in the w direction for each projection, [x1,y1,z1,x2,y2,z2,etc]
 
 //Problem-level geometry variables for backprojection
 float* hMidPlane;
 float* sina;
 float* cosa;
 float sourceSampleDistance;
 
 //Sub-problem-level data descriptors
 int subVolDim[3];
 int subProjDim[3];
 int numSubProblems;
 int* xxStart; //All start arrays are array-indices, not coordinates, as measured from the (0,0,0) of the CT problem. They may be -ve; the getData routines must tolerate this.
 int* yyStart;
 int* zzStart;
 int** wwStart;
 int** hhStart;
 int* ttStart;
 
 //Sub-problem geometry
 int** sweepStart; //The number of single-pixel steps that need to be taken along a ray-path, before beginning to sweep out the entire volume. Indexed [subproblem][projection-in-that-subproblem].
 int** sweepNumSteps; //The number of single-pixel steps that need to be taken during projection, to sweep out the entire volume.

 private:
 
 //Back-projection geometry variables
 
 
 void init(int volD[3], int projD[3],  int startV[3], int endV[3], int projDataSize[2], int* projDataWStart, int* projDataHStart, float* angles, float* hMid, float sourceSampleDist, float* misAlignments);
 void freeSubProblem();
 void setupSubProblem( int newSubVolDim[3] , int numProj );
 void freeProblemGeometry();
 void calculateProblemGeometry(float* angles, float* hMid, float sourceSampleDist, float* misAlignments);
 void freeSweep();
 void calculateSweep();
 
 
 void projectPoint( float xx , float yy , float zz , int tt , float* ww , float* hh );
 void projectCube(float xxBeg, float xxEnd, float yyBeg, float yyEnd, float zzBeg, float zzEnd, int tt, int* wwEdge, int* hhEdge);
 };

*******************************************************************************************************/

coneBeamGeometry::coneBeamGeometry(int volD[3], int projD[3], int startV[3], int endV[3], int projDataSize[2], int* projDataWStart, int* projDataHStart, float* angles, float* hMid, float sourceSampleDist, float pixelW)
{
  float* misAlignments;
  misAlignments = new float [6];
  misAlignments[0] = 0.0; //Dw
  misAlignments[1] = 0.0; //Dh
  misAlignments[2] = 0.0; //Dl
  misAlignments[3] = 0.0; //Dphi
  misAlignments[4] = 0.0; //Dtheta
  misAlignments[5] = 0.0; //Dpsi
  _init(volD, projD, startV, endV, projDataSize, projDataWStart, projDataHStart, angles, hMid, sourceSampleDist, misAlignments, pixelW);
  delete [] misAlignments;
}

coneBeamGeometry::coneBeamGeometry(int volD[3], int projD[3], int startV[3], int endV[3], int projDataSize[2], int* projDataWStart, int* projDataHStart, float* angles, float* hMid, float sourceSampleDist, float* misAlignments, float pixelW)
{
  _init(volD, projD, startV, endV, projDataSize, projDataWStart, projDataHStart, angles, hMid, sourceSampleDist, misAlignments, pixelW);
}

void coneBeamGeometry::_init(int volD[3], int projD[3], int startV[3], int endV[3], int projDataSize[2], int* projDataWStart, int* projDataHStart, float* angles, float* hMid, float sourceSampleDist, float* misAlignments, float pixelW)
{
  
  //Note that hMid is measuring from the bottom of the array. 
  //It is converted to hMidPlane, which is measuring from the zero coordinate (array index volD[2] / 2). 
  //Allocate some local variables
  int maxSubVolDim[3];
  int maxProjPerSubProblem;
  //Initialize geometry parameters  
  hMidPlane = NULL; 
  sina = NULL;
  cosa = NULL;
  sourcePosition_xyz = NULL;
  detectorCentre_xyz = NULL;
  hStep_xyz = NULL;
  wStep_xyz = NULL;
  //Initialize subProblem dependent parameters
  xxStart = NULL;
  yyStart = NULL;
  zzStart = NULL;
  wwStart = NULL;
  hhStart = NULL;
  ttStart = NULL;
  sweepStart = NULL;
  sweepNumSteps = NULL;
  projWOffset = NULL;
  projHOffset = NULL;
  //Set the dimension variables from the arguments
  _calculateDimensions(volD, projD, startV, endV, projDataSize, projDataWStart, projDataHStart);
  //Calculate the geometry parameters
  _calculateProblemGeometry(angles, hMid, sourceSampleDist, misAlignments, pixelW);
  //These are hard-coded. No real way around it - it's just about what dimensions work best
  // on a given device. Eventually I want this to be dynamically determined. There's also the 5
  // second device time-out to consider. If any of these numbers are >2048, CUDA will !@#$ itself.
  maxSubVolDim[0] = 1024;
  maxSubVolDim[1] = 1024;
  maxSubVolDim[2] = 512;
  maxProjPerSubProblem = 256;
  _setupSubProblem(maxSubVolDim, maxProjPerSubProblem);
  
  //printf( "subProjDims: %d , %d , %d. subVolDims: %d , %d , %d.\n" , subProjDim[0] , subProjDim[1] , subProjDim[2] , subVolDim[0] , subVolDim[1] , subVolDim[2] );
  //printf( "projDims: %d , %d , %d. volDims: %d , %d , %d.\n" , projDim[0] , projDim[1] , projDim[2] , volDim[0] , volDim[1] , volDim[2] );
  //fflush(stdout);
  
  //Allocate sweep variables
  _calculateSweep();
  return;
}

coneBeamGeometry::~coneBeamGeometry()
{
  //Delete various arrays, if they exist.
  _freeSweep();
  _freeSubProblem();
  _freeProblemGeometry();
  _freeDimensions();
}

void coneBeamGeometry::_calculateDimensions(int volD[3], int projD[3], int startV[3], int endV[3], int projDataSize[2], int* projDataWStart, int* projDataHStart)
{
  int tt, ii;
  //Allocate parameters set in this function:
  _freeDimensions();
  projWOffset = new int[projD[2]];
  projHOffset = new int[projD[2]];
  //Copy values from arguments into structure-level variables.
  for( tt = 0 ; tt < projD[2] ; tt++ ){
    projWOffset[tt] = projDataWStart[tt];
    projHOffset[tt] = projDataHStart[tt];
  }
  for(ii=0; ii<3; ii++){
    volDim[ii] = volD[ii];
    volStart[ii] = startV[ii];
    volEnd[ii] = endV[ii];
    projDim[ii] = projD[ii];
  }
  projWHSize[0] = projDataSize[0];
  projWHSize[1] = projDataSize[1];
  return;
}

void coneBeamGeometry::_freeDimensions()
{
  if (projWOffset){delete [] projWOffset;}
  if (projHOffset){delete [] projHOffset;}
  projWOffset = NULL;
  projHOffset = NULL;
  return;
}

void coneBeamGeometry::_freeSubProblem()
{
  int pp;
  if( xxStart ){ delete [] xxStart; }
  if( yyStart ){ delete [] yyStart; }
  if( zzStart ){ delete [] zzStart; }
  if( wwStart ){
    for( pp = 0 ; pp < numSubProblems ; pp++ ){
      if( wwStart[pp] ) { delete [] wwStart[pp]; }
    }
    delete [] wwStart;
  }
  if( hhStart ){
    for( pp = 0 ; pp < numSubProblems ; pp++ ){
      if( hhStart[pp] ) { delete [] hhStart[pp]; }
    }
    delete [] hhStart;
  }
  if( ttStart ){ delete [] ttStart; }
  //Re-initialise structure-level variables.
  xxStart = NULL;
  yyStart = NULL;
  zzStart = NULL;
  wwStart = NULL;
  hhStart = NULL;
  ttStart = NULL;
  
  numSubProblems = 0;
  
  subProjDim[0] = 0;
  subProjDim[1] = 0;
  subProjDim[2] = 0;
  
  subVolDim[0] = 0;
  subVolDim[1] = 0;
  subVolDim[2] = 0;
  
  return;
}

void coneBeamGeometry::_setupSubProblem(int maxSubVolDim[3], int maxProjPerSubProblem)
{
  int tt , pp, ii , jj , kk , index;
  int numSubProb_x, numSubProb_y, numSubProb_z, numSubProb_p;
  float xx , yy , zz;
  int wwEdge[2];
  int hhEdge[2];
  //Free old memory (if allocated). Needs to be done before we forget how many sub-problems there used to be.
  _freeSubProblem();
  //Shrink the maximum sub-volume dimensions to ensure we're not wasting time.
  for( tt = 0 ; tt < 3 ; tt++ ){
    subVolDim[tt] = maxSubVolDim[tt];
    if( subVolDim[tt] > ( volEnd[tt] - volStart[tt] ) ){
      subVolDim[tt] = volEnd[tt] - volStart[tt];
    }
  }
  //Increase subVolDims to next multiple of appropriate BLOCK_SIZE
  subVolDim[0] = ((subVolDim[0] + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X) * BLOCK_SIZE_X;
  subVolDim[1] = ((subVolDim[1] + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y) * BLOCK_SIZE_Y;
  subVolDim[2] = ((subVolDim[2] + 4*BLOCK_SIZE_Z - 1) / (4*BLOCK_SIZE_Z)) * 4*BLOCK_SIZE_Z;
  //Set up the sub-projection dimensions
  subProjDim[0] = 1; //These numbers will be inflated to whatever is needed later on.
  subProjDim[1] = 1; //These numbers will be inflated to whatever is needed later on.
  numSubProb_p = (projDim[2] + maxProjPerSubProblem - 1) / maxProjPerSubProblem;
  subProjDim[2] = (projDim[2] + numSubProb_p - 1) / numSubProb_p;
  //Calculate number of subProblems; ensure complete coverage of volume.
  numSubProb_x = (volEnd[0] - volStart[0] + subVolDim[0] - 1) / subVolDim[0];
  numSubProb_y = (volEnd[1] - volStart[1] + subVolDim[1] - 1) / subVolDim[1];
  numSubProb_z = (volEnd[2] - volStart[2] + subVolDim[2] - 1) / subVolDim[2];
  numSubProb_p = (projDim[2] + subProjDim[2] - 1) / subProjDim[2];  
  numSubProblems = numSubProb_x * numSubProb_y * numSubProb_z * numSubProb_p;
  _padArrays(numSubProb_p * subProjDim[2]);
  //printf("CBG: %d, %d, %d, %d, %d\n", numSubProblems, numSubProb_x, numSubProb_y, numSubProb_z, numSubProb_p);
  //fflush(stdout);
  //Allocate new memory for Start lists
  xxStart = new int [ numSubProblems ];
  yyStart = new int [ numSubProblems ];
  zzStart = new int [ numSubProblems ];
  wwStart = new int* [ numSubProblems ];
  hhStart = new int* [ numSubProblems ];
  for( pp = 0 ; pp < numSubProblems ; pp++ ){
    wwStart[pp] = new int [subProjDim[2]];
    hhStart[pp] = new int [subProjDim[2]];
  }
  ttStart = new int [ numSubProblems ];
  //Calculate the x, y, and z starting indices.
  index = 0;
  for( kk = 0 ; kk < numSubProb_z ; kk++ ){
    for( jj = 0 ; jj < numSubProb_y ; jj++ ){
      for( ii = 0 ; ii < numSubProb_x ; ii++ ){
        for( pp = 0 ; pp < numSubProb_p ; pp++ ){
          xxStart[index] = ii * subVolDim[0] + volStart[0];
          yyStart[index] = jj * subVolDim[1] + volStart[1];
          zzStart[index] = kk * subVolDim[2] + volStart[2];
          ttStart[index] = pp * subProjDim[2];
          index++;
        }
      }
    }
  }
  //Calculate the ww and hh starting indices, and the subProblem ww and hh sizes.
  for(pp=0; pp<numSubProblems; pp++){
    //Get volume start positions for this sub-Problem
    xx = xxStart[pp];
    yy = yyStart[pp];
    zz = zzStart[pp];
    //cycle through all projections of this sub-problem, getting the maximum and minimum hh as we go.
    for( tt = 0 ; tt < subProjDim[2] ; tt++ ){
      _projectCube(xx, xx+subVolDim[0], yy, yy+subVolDim[1], zz, zz+subVolDim[2], tt+ttStart[pp], wwEdge, hhEdge);
      //Constrain the start and end positions of the sub-projection to within 2 elements of the actual projection data for this projection.
      wwEdge[0] = max(wwEdge[0], projWOffset[tt+ttStart[pp]]-4);
      wwEdge[1] = max(wwEdge[1], projWOffset[tt+ttStart[pp]]-2);
      hhEdge[0] = max(hhEdge[0], projHOffset[tt+ttStart[pp]]-4);
      hhEdge[1] = max(hhEdge[1], projHOffset[tt+ttStart[pp]]-2);
      //Now upper constraint
      wwEdge[0] = min(wwEdge[0], projWOffset[tt+ttStart[pp]]+projWHSize[0]+4);
      wwEdge[1] = min(wwEdge[1], projWOffset[tt+ttStart[pp]]+projWHSize[0]+2);
      hhEdge[0] = min(hhEdge[0], projHOffset[tt+ttStart[pp]]+projWHSize[1]+4);
      hhEdge[1] = min(hhEdge[1], projHOffset[tt+ttStart[pp]]+projWHSize[1]+2);
      //Save the starting positions
      wwStart[pp][tt] = wwEdge[0];
      hhStart[pp][tt] = hhEdge[0];
      //Increase the subProjDims if necessary
      subProjDim[0] = max(subProjDim[0], wwEdge[1]-wwEdge[0]);
      subProjDim[1] = max(subProjDim[1], hhEdge[1]-hhEdge[0]);
    }
  }
  //Inflate sub-problem dimensions to the next highest multiple of BLOCK_SIZE_W, and BLOCK_SIZE_H
  subProjDim[0] = BLOCK_SIZE_W * (1+(subProjDim[0] / BLOCK_SIZE_W));
  subProjDim[1] = BLOCK_SIZE_H * (1+(subProjDim[1] / BLOCK_SIZE_H));
  //To make this increase symmetric in all dimensions, shift the subProj regions half a block.
  for( pp = 0 ; pp < numSubProblems ; pp++ ){
    for( tt = 0 ; tt < subProjDim[2] ; tt++ ){
      wwStart[pp][tt] -= (BLOCK_SIZE_W / 2);
      hhStart[pp][tt] -= (BLOCK_SIZE_H / 2);
    }
  }
        //printf("x,y,z,w,h: %d, %d, %d, %d, %d\n", xxStart[0], yyStart[0], zzStart[0], wwStart[0][0], hhStart[0][0]);
  return;
  
}

void coneBeamGeometry::_padArrays(int newSize)
{
  //Declare placeholders
  _padOneIntArray(&projWOffset, newSize, projDim[2]);
  _padOneIntArray(&projHOffset, newSize, projDim[2]);
  _padOneFltArray(&hMidPlane, newSize, projDim[2]);
  _padOneFltArray(&sina, newSize, projDim[2]);
  _padOneFltArray(&cosa, newSize, projDim[2]);
  _padOneFltArray(&sourcePosition_xyz, 3 * newSize, 3 * projDim[2]);
  _padOneFltArray(&detectorCentre_xyz, 3 * newSize, 3 * projDim[2]);
  _padOneFltArray(&hStep_xyz, 3 * newSize, 3 * projDim[2]);
  _padOneFltArray(&wStep_xyz, 3 * newSize, 3 * projDim[2]);
  return;
}

void coneBeamGeometry::_padOneIntArray(int** array, int newSize, int oldSize)
{
  int* oldArray;
  int ii;
  oldArray = *array;
  if(oldArray){
    *array = new int [newSize];
    memset(*array, 0, newSize * sizeof(int));
    for(ii=0; ii<oldSize; ii++){
      (*array)[ii] = oldArray[ii];
    }
    delete [] oldArray;
    oldArray = NULL;
  }
  return;
}

void coneBeamGeometry::_padOneFltArray(float** array, int newSize, int oldSize)
{
  float* oldArray;
  int ii;
  oldArray = *array;
  if(oldArray){
    *array = new float [newSize];
    memset(*array, 0, newSize * sizeof(float));
    for(ii=0; ii<oldSize; ii++){
      (*array)[ii] = oldArray[ii];
    }
    delete [] oldArray;
    oldArray = NULL;
  }
  return;
}

void coneBeamGeometry::_freeProblemGeometry()
{
  if (hMidPlane) {delete [] hMidPlane; hMidPlane = NULL;}
  if (sina) {delete [] sina; sina = NULL;}
  if (cosa) {delete [] cosa; cosa = NULL;}
  if (sourcePosition_xyz) {delete [] sourcePosition_xyz; sourcePosition_xyz = NULL;}
  if (detectorCentre_xyz) {delete [] detectorCentre_xyz; detectorCentre_xyz = NULL;}
  if (hStep_xyz) {delete [] hStep_xyz; hStep_xyz = NULL;}
  if (wStep_xyz) {delete [] wStep_xyz; wStep_xyz = NULL;}
  return;
}

void coneBeamGeometry::_calculateProblemGeometry(float* angles, float* hMid, float sourceSampleDist, float* misAlignments, float pixelW)
{
  //Declarations
  int tt;
  float3 sourcePos, detectorPos, hStep, wStep, whPerp, wnPerp, hnPerp;
  //Allocate memory (after de-allocating if necessary)
  _freeProblemGeometry();
  hMidPlane = new float [projDim[2]];
  sina = new float [projDim[2]];
  cosa = new float [projDim[2]];
  sourcePosition_xyz = new float [3*projDim[2]];
  detectorCentre_xyz = new float [3*projDim[2]];
  hStep_xyz = new float [3*projDim[2]];
  wStep_xyz = new float [3*projDim[2]];
  //Fill the primitive geometry variables
  sourceSampleDistance = sourceSampleDist;
  pixelWidth = pixelW;
  for(tt = 0; tt < projDim[2]; tt++){
    hMidPlane[tt] = hMid[tt] - volDim[2] / 2;
    sina[tt] = sin(angles[tt]);
    cosa[tt] = cos(angles[tt]);
  }
  //Fill the sophisticated geometry variables
  for(tt = 0; tt < projDim[2]; tt++){
    //Create vectors for an aligned system
    sourcePos = make_float3(-1.0f * cos(angles[tt]) * sourceSampleDist, -1.0f * sin(angles[tt]) * sourceSampleDist, hMid[tt] - volDim[2] / 2);
    detectorPos = make_float3(0.0, 0.0, hMid[tt] - volDim[2] / 2); //The detector centre is now in the same plane as the source (before misalignments)
    hStep = make_float3(0.0, 0.0, pixelWidth);
    wStep = make_float3(-1.0f * pixelWidth * sin(angles[tt]), pixelWidth * cos(angles[tt]), 0.0);
    //Apply misalignment Dw
    detectorPos += misAlignments[0] * wStep;
    //Apply misalignment Dh
    detectorPos += misAlignments[1] * hStep;
    //Apply misalignment Dl
    detectorPos += misAlignments[2] * normalize(detectorPos - sourcePos);
    //Apply misalignment Dphi, a rotation about the w axis.
    whPerp = cross(wStep, hStep);
    hStep = length(hStep) * (cos(misAlignments[3]) * normalize(hStep) + sin(misAlignments[3]) * normalize(whPerp));
    //Apply misalignment Dtheta, a rotation about the h axis.
    whPerp = cross(wStep, hStep);
    wStep = length(wStep) * (cos(misAlignments[4]) * normalize(wStep) + sin(misAlignments[4]) * normalize(whPerp));
    //Apply misalignment Dpsi, a rotation in the w-h plane.
    whPerp = cross(wStep, hStep);
    wnPerp = cross(wStep, whPerp);
    hnPerp = cross(hStep, whPerp);
    wStep = length(wStep) * (cos(misAlignments[5]) * normalize(wStep) - sin(misAlignments[5]) * normalize(wnPerp));
    hStep = length(hStep) * (cos(misAlignments[5]) * normalize(hStep) + sin(misAlignments[5]) * normalize(hnPerp));
    //Store calculated values into arrays.
    sourcePosition_xyz[3 * tt    ] = sourcePos.x;
    sourcePosition_xyz[3 * tt + 1] = sourcePos.y;
    sourcePosition_xyz[3 * tt + 2] = sourcePos.z;
    detectorCentre_xyz[3 * tt    ] = detectorPos.x;
    detectorCentre_xyz[3 * tt + 1] = detectorPos.y;
    detectorCentre_xyz[3 * tt + 2] = detectorPos.z;
    hStep_xyz[3 * tt    ] = hStep.x;
    hStep_xyz[3 * tt + 1] = hStep.y;
    hStep_xyz[3 * tt + 2] = hStep.z;
    wStep_xyz[3 * tt    ] = wStep.x;
    wStep_xyz[3 * tt + 1] = wStep.y;
    wStep_xyz[3 * tt + 2] = wStep.z;
  }
  return;
}

void coneBeamGeometry::_freeSweep()
{
  int pp;
  //Free sweep variables if necessary
  if(sweepStart){
    for(pp = 0; pp < numSubProblems; pp++){
      if(sweepStart[pp]){delete [] sweepStart[pp];}
    }
    delete [] sweepStart;
    sweepStart = NULL;
  }
  if(sweepNumSteps){
    for(pp = 0; pp < numSubProblems; pp++){
      if(sweepNumSteps[pp]){delete [] sweepNumSteps[pp];}
    }
    delete [] sweepNumSteps;
    sweepNumSteps = NULL;
  }
  return;
}

void coneBeamGeometry::_calculateSweep()
{
  int tt, ii;
  double radiusSubVol;
  int globalProjIndex;
  float3 sourcePosition, subVolCentre;
  //Free sweep variables if necessary
  _freeSweep();
  //Allocate sweep variables
  sweepStart = new int* [numSubProblems];
  sweepNumSteps = new int* [numSubProblems];
  for( tt = 0 ; tt < numSubProblems ; tt++ ){
    sweepStart[tt] = new int [subProjDim[2]];
    sweepNumSteps[tt] = new int [subProjDim[2]];
  }
  //Begin by calculating the radius of a ball that completely contains each subVolume. A 10% fudge-factor is included here, to make sure we get everything.
  radiusSubVol = 0.55 * sqrt(subVolDim[0] * subVolDim[0] + subVolDim[1] * subVolDim[1] + subVolDim[2] * subVolDim[2]);
  for(tt=0; tt<numSubProblems; tt++){
    //Find the centre of this subVolume in coordinates.
    subVolCentre = make_float3(xxStart[tt] + 0.5 * subVolDim[0] - 0.5 * volDim[0], yyStart[tt] + 0.5 * subVolDim[1] - 0.5 * volDim[1], zzStart[tt] + 0.5 * subVolDim[2] - 0.5 * volDim[2]);
    for(ii=0; ii<subProjDim[2]; ii++){
      //For each projection, get the coordinates of the source.
      globalProjIndex = ii + ttStart[tt];
      sourcePosition = make_float3(sourcePosition_xyz[3*globalProjIndex], sourcePosition_xyz[3*globalProjIndex + 1], sourcePosition_xyz[3*globalProjIndex + 2]);
      sweepStart[tt][ii] = (int) (length(subVolCentre - sourcePosition) - radiusSubVol);
      sweepNumSteps[tt][ii] = (int) (2.0 * radiusSubVol);
    }
  }
  return;
}

void coneBeamGeometry::_projectCube(float xxBeg, float xxEnd, float yyBeg, float yyEnd, float zzBeg, float zzEnd, int tt, int* wwEdge, int* hhEdge)
{
  //projectCube(xx, xx+subVolDim[0], yy, yy+subVolDim[1], zz, zz+subVolDim[2], tt+ttStart[pp], wwEdge, hhEdge);
  float wwCorners[8];
  float hhCorners[8];
  int ii;
  //Figure out where the corners of this sub-problem project to.
  _projectPoint(xxBeg, yyBeg, zzBeg, tt, wwCorners, hhCorners);
  _projectPoint(xxEnd, yyBeg, zzBeg, tt, wwCorners + 1 , hhCorners + 1 );
  _projectPoint(xxBeg, yyEnd, zzBeg, tt, wwCorners + 2 , hhCorners + 2 );
  _projectPoint(xxEnd, yyEnd, zzBeg, tt, wwCorners + 3 , hhCorners + 3 );
  _projectPoint(xxBeg, yyBeg, zzEnd, tt, wwCorners + 4 , hhCorners + 4 );
  _projectPoint(xxEnd, yyBeg, zzEnd, tt, wwCorners + 5 , hhCorners + 5 );
  _projectPoint(xxBeg, yyEnd, zzEnd, tt, wwCorners + 6 , hhCorners + 6 );
  _projectPoint(xxEnd, yyEnd, zzEnd, tt, wwCorners + 7 , hhCorners + 7 );
  //Cycle through all the corners, and get the extremes in ww and hh.
  wwEdge[0] = (int)wwCorners[0];
  wwEdge[1] = (int)wwCorners[0]+1;
  hhEdge[0] = (int)hhCorners[0];
  hhEdge[1] = (int)hhCorners[0]+1;
  for( ii = 1 ; ii < 8 ; ii++ ){
    wwEdge[0] = min(wwEdge[0], (int)wwCorners[ii]);
    hhEdge[0] = min(hhEdge[0], (int)hhCorners[ii]);
    wwEdge[1] = max(wwEdge[1], (int)wwCorners[ii]+1);
    hhEdge[1] = max(hhEdge[1], (int)hhCorners[ii]+1);
  }
}

void coneBeamGeometry::_projectPoint( float xc , float yc , float zc , int tt , float* wc , float* hc )
{
#warning Vector math is done with floating point numbers, so if the source distance is too large it will cause problems.
  float wDotW, hDotH, wDotH, iDotW, iDotH;
  float distanceFromSource;
  float3 objectPos, sourcePos, detectorPos, hStep, wStep, whPerp, intersection;
  //Create 3-vectors for geometry calculations. Convert xc, yc, and zc to coordinates in the process.
  objectPos = make_float3(xc - (volDim[0] / 2), yc - (volDim[1] / 2), zc - (volDim[2] / 2));
  sourcePos = make_float3(sourcePosition_xyz[3*tt], sourcePosition_xyz[3*tt + 1], sourcePosition_xyz[3*tt + 2]);
  detectorPos = make_float3(detectorCentre_xyz[3*tt], detectorCentre_xyz[3*tt + 1], detectorCentre_xyz[3*tt + 2]);
  hStep = make_float3(hStep_xyz[3*tt], hStep_xyz[3*tt + 1], hStep_xyz[3*tt + 2]);
  wStep = make_float3(wStep_xyz[3*tt], wStep_xyz[3*tt + 1], wStep_xyz[3*tt + 2]);
  //Calculate normal unit vector to the detector plane.
  whPerp = normalize(cross(wStep, hStep));
  //Calculate distance of plane-ray intersection from the source.
  distanceFromSource = dot(detectorPos - sourcePos, whPerp) / dot(normalize(objectPos - sourcePos), whPerp);
  //Calculate the point of intersection, as an offset from the centre of the detector.
  intersection = sourcePos + distanceFromSource * normalize(objectPos - sourcePos) - detectorPos;
  //Take some dot products to make the next step readable, and then calculate wc and hc. We ackowledge that wStep and hStep need not be orthogonal, or normalized.
  wDotW = dot(wStep, wStep);
  wDotH = dot(wStep, hStep);
  hDotH = dot(hStep, hStep);
  iDotW = dot(intersection, wStep);
  iDotH = dot(intersection, hStep);
  *wc = (iDotW * hDotH - iDotH * wDotH) / (wDotW * hDotH - wDotH * wDotH);
  *hc = (iDotH * wDotW - iDotW * wDotH) / (wDotW * hDotH - wDotH * wDotH);  
  //printf( "NEW CODE, _projectPoint: xx= %f , yy= %f, zz= %f; wc= %f, hc=%f; sourcePos= (%f, %f, %f); detectorPos= (%f, %f, %f); hStep= (%f, %f, %f), wStep= (%f, %f, %f)\n" , objectPos.x, objectPos.y, objectPos.z, *wc, *hc, sourcePos.x, sourcePos.y, sourcePos.z, detectorPos.x, detectorPos.y, detectorPos.z, hStep.x, hStep.y, hStep.z, wStep.x, wStep.y, wStep.z);
  //fflush(stdout);
  //Convert wc and hc to array indicies.
  *wc += ( projDim[0] / 2 );
  *hc += ( projDim[1] / 2 );
  return;
  
}

