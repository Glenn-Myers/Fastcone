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

#include "unit_coneBeamGPU.h"
#include "coneBeamGPU.h"


//****************************************
//Device management functions
//****************************************

//These are just wrappers for the CUDA functions in Fast_CB_Kernel.

int getNumberOfDevices( )
{
  int ii;
  ii = getNumDevices( );
  return ii;
}

void setDeviceNumber( int deviceNumber )
{
  setDeviceNum( deviceNumber ); 
  return;
}

void cudaCleanup( )
{
  cudaThreadExit( );
  return;
}


//****************************************
//Projection/backprojection management functions
//****************************************

 
//Importantly, the device number should already be set; we don't do it here because each thread can only do it once.
 
//Compiler warning so that I don't forget to do things.

//The "problem-solver" routines. The volume to be solved is that between volStart and volEnd, and has no size restrictions.
  
//Note on the callbacks: (1) The "store" callbacks must use an "add-and-save" type operation - it's impossible to backproject
// all the projections at once due to restrictions on the size of 3D textures, and the inability to create an array of
// 2D textures. (2) The "get" data callbacks should supply "out-of-bounds" data as 0.0f. (3) The getVolume routine 
// and getProjection routines must allocate memory for the pointers they return. (4) This memory should be freed 
// in the storeVolume and doneWithProjection routines. This is because we don't know what language these routines are
// being called from (e.g. new, or malloc?)

int coneBeamProject(
  float* volume,
  float* projections,
  int   volDim[3] ,
  int   volStart[3] ,
  int   volEnd[3] ,
  int   projDim[3] ,
  int   projWHSize[2],
  int*  projWOffset,
  int*  projHOffset,
  float*  angles ,
  float*  hMidPlane , //As measured from the zeroeth array index of the total problem.
  float sourceSampleDistance,
  float* normalizationFactor,
  float*  misAlignments,
  float pixelWidth,
  void  ( *getProjection    )( float** , int[3] , int* , int* , int ) ,
  void  ( *storeProj      )( float** , int[3] , int* , int* , int ) ,
  void  ( *getVolume      )( float** , int[3] , int , int , int ) ,
  void  ( *donewithVolume   )( float** , int[3] ) )
{
  
  //Declarations
  int ii;
  int paddedSubVolDim[3];
  //printf("USING CODE IN LOCAL REPOSITORY\n");
  //fflush(stdout);
  //Build geometry class - this automatically divides the problem up into sub-problems that will fit on the graphics card.
  coneBeamGeometry geometry(volDim, projDim, volStart, volEnd, projWHSize, projWOffset, projHOffset, angles, hMidPlane, sourceSampleDistance, misAlignments, pixelWidth);
  //printf("File: %s.   Line: %d.\n, Running pixelWidth enabled code.\n", __FILE__ , __LINE__ );
  //fflush(stdout);
  //printf("P: Done building geometry %d %d %d %d\n", geometry.numSubProblems, geometry.subVolDim[0], geometry.subVolDim[1], geometry.subVolDim[2]);
  //fflush(stdout); 
  //Pad out the sub-volume dimensions with 1 element on each side (need out-of-bounds texture fetches to return zero).
  paddedSubVolDim[0] = geometry.subVolDim[0] + 2;
  paddedSubVolDim[1] = geometry.subVolDim[1] + 2;
  paddedSubVolDim[2] = geometry.subVolDim[2] + 2;
  //Allocate arrays on device.
  gpuFltArray* deviceProjPtr = new gpuFltArray(geometry.subProjDim);
  gpuTexArray* deviceVolumePtr = new gpuTexArray(paddedSubVolDim);
  gpuFltArray* sourcePosition_xyz_d = new gpuFltArray(3 * geometry.subProjDim[2]);
  gpuFltArray* detectorCentre_xyz_d = new gpuFltArray(3 * geometry.subProjDim[2]);
  gpuFltArray* hStep_xyz_d = new gpuFltArray(3 * geometry.subProjDim[2]);
  gpuFltArray* wStep_xyz_d = new gpuFltArray(3 * geometry.subProjDim[2]);
  gpuIntArray* sStart_d = new gpuIntArray(geometry.subProjDim[2]);
  gpuIntArray* sNumSteps_d = new gpuIntArray(geometry.subProjDim[2]);
  gpuIntArray* NwOffset_d = new gpuIntArray(geometry.subProjDim[2]);
  gpuIntArray* NhOffset_d = new gpuIntArray(geometry.subProjDim[2]);
  //printf("P: Done allocating tex and float\n");
  //fflush(stdout); 
  //Loop over all subproblems. This loop can be re-structured so that the host-side processing (getting and storing
  // data) all occurs whilst the GPU is number-crunching, but it's much less clear, and there's not that much of a
  // speed increase.
  //printf("P: Printing geom.numSubProblems\n");
  //fflush(stdout);
  //printf("P: %d\n", geometry.numSubProblems);
  //Pre-fill Cache for first sub-problem.
  deviceVolumePtr->fillCache(volume, geometry.volStart, geometry.volEnd, geometry.xxStart[0] - 1, geometry.yyStart[0] - 1, geometry.zzStart[0] - 1);
  deviceVolumePtr->zeroCacheBoundary();
  //Loop over all sub-problems
  for( ii = 0 ; ii < geometry.numSubProblems ; ii++ ) {
    deviceVolumePtr->sendCacheToGPU();
    //Process this subproblem on the GPU
    newProjectSubVolume(ii, &geometry, deviceVolumePtr, deviceProjPtr, sourcePosition_xyz_d, detectorCentre_xyz_d, hStep_xyz_d, wStep_xyz_d, sStart_d, sNumSteps_d, NwOffset_d, NhOffset_d, __ASYNC__);
    //If it makes sense, do some cacheing while we're waiting for the GPU.
    if((ii + 1) < geometry.numSubProblems){
      deviceVolumePtr->fillCache(volume, geometry.volStart, geometry.volEnd, geometry.xxStart[ii+1] - 1, geometry.yyStart[ii+1] - 1, geometry.zzStart[ii+1] - 1);
      deviceVolumePtr->zeroCacheBoundary();
    }
    if(ii != 0){ deviceProjPtr->addCache(projections, geometry.projWHSize, geometry.projDim[2], geometry.projWOffset, geometry.projHOffset, 0, normalizationFactor, geometry.wwStart[ii-1] , geometry.hhStart[ii-1] , geometry.ttStart[ii-1]); }
    //Store the projection data.
    deviceProjPtr->getCacheFromGPU();
  } 
  //Empty Cache from final sub-problem
  deviceProjPtr->addCache(projections, geometry.projWHSize, geometry.projDim[2], geometry.projWOffset, geometry.projHOffset, 0, normalizationFactor, geometry.wwStart[geometry.numSubProblems-1] , geometry.hhStart[geometry.numSubProblems-1] , geometry.ttStart[geometry.numSubProblems-1]);
  //No dynamically allocated memory here.
  delete deviceVolumePtr;
  delete deviceProjPtr;
  delete sourcePosition_xyz_d;
  delete detectorCentre_xyz_d;
  delete hStep_xyz_d;
  delete wStep_xyz_d;
  delete sStart_d;
  delete sNumSteps_d;
  delete NwOffset_d;
  delete NhOffset_d;
  return 1;
}

int coneBeamBackproject(
  float* volume,
  float* projections,
  int   volDim[3] ,
  int   volStart[3] ,
  int   volEnd[3] ,
  int   projDim[3] ,
  int   projWHSize[2],
  int*  projWOffset,
  int*  projHOffset,
  float*  angles ,
  float*  hMidPlane , //As measured from the zeroeth array index of the total problem.
  float sourceSampleDistance ,
  float* normalizationFactor,
  float pixelWidth,
  int kernelMode //Either __KATSEVICH__ or __FELDKAMP__
)
{
  
  //Declarations
  int ii;
  int deviceGeometryArrayDim[3] = { 1 , 1 , 1 };  
  //Build geometry class
  coneBeamGeometry geometry(volDim , projDim , volStart , volEnd, projWHSize, projWOffset, projHOffset, angles , hMidPlane , sourceSampleDistance , pixelWidth);
  //printf("File: %s.   Line: %d.\n", __FILE__ , __LINE__ );
  //fflush(stdout);   
  //Create arrays on device.
  deviceGeometryArrayDim[0] = geometry.subProjDim[2];
  gpuFltArray* deviceVolumePtr = new gpuFltArray ( geometry.subVolDim );
  gpuTexArray* deviceProjPtr = new gpuTexArray ( geometry.subProjDim );
  gpuFltArray* deviceHmidPtr = new gpuFltArray ( deviceGeometryArrayDim );
  gpuFltArray* deviceSinaPtr = new gpuFltArray ( deviceGeometryArrayDim );  
  gpuFltArray* deviceCosaPtr = new gpuFltArray ( deviceGeometryArrayDim );  
  gpuIntArray* devicewwOffsetPtr = new gpuIntArray ( deviceGeometryArrayDim );
  gpuIntArray* devicehhOffsetPtr = new gpuIntArray ( deviceGeometryArrayDim );
  //printf("File: %s.   Line: %d.\n, Running pixelWidth enabled code.\n", __FILE__ , __LINE__ );
  //fflush(stdout); 
  //At this stage, there's little gain in adopting a more sophisticated scheme, where the host operations
  //are performed whilst the Kernel is executing. This may change when we start dealing with faster cards,
  //and MPI setups.
  //Begin by filling the cache for the zeroeth iteration
  deviceProjPtr->fillCache(projections, geometry.projWHSize, geometry.projDim[2], geometry.projWOffset, geometry.projHOffset, 0, normalizationFactor, geometry.wwStart[0], geometry.hhStart[0], geometry.ttStart[0]);
  //Loop over all sub-problems
  for( ii = 0 ; ii < geometry.numSubProblems ; ii++ ) {
    printf("Sub problem %d, wwStart %d, hhStart %d, ttStart %d, xxStart %d, yyStart %d, zzStart %d.\n", ii, geometry.wwStart[ii][0] , geometry.hhStart[ii][0] , geometry.ttStart[ii], geometry.xxStart[ii] , geometry.yyStart[ii] , geometry.zzStart[ii]);
    fflush(stdout);
    //Send the data to the GPU and process it.
    deviceProjPtr->sendCacheToGPU();
    backprojectSubVolume(ii, &geometry, deviceVolumePtr, deviceProjPtr, deviceHmidPtr, deviceSinaPtr, deviceCosaPtr, devicewwOffsetPtr, devicehhOffsetPtr, __ASYNC__, kernelMode);
    //If it won't cause errors, do some cacheing whilst we wait for the GPU to finish.
    if((ii + 1) < geometry.numSubProblems){ deviceProjPtr->fillCache(projections, geometry.projWHSize, geometry.projDim[2], geometry.projWOffset, geometry.projHOffset, 0, normalizationFactor, geometry.wwStart[ii+1], geometry.hhStart[ii+1], geometry.ttStart[ii+1]); }
    if(ii != 0){ deviceVolumePtr->addCache(volume, geometry.volStart, geometry.volEnd, geometry.xxStart[ii-1], geometry.yyStart[ii-1], geometry.zzStart[ii-1]); }
    //Once the GPU is finished, get the data from the GPU
    deviceVolumePtr->getCacheFromGPU();
  }
  //Empty the cache from the previous problem.
  deviceVolumePtr->addCache(volume, geometry.volStart, geometry.volEnd, geometry.xxStart[geometry.numSubProblems-1], geometry.yyStart[geometry.numSubProblems-1], geometry.zzStart[geometry.numSubProblems-1]);
  delete deviceVolumePtr;
  delete deviceProjPtr;
  delete deviceHmidPtr;
  delete deviceSinaPtr;
  delete deviceCosaPtr;
  delete devicewwOffsetPtr;
  delete devicehhOffsetPtr; 
  return 1;
}

int coneBeamRayTraceBackproject(
  float* volume,
  float* projections,
  int volDim[3] ,
  int volStart[3],
  int volEnd[3],
  int projDim[3],
  int projWHSize[2],
  int* projWOffset,
  int* projHOffset,
  float* angles ,
  float* hMidPlane , //As measured from the zeroeth array index of the total problem.
  float sourceSampleDistance ,
  float* normalizationFactor,
  float* misAlignments,
  float pixelWidth)
{
  //Declarations
  int ii;
  //Build geometry class
  coneBeamGeometry geometry(volDim, projDim, volStart, volEnd, projWHSize, projWOffset, projHOffset, angles, hMidPlane, sourceSampleDistance, misAlignments, pixelWidth);
  //printf("USING CODE IN LOCAL REPOSITORY: RAYTRACE VERSION\n");
  //printf("File: %s.   Line: %d.\n", __FILE__ , __LINE__ );
  //fflush(stdout);
  //Create arrays on device.
  gpuTexArray* deviceProjPtr = new gpuTexArray(geometry.subProjDim);
  gpuFltArray* deviceVolumePtr = new gpuFltArray(geometry.subVolDim);
  gpuFltArray* sourcePosition_xyz_d = new gpuFltArray(3 * geometry.subProjDim[2]);
  gpuFltArray* detectorCentre_xyz_d = new gpuFltArray(3 * geometry.subProjDim[2]);
  gpuFltArray* hStep_xyz_d = new gpuFltArray(3 * geometry.subProjDim[2]);
  gpuFltArray* wStep_xyz_d = new gpuFltArray(3 * geometry.subProjDim[2]);
  gpuIntArray* NwOffset_d = new gpuIntArray(geometry.subProjDim[2]);
  gpuIntArray* NhOffset_d = new gpuIntArray(geometry.subProjDim[2]);
  //printf("File: %s.   Line: %d.\n", __FILE__ , __LINE__ );
  //fflush(stdout);
  //At this stage, there's little gain in adopting a more sophisticated scheme, where the host operations
  //are performed whilst the Kernel is executing. This may change when we start dealing with faster cards,
  //and MPI setups.
  //Begin by filling the cache for the zeroeth iteration
  deviceProjPtr->fillCache(projections, geometry.projWHSize, geometry.projDim[2], geometry.projWOffset, geometry.projHOffset, 0, normalizationFactor, geometry.wwStart[0], geometry.hhStart[0], geometry.ttStart[0]);
  //Loop over all sub-problems
  for( ii = 0 ; ii < geometry.numSubProblems ; ii++ ) {
    printf("Sub problem %d, wwStart %d, hhStart %d, ttStart %d, xxStart %d, yyStart %d, zzStart %d.\n", ii, geometry.wwStart[ii][0] , geometry.hhStart[ii][0] , geometry.ttStart[ii], geometry.xxStart[ii] , geometry.yyStart[ii] , geometry.zzStart[ii]);
    fflush(stdout);
    //Send the data to the GPU and process it.
    deviceProjPtr->sendCacheToGPU();
    raytraceBackprojectSubVolume(ii, &geometry, deviceVolumePtr, deviceProjPtr, sourcePosition_xyz_d, detectorCentre_xyz_d, hStep_xyz_d, wStep_xyz_d, NwOffset_d, NhOffset_d, __ASYNC__);
    //If it won't cause errors, do some cacheing whilst we wait for the GPU to finish.
    if((ii + 1) < geometry.numSubProblems){ deviceProjPtr->fillCache(projections, geometry.projWHSize, geometry.projDim[2], geometry.projWOffset, geometry.projHOffset, 0, normalizationFactor, geometry.wwStart[ii+1], geometry.hhStart[ii+1], geometry.ttStart[ii+1]); }
    if(ii != 0){ deviceVolumePtr->addCache(volume, geometry.volStart, geometry.volEnd, geometry.xxStart[ii-1], geometry.yyStart[ii-1], geometry.zzStart[ii-1]); }
    //Once the GPU is finished, get the data from the GPU
    deviceVolumePtr->getCacheFromGPU();
  }
  //Empty the cache from the previous problem.
  deviceVolumePtr->addCache(volume, geometry.volStart, geometry.volEnd, geometry.xxStart[geometry.numSubProblems-1], geometry.yyStart[geometry.numSubProblems-1], geometry.zzStart[geometry.numSubProblems-1]);
  //Free memory
  delete deviceVolumePtr;
  delete deviceProjPtr;
  delete sourcePosition_xyz_d;
  delete detectorCentre_xyz_d;
  delete hStep_xyz_d;
  delete wStep_xyz_d;
  delete NwOffset_d;
  delete NhOffset_d;
  return 1;
}

