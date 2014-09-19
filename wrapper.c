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
//#include <fftw3.h>
#include <string.h>
#include <pthread.h>

#include "unit_coneBeamGPU.h"

/****************************************************************
 
//File-scope variables
 
****************************************************************/
 
static float* GLOBAL_vol;
static int GLOBAL_volDim[3];
static int GLOBAL_volStart[3];
static int GLOBAL_volEnd[3];

static float* GLOBAL_proj;
static int GLOBAL_projDim[3];
static int GLOBAL_projWHSize[2];
static int* GLOBAL_projWOffset;
static int* GLOBAL_projHOffset;

static float* GLOBAL_normalizationFactor;

void forceGoodCylinder( float* volume ,
					   int Nx , int Ny , int Nz , int Nt ,
					   float* hMidPlane ,
					   float sourceSampleDistance ,
					   int Nw , int Nh );

/****************************************************************
 
Local functions
 
 ****************************************************************/

void _setGlobalVariables(float* volume,
                         float* projections,
                         int volDim[3],
                         int volStart[3],
                         int volEnd[3],
                         int projDim[3],
                         int projWHSize[2],
                         int* projWOffset,
                         int* projHOffset,
                         float* normalizationFactor
){
  //Sets the appropriate global variables.
  int ii;
	//Populate local and file-scope variables with input arguments
	for( ii=0; ii<3; ii++){
	  GLOBAL_volDim[ii] = volDim[ii];
	  GLOBAL_volStart[ii] = volStart[ii];
	  GLOBAL_volEnd[ii] = volEnd[ii];
	  GLOBAL_projDim[ii] = projDim[ii];
	}
	GLOBAL_projWHSize[0] = projWHSize[0];
	GLOBAL_projWHSize[1] = projWHSize[1];
	GLOBAL_vol = volume;
	GLOBAL_proj = projections;
	GLOBAL_projWOffset = projWOffset;
	GLOBAL_projHOffset = projHOffset;
	GLOBAL_normalizationFactor = normalizationFactor;
  return;
}

/****************************************************************


****************************************************************/


//New backprojecton function, with kernelMode options. kernelMode 1 = Katsevich, 2 = Feldkamp
void GPU_CB_K_backproject(int GPUID,
                          float* volume,
                          float* projections,
                          float* angles,
                          float* hMidPlane,
                          float sourceSampleDistance,
                          float pixelWidth,
                          int volDim[3],
                          int volStart[3],
                          int volEnd[3],
                          int projDim[3],
                          int projWHSize[2],
                          int* projWOffset,
                          int* projHOffset,
                          float* normalizationFactor,
                          int kernelMode )// 0 for unweighted, 1 for Katsevich weighting, 2 for Feldkamp weighting. See coneBeamGPU.h definitions.
{
  //Note: in the python code, all the dimensions are specified in order [z,y,x]. These arguments are in the opposite order: [x,y,z].
  printf("Backprojecting with volDim = %d,%d,%d, projDim = %d,%d,%d, volStart = %d,%d,%d, volEnd = %d,%d,%d, projWHSize = %d,%d, kernelMode = %d \n",volDim[0],volDim[1],volDim[2],projDim[0],projDim[1],projDim[2],volStart[0],volStart[1],volStart[2],volEnd[0],volEnd[1],volEnd[2],projWHSize[0],projWHSize[1], kernelMode);
  fflush(stdout);
  //Call cone-beam backprojection routine
  setDeviceNumber( GPUID );
  if(kernelMode == 0){
    float* misAlignments;
    int ii;
    misAlignments = (float*) calloc(6, sizeof(float));
    for(ii=0; ii<6; ii++){ misAlignments[ii] = 0.0f; }
    coneBeamRayTraceBackproject(volume, projections, volDim, volStart, volEnd, projDim, projWHSize, projWOffset, projHOffset, angles, hMidPlane, sourceSampleDistance, normalizationFactor, misAlignments, pixelWidth);
    free(misAlignments);
  } else {
    coneBeamBackproject(volume, projections, volDim, volStart, volEnd, projDim, projWHSize, projWOffset, projHOffset, angles, hMidPlane, sourceSampleDistance, normalizationFactor, pixelWidth, kernelMode);
  }
  cudaCleanup();
  return;
}

//Backprojection function for misaligned detector. Will run the calculation in kernel mode 0.
void GPU_CB_misalign_backproject(int GPUID,
                                 float* volume,
                                 float* projections,
                                 float* angles,
                                 float* hMidPlane,
                                 float sourceSampleDistance,
                                 float pixelWidth,
                                 int volDim[3],
                                 int volStart[3],
                                 int volEnd[3],
                                 int projDim[3],
                                 int projWHSize[2],
                                 int* projWOffset,
                                 int* projHOffset,
                                 float* normalizationFactor,
                                 float* misAlignments)// 0 for unweighted, 1 for Katsevich weighting, 2 for Feldkamp weighting. See coneBeamGPU.h definitions.
{
  //Note: in the python code, all the dimensions are specified in order [z,y,x]. These arguments are in the opposite order: [x,y,z].
  printf("Backprojecting with projDim = %d,%d,%d, volStart = %d,%d,%d, volEnd = %d,%d,%d, projWHSize = %d,%d\n",projDim[0],projDim[1],projDim[2],volStart[0],volStart[1],volStart[2],volEnd[0],volEnd[1],volEnd[2],projWHSize[0],projWHSize[1]);
  fflush(stdout);
  //Call cone-beam backprojection routine
  setDeviceNumber( GPUID );
  coneBeamRayTraceBackproject(volume, projections, volDim, volStart, volEnd, projDim, projWHSize, projWOffset, projHOffset, angles, hMidPlane, sourceSampleDistance, normalizationFactor, misAlignments, pixelWidth);
  cudaCleanup();
  return;
}

//Deprecated backprojection function without kernelMode option. Included for backwards compatibility.
void GPU_CB_backproject(
  int GPUID,
  float* volume,
  float* projections,
  float* angles,
  float* hMidPlane,
  float sourceSampleDistance,
  float pixelWidth,
  int volDim[3], int volStart[3], int volEnd[3],
  int projDim[3], int projWHSize[2],
  int* projWOffset,
  int* projHOffset,
  float* normalizationFactor)
{
  //Call BP function with Feldkamp (default) filtering.
  GPU_CB_K_backproject(GPUID, volume, projections, angles, hMidPlane, sourceSampleDistance, pixelWidth, volDim, volStart, volEnd, projDim, projWHSize, projWOffset, projHOffset, normalizationFactor, 2);
  return;
}


/****************************************************************
 
 //Callback functions for M_GPU_CB_project
 
 ****************************************************************/

void GPU_CB_misalign_project(int GPUID,
					float* volume, 
					float* projections,
					float* angles,
					float* hMidPlane,
          float sourceSampleDistance,
          float pixelWidth,
					int volDim[3], int volStart[3], int volEnd[3],
					int projDim[3], int projWHSize[2], 
					int* projWOffset,
					int* projHOffset,
					float* normalizationFactor,
					float* misAlignments)
{
  //Declatarions
  int zz;	
  //Populate local and file-scope variables with input arguments
  _setGlobalVariables(volume, projections, volDim, volStart, volEnd, projDim, projWHSize, projWOffset, projHOffset, normalizationFactor);
  //printf("projDim = %d,%d,%d, volStart = %d,%d,%d, volEnd = %d,%d,%d, projWHSize = %d,%d\n",projDim[0],projDim[1],projDim[2],volStart[0],volStart[1],volStart[2],volEnd[0],volEnd[1],volEnd[2],projWHSize[0],projWHSize[1]);
  //Call cone-beam backprojection routine
  setDeviceNumber(GPUID);
  coneBeamProject(volume, projections, volDim, volStart, volEnd, projDim, projWHSize, projWOffset, projHOffset, angles, hMidPlane, sourceSampleDistance, normalizationFactor, misAlignments, pixelWidth, &getEmptyProj, &storeProj, &getVol, &doneVol);
  cudaCleanup();	
  return;
}

void GPU_CB_project(int GPUID,
                                        float* volume,
                                        float* projections,
                                        float* angles,
                                        float* hMidPlane,
                                        float sourceSampleDistance,
                                        float pixelWidth,
                                        int volDim[3], int volStart[3], int volEnd[3],
                                        int projDim[3], int projWHSize[2],
                                        int* projWOffset,
                                        int* projHOffset,
                                        float* normalizationFactor)
{
  float* misAlignments;
  int ii;
  misAlignments = (float*) calloc(6, sizeof(float));
  for(ii=0; ii<6; ii++){ misAlignments[ii] = 0.0f; }
  GPU_CB_misalign_project(GPUID, volume, projections, angles, hMidPlane, sourceSampleDistance, pixelWidth, volDim, volStart, volEnd, projDim, projWHSize, projWOffset, projHOffset, normalizationFactor, misAlignments);
  free(misAlignments);
  return;
}