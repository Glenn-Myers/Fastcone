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

#define __ASYNC__ 0
#define __SYNC__ 1

#define __KATSEVICH__ 1
#define __FELDKAMP__ 2

//Classes for arrays to be synced to the device

class gpuCachedArray {
public:
  gpuCachedArray( );
  ~gpuCachedArray( );
  void fillCache(float* source, int sourceStart[3], int sourceEnd[3], int xStart, int yStart, int zStart);
  void fillCache(float* source, int sourceWHDims[2], int sourcePDim, int* sourceWOffset, int* sourceHOffset, int sourcePOffset, float* sourceNorms, int* wStart, int* hStart, int pStart);
  void addCache(float* dest, int destStart[3], int destEnd[3], int xStart, int yStart, int zStart);
  void addCache(float* dest, int destWHDims[2], int destPDim, int* destWOffset, int* destHOffset, int sourcePOffset, float* destNorms, int* wStart, int* hStart, int pStart);
  int getDim( int dimension );
  void zeroCacheBoundary();
  
protected:
  int     _dims[3];
  float*  _hostCache;
  void _initCache( );
  void _zeroCache( );
  void _freeCache( );
};

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
  cudaArray*          deviceData_ ;
  cudaChannelFormatDesc    channelDesc_ ;
  cudaExtent          extent_ ;
  int             isTexture_ ;
};

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

class gpuIntArray {
public:
	gpuIntArray(int initDims[3]);
        gpuIntArray(int singleDim);
	~gpuIntArray( );
	
	void sendDataToGPU( int* hostMemory );
	void getDataFromGPU( int* hostMemory );
	
	int* getDevicePtr( );
	
	int getDim( int dimension );
	
private:
	int							dims_[3];
	int*						deviceData_;
};

class coneBeamGeometry {
public:
  coneBeamGeometry(int volD[3], int projD[3],  int startV[3], int endV[3], int projDataSize[2], int* projDataWStart, int* projDataHStart, float* angles, float* hMid, float sourceSampleDist, float pixelW);
  coneBeamGeometry(int volD[3], int projD[3],  int startV[3], int endV[3], int projDataSize[2], int* projDataWStart, int* projDataHStart, float* angles, float* hMid, float sourceSampleDist, float* misAlignments, float pixelW);
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
  float pixelWidth;
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
  void _init(int volD[3], int projD[3],  int startV[3], int endV[3], int projDataSize[2], int* projDataWStart, int* projDataHStart, float* angles, float* hMid, float sourceSampleDist, float* misAlignments, float pixelW);
  void _calculateDimensions(int volD[3], int projD[3], int startV[3], int endV[3], int projDataSize[2], int* projDataWStart, int* projDataHStart);
  void _freeDimensions();
  void _freeSubProblem();
  void _setupSubProblem( int newSubVolDim[3] , int numProj );
  void _freeProblemGeometry();
  void _calculateProblemGeometry(float* angles, float* hMid, float sourceSampleDist, float* misAlignments, float pixelW);
  void _freeSweep();
  void _calculateSweep();
  void _projectPoint( float xx , float yy , float zz , int tt , float* ww , float* hh );
  void _projectCube(float xxBeg, float xxEnd, float yyBeg, float yyEnd, float zzBeg, float zzEnd, int tt, int* wwEdge, int* hhEdge);
  void _padArrays(int);
  void _padOneIntArray(int**,int,int);
  void _padOneFltArray(float**,int,int);
};


int getDeviceRAM( );

int getNumDevices( );

void setDeviceNum( int deviceNumber );

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
);

void backprojectSubVolume(int ss , //The index of this sub-problem
  coneBeamGeometry* geometry ,
  gpuFltArray* deviceVolume ,
  gpuTexArray* deviceProj ,
  gpuFltArray* deviceHmid ,
  gpuFltArray* deviceSina ,
  gpuFltArray* deviceCosa ,
  gpuIntArray* devicewwOffset ,
  gpuIntArray* devicehhOffset ,
  int executionMode , //Either __SYNC__ or __ASYNC__.
  int kernelMode //Either __KATSEVICH__ or __FELDKAMP__
);

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
);
