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

extern "C" {
  //Deprecated backprojection function without kernelMode option. Included for backwards compatibility.
  void GPU_CB_backproject(int GPUID,
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
                          float* normalizationFactor);

  //New backprojecton function, with kernelMode options. kernelMode 1 = Katsevich, 2 = Feldkamp (actually for now anything other than 1 is FDK)
  void GPU_CB_K_backproject(int GPUID,
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
                            float pixelWidth,
                            int kernelMode);

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
                                   float* misAlignments,
                                   float pixelWidth);

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
                               float* misAlignments,
                               float pixelWidth);

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
                      float* normalizationFactor,
                      float pixelWidth);

}
