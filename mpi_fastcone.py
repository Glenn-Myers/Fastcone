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


"""
GENERAL OVERVIEW:
This is an mpi layer intended to sit between fastcone.py and the user. The structures specify a certain "defualt" arrangement of data, and
contain the absolute minimum amount of data. For particular applications, these structures may need to be expanded or altered; rather than altering
the lightweight default class in this file, users should create their own classes using the following as base classes.

There are 2 modes of invocation for the project and backproject functions: simple and advanced.

ASSUMPTIONS ABOUT DATA STRUCTURE:
Projections are stored in a minimum unit of one complete projection image. Some CPUs may be missing projection information entirely, and others may have
multiple projections.
Volume information is stored in a minimum unit of a "chunk", which is at least 256 on each side. Some CPUs may be missing volume information entirely, and
others may have multiple chunks.

SIMPLE INVOCATION:


ADVANCED INVOCATION:


TODO:
* Split out certain parts of INIT statements, for easy replacement in inheriting classes.
* Overload init statements, for (e.g.) generating the numbers directly from filenames, and/or generating vInfo from pInfo.s
* Optimise MPI send/recv order to avoid deadlocks.
"""

#Setup paths
import sys
sys.path.append("/home/110/grm110/lib/python/")
sys.path.append("/home/110/grm110/lib/python/mpi4py")
sys.path.append("/home/110/grm110/lib/python/am_devlib")

#Import packages
import math
import numpy as np
import fastcone as cone
from mpi4py import MPI
#For readXCT
import netCDF4
import os
import string

from fastcone import INT_TYPE
from fastcone import FLOAT_TYPE

class infoMPI:
  """
  self.comm = the communicator in question.
  self.size = the size of the communicator
  self.rank = the rank of this process
  self.nodeOfCPU[rank] = the node that CPU rank is on.
  self.isG = Does this process have a GPU attached?
  self.numG = The total number of GPUs
  self.rankOfGPU[GPU number] = an array of dimension numG. It maps the index of the GPU, to the index of the attached CPU.
  self.localIndexOfGPU[GPU_number] = an array of dimension numG. It maps the index of the GPU, to the index of the GPU on the local node. So GPU_number 5 might return index 1, on node 2.
  """
  def __init__(self, comm, GPUsPerNode=3, GPUsOffset=0, oneToMany=False, verbose=True):
    """
    Class constructor.
    comm = the MPI communicator in question.
    GPUsPerNode = the number of GPUs per compute node to be used by this process.
    GPUsOffset = the number of GPUs per node to ignore. For example, if GPUsPerNode = 3, and GPUsOffset = 2, then GPUs 2, 3, and 4 will be used by this process.
    oneToMany = If False, one CPU thread is linked to one GPU thread. Otherwise, many CPU threads are linked to each GPU thread.
    """
    #Retrieve data directly from communicator
    self.comm = comm
    self.size = comm.Get_size()
    self.rank = comm.Get_rank()
    #Build a list of all the node names.
    nodeName = MPI.Get_processor_name()
    listOfNodeNames = []
    for rr in range(self.size):
      listOfNodeNames.append(comm.bcast(nodeName, root=rr))
    listOfUniqueNodeNames = list(set(listOfNodeNames)) #Whittles the list down to its unique elements.
    #Build the list of which node each CPU is on.
    self.nodeOfCPU = np.zeros(self.size, dtype = INT_TYPE)
    rankOnThisNode = np.zeros(self.size, dtype = INT_TYPE)
    for rr in range(self.size):
      self.nodeOfCPU[rr] = listOfUniqueNodeNames.index(listOfNodeNames[rr])
      #For every rank, figure out which rank they are within their own node
      rankOnThisNode[rr] = np.sum(self.nodeOfCPU[0:rr] == self.nodeOfCPU[rr])
    #Reconfigure ranks if it's one-to-many execution.
    if oneToMany:
      rankOnThisNode = rankOnThisNode % GPUsPerNode #This line changes it so that every process links to a GPU, rather than it being 1:1
    #Decide whether there is a GPU on this node.
    self.isG = (rankOnThisNode[self.rank] < GPUsPerNode)
    #Figure out how many GPUs there are
    self.numG = comm.allreduce(sendobj=int(self.isG), op=MPI.SUM)
    #Build arrays for rank and local index
    self.rankOfGPU = np.zeros(self.numG, dtype = INT_TYPE)
    self.localIndexOfGPU = np.zeros(self.numG, dtype = INT_TYPE)
    #Check if we have a GPU on this processor
    if self.isG:
      #Figure out the index of this GPU
      gg = np.sum(rankOnThisNode[0:self.rank] < GPUsPerNode)
      self.rankOfGPU[gg] = self.rank
      self.localIndexOfGPU[gg] = rankOnThisNode[self.rank] + GPUsOffset
    else:
      gg = None
    #Reduce down the rankOfGPU and localIndexOfGPU arrays
    self.rankOfGPU = comm.allreduce(sendobj=self.rankOfGPU, op=MPI.SUM) 
    self.localIndexOfGPU = comm.allreduce(sendobj=self.localIndexOfGPU, op=MPI.SUM)
    #Print out a report on what has been decided
    if (verbose == True) and (self.rank == 0):
      print("GPU allocation complete.")
      for gg in range(self.numG):
        print("Logical GPU ", gg, " is physical GPU ", self.localIndexOfGPU[gg], " on node ", self.nodeOfCPU[self.rank], " and is attached to CPU rank ", self.rankOfGPU[gg])
      print("GPUs per node ", GPUsPerNode, ", and GPU offset is ", GPUsOffset)


class infoProj:
        """
        Contains information on how the projection data is arranged across the CPUs.
        self.dims = #[dimension] The (global) dimensions of the projection data set. Dims are in order of [theta, height, width].
        self.localDims = #[rank, dimension] = The dimensions of the array used to store projection data on CPU="rank". Equal to zero when no projection data present.
        self.onCPU = #[projection]
        self.localIndex = #[projection]
        self.cpuHasData[rank] = a boolean, set to True if there is projection data here, and False otherwise.
        """
        def __init__(self, lDims, cInfo):
                """
                Input arguments:
                lDims[theta, height, width] = the dimensions of the projection data on this CPU. Zeroes if no projection data present.
                        This algorithm assumes that height and width are the height and width of the full projection data set.
                cInfo = an infoMPI structure.
                """
                #Calculate dimension information
                self.localDims = np.zeros([cInfo.size, 3], dtype=INT_TYPE)
                self.localDims[cInfo.rank, :] = lDims[:].astype(INT_TYPE)
                self.localDims = cInfo.comm.allreduce(sendobj=self.localDims, op=MPI.SUM)
                self.dims = np.array([self.localDims[:,0].sum(), self.localDims[:,1].max(), self.localDims[:,2].max()], dtype=INT_TYPE)
                #Create cpuHasProjData
                self.cpuHasData = (self.localDims[:,0] != 0)
                #Create onCPU and localIndex. TODO: vectorise this code.
                pp = 0
                self.onCPU = np.zeros(self.dims[0])
                self.localIndex = np.zeros(self.dims[0])
                for rank in range(0,cInfo.size,1):
                        if self.cpuHasData[rank]:
                                self.onCPU[pp:(pp+self.localDims[rank,0])] = rank
                                self.localIndex[pp:(pp+self.localDims[rank,0])] = np.arange(self.localDims[rank,0])
                                pp += self.localDims[rank,0]

class infoVol:
        """
        Contains information on how the projection data is arranged across the CPUs.
        BE CAREFUL NOT TO CONFUSE ARRAYOFFSET AND LOCALOFFSET.
        self.dims[dimension] = A 4-element array, containing the dimensions of the (global) volume data set, in the order [t,z,y,x].
        self.chunkDims[dimension] = A 4-element array, containing the dimensions of a chunk, in the order accepted by np.zeros: [t,z,y,x].
        self.localDims[rank, dimension] = A 4-element array, containing the dimensions of the array for volume on the CPU="rank", in the order accepted by np.zeros: [t,z,y,x]. Equal to zero if isV ~= 0.
        self.arrayOffset[rank, dimension] = An array containing the offset of the array for volume on the CPU="rank".
        All the following elements in this class are indexed by chunk number.
        self.onCPU = #[chunk]
        self.cpuHasData[rank] = boolean.
        self.offset = #offset[chunk,dimension] = (t,z,y,x) coordinates of the start of this chunk
        self.localOffset = #offset[chunk,dimension] = (t,z,y,x) coordinates of the start of this chunk, relative to the volume array on the local CPU.
        """
        def __init__(self, lDims, lStart, totalDims, cInfo, chunk=np.array([1,512,512,512], dtype=INT_TYPE)):
                """
                Input arguments:
                lDims = the dimensions of the volume data on the local CPU.
                lStart = the offset of this data from the [0,0,0,0] of the global data set.
                totalDims = the dimensions of the total data set.
                These should both be set to zeroes if there's no volume data present.
                """
                #Define chunk, local, and global dimensions
                self.chunkDims = chunk.astype(INT_TYPE).copy()
                self.localDims = np.zeros([cInfo.size,4], dtype=INT_TYPE)
                self.localDims[cInfo.rank, :] = lDims[:].astype(INT_TYPE)
                self.localDims = cInfo.comm.allreduce(sendobj=self.localDims, op=MPI.SUM)
                self.dims = totalDims.astype(INT_TYPE)
                #Determine which CPUs have volume data on them.
                self.cpuHasData = (self.localDims[:,0] != 0)
                #Spread around lStart information
                localStart = np.zeros([cInfo.size,4], dtype=INT_TYPE)
                localStart[cInfo.rank, :] = lStart[:].astype(INT_TYPE)
                localStart = cInfo.comm.allreduce(sendobj=localStart, op=MPI.SUM)
                self.arrayOffset = localStart
                #Calculate how many chunks we have in all, so that we can allocate the arrays for onCPU, offset, and localOffset
                numchunks = self.localDims / self.chunkDims
                numchunks = np.sum(numchunks[:,0]*numchunks[:,1]*numchunks[:,2]*numchunks[:,3])
                self.onCPU = np.zeros(numchunks, dtype=INT_TYPE)
                self.offset = np.zeros([numchunks,4], dtype=INT_TYPE)
                self.localOffset = np.zeros([numchunks,4], dtype=INT_TYPE)
                #For every chunk, calculate onCPU, offset, and localOffset
                chunk = 0
                for rank in range(0,cInfo.size,1):
                        if self.cpuHasData[rank]:
                                step = (self.localDims[rank,0]/self.chunkDims[0]) * (self.localDims[rank,1]/self.chunkDims[1]) * (self.localDims[rank,2]/self.chunkDims[2]) * (self.localDims[rank,3]/self.chunkDims[3])
                                self.onCPU[chunk:(chunk+step)] = rank
                                temp = np.indices(self.localDims[rank,:] / self.chunkDims)
                                temp = np.transpose(temp.reshape([4, temp.size/4])) #Now indexed [chunk,dims]
                                temp *= self.chunkDims #Broadcasting should occur over second dimension (i.e. chunk), even if there are 4 chunks.
                                self.localOffset[chunk:(chunk+step), :] = temp[:,:]
                                self.offset[chunk:(chunk+step), :] = self.localOffset[chunk:(chunk+step), :] + localStart[rank,:]
                                chunk += step

class infoGeometry:
        """
        Input arguments:
        pInfo - an infoProj structure
        vInfo - an infoVol structure
        localAngles - the projection angles corresponding to the projection data on this CPU. Should be None if no projection data is present, but it's ignored in that case anyway.
        zSourceIndex[local projection index] = the z index (measured from element [0,0,0] of the volume) of the source AND detector centre.
        sampleDist = the distance from the source to the centre of the sample volume, in pixels.
        self.angle
        self.zSourceIndex[global projection index] = see zSourceIndex.
        self.sampleDistance
        self.pSize[chunk] = the number of projections related to this chunk
        self.wStart[chunk,ii] = the global w coordinate of the [0,0] element of proj_G array. Must be a valid index; no need to zero-pad proj_G.
        self.wSize[chunk,ii]
        self.hStart[chunk,ii] = the global h coordinate of the [0,0] element of proj_G array. Must be a valid index; no need to zero-pad proj_G.
        self.hSize[chunk,ii]
        self.projWeightings[chunk,ii] = the weighting applied to the projection operation from this chunk, when projected to the iith relevant projection.
        self.backWeightings[chunk,ii]
        """
        
        def __init__(self, pInfo, vInfo, cInfo, localAngles, zSourceIndex, sampleDist, misAlignments=np.array([0,0,0,0,0,0], dtype=FLOAT_TYPE), projStride=1, projOffset=0):
                #Assemble geometry information from arguments.
                self.misAlignments=misAlignments
                self.sampleDistance = sampleDist
                self.angle = np.zeros(pInfo.dims[0], dtype=FLOAT_TYPE)
                self.zSourceIndex = np.zeros(pInfo.dims[0], dtype=FLOAT_TYPE)
                if pInfo.cpuHasData[cInfo.rank]:
                        self.angle[pInfo.onCPU == cInfo.rank] = localAngles[:]
                        self.zSourceIndex[pInfo.onCPU == cInfo.rank] = zSourceIndex[:]
                self.angle = cInfo.comm.allreduce(sendobj=self.angle, op=MPI.SUM)
                self.zSourceIndex = cInfo.comm.allreduce(sendobj=self.zSourceIndex, op=MPI.SUM)
                #Determine pStart and pSize.
                self.pSize = np.zeros(vInfo.onCPU.size, dtype=INT_TYPE)
                self.pKey = np.zeros([vInfo.onCPU.size, pInfo.dims[0]], dtype=INT_TYPE)
                for chunk in range(self.pKey.shape[0]):
                  ppLocal = 0;
                  for ppGlobal in range(self.pKey.shape[1]):
                    corners = self._findCorners(pInfo, vInfo, chunk, ppGlobal) #[dimension,corner number]
                    wS = np.min(corners[1,:])
                    wE = np.max(corners[1,:])
                    hS = np.min(corners[0,:])
                    hE = np.max(corners[0,:])
                    isValid = (wS <= pInfo.localDims[pInfo.onCPU[ppGlobal],2]) and (wE >= 0.0)
                    isValid = isValid and ((hS <= pInfo.localDims[pInfo.onCPU[ppGlobal],2]) and (hE >= 0.0))
                    isValid = isValid and (((ppGlobal + projOffset) % projStride) == 0)
                    if isValid:
                      self.pKey[chunk, ppLocal] = ppGlobal
                      ppLocal += 1
                  self.pSize[chunk] = ppLocal
                #Determine weightings.
                self.projWeightings = np.ones([vInfo.onCPU.size, self.pSize.max()], dtype=FLOAT_TYPE)
                self.backWeightings = np.ones(self.projWeightings.shape, dtype=FLOAT_TYPE) / pInfo.dims[0]
                #Determine hStart, hSize, wStart, and wSize.
                self.wStart = np.zeros(self.projWeightings.shape, dtype=INT_TYPE)
                self.wSize = np.zeros(self.projWeightings.shape, dtype=INT_TYPE)
                self.hStart = np.zeros(self.projWeightings.shape, dtype=INT_TYPE)
                self.hSize = np.zeros(self.projWeightings.shape, dtype=INT_TYPE)
                for chunk in range(0,vInfo.onCPU.size,1):
                        for ppLocal in range(0, self.pSize[chunk], 1):
                                pp = self.pKey[chunk, ppLocal]
                                corners = self._findCorners(pInfo, vInfo, chunk, pp) #[dimension,corner number]
                                #print("Chunk ", chunk, "proj corners: ", corners)
                                self.wStart[chunk,ppLocal] = np.floor(np.clip(np.min(corners[1,:]), 0, pInfo.localDims[pInfo.onCPU[pp],2] - 1))
                                self.wSize[chunk,ppLocal] = np.ceil(np.clip(np.max(corners[1,:]), 0, pInfo.localDims[pInfo.onCPU[pp],2] - 1)) - self.wStart[chunk,ppLocal] + 1
                                self.hStart[chunk,ppLocal] = np.floor(np.clip(np.min(corners[0,:]), 0, pInfo.localDims[pInfo.onCPU[pp],1] - 1))
                                self.hSize[chunk,ppLocal] = np.ceil(np.clip(np.max(corners[0,:]), 0, pInfo.localDims[pInfo.onCPU[pp],1] - 1)) - self.hStart[chunk,ppLocal] + 1
                #np.set_printoptions(threshold=np.nan)
                #print("zSourceIndex = ", self.zSourceIndex)

        def _findCorners(self, pInfo, vInfo, chunk, pp):
                #Repeat this once for each corner. Only floating point values are required.
                projCorners = np.zeros([2,8], dtype=FLOAT_TYPE)
                volCorners = np.indices([2,2,2], dtype=FLOAT_TYPE)
                volCorners = np.transpose(volCorners.reshape(3,8)) #volCorners is now indexed by [corner, dimension]
                volCorners *= vInfo.chunkDims[1:4]
                volCorners += vInfo.offset[chunk,1:4]
                #print("Chunk ", chunk, "corners: ", volCorners)
                for corner in np.arange(8):
                        projCorners[0,corner] = self._projPoint(volCorners[corner,:].copy(), pp, pInfo, vInfo)[0]
                        projCorners[1,corner] = self._projPoint(volCorners[corner,:].copy(), pp, pInfo, vInfo)[1]
                return projCorners
        def _projPoint(self, objectPos, pp, pInfo, vInfo):
                objectPos -= (vInfo.dims[1:4] // 2) #Convert array indicies to coordinates
                def _normalize(a):
                  return a / np.linalg.norm(a)
                misAlignments = self.misAlignments
                if (misAlignments == None) or np.all(misAlignments == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
                        #Do projection calculations.
                        temp = self.sampleDistance / (self.sampleDistance + objectPos[2]*np.cos(self.angle[pp]) + objectPos[1]*np.sin(self.angle[pp]))
                        ww = (objectPos[1]*np.cos(self.angle[pp]) - objectPos[2]*np.sin(self.angle[pp])) * temp
                        hh = (objectPos[0] - (self.zSourceIndex[pp] - vInfo.dims[1] // 2)) * temp 
                else:
                        #Build geometry vectors
                        sourcePos = np.array([self.zSourceIndex[pp] - vInfo.dims[1] / 2.0, -1.0 * np.sin(self.angle[pp]) * self.sampleDistance, -1.0 * np.cos(self.angle[pp]) * self.sampleDistance])
                        detectorPos = np.array([self.zSourceIndex[pp] - vInfo.dims[1] / 2.0, 0.0, 0.0])
                        hStep = np.array([1.0, 0.0, 0.0])
                        wStep = np.array([0.0, np.cos(self.angle[pp]), -1.0 * np.sin(self.angle[pp])])
                        #Apply misalignment Dw
                        detectorPos += misAlignments[0] * wStep
                        #Apply misalignment Dh
                        detectorPos += misAlignments[1] * hStep
                        #Apply misalignment Dl
                        detectorPos += misAlignments[2] * _normalize(detectorPos - sourcePos)
                        #Apply misalignment Dphi, a rotation about the w axis.
                        whPerp = np.cross(wStep, hStep)
                        hStep = np.linalg.norm(hStep) * (np.cos(misAlignments[3]) * _normalize(hStep) + np.sin(misAlignments[3]) * _normalize(whPerp))
                        #Apply misalignment Dtheta, a rotation about the h axis.
                        whPerp = np.cross(wStep, hStep)
                        wStep = np.linalg.norm(wStep) * (np.cos(misAlignments[4]) * _normalize(wStep) + np.sin(misAlignments[4]) * _normalize(whPerp))
                        #Apply misalignment Dpsi, a rotation in the w-h plane.
                        whPerp = np.cross(wStep, hStep)
                        wnPerp = np.cross(wStep, whPerp)
                        hnPerp = np.cross(hStep, whPerp)
                        wStep = np.linalg.norm(wStep) * (np.cos(misAlignments[5]) * _normalize(wStep) - np.sin(misAlignments[5]) * _normalize(wnPerp))
                        hStep = np.linalg.norm(hStep) * (np.cos(misAlignments[5]) * _normalize(hStep) + np.sin(misAlignments[5]) * _normalize(hnPerp))
                        #Calculate normal unit vector to the detector plane.
                        whPerp = np.cross(wStep, hStep)
                        whPerp /= np.linalg.norm(whPerp)
                        #Calculate distance of plane-ray intersection from the source.
                        distanceFromSource = np.dot(detectorPos - sourcePos, whPerp) / np.dot(_normalize(objectPos - sourcePos), whPerp)
                        #Calculate the point of intersection, as an offset from the centre of the detector.
                        intersection = sourcePos + distanceFromSource * _normalize(objectPos - sourcePos) - detectorPos
                        #Take some dot products to make the next step readable, and then calculate wc and hc. We ackowledge that wStep and hStep need not be orthogonal, or normalized.
                        wDotW = np.dot(wStep, wStep)
                        wDotH = np.dot(wStep, hStep)
                        hDotH = np.dot(hStep, hStep)
                        iDotW = np.dot(intersection, wStep)
                        iDotH = np.dot(intersection, hStep)
                        ww = (iDotW * hDotH - iDotH * wDotH) / (wDotW * hDotH - wDotH * wDotH)
                        hh = (iDotH * wDotW - iDotW * wDotH) / (wDotW * hDotH - wDotH * wDotH)
                #Convert wc and hc to array indicies.
                ww += (pInfo.dims[2] // 2)
                hh += (pInfo.dims[1] // 2)
                #Return a tuple
                return np.array([hh, ww], dtype=FLOAT_TYPE)


def project(cpuInfo, volInfo, projInfo, geomInfo, vol_V, proj_P, misAlignments=np.array([0,0,0,0,0,0], dtype=FLOAT_TYPE), verbose=True):
        #DOES NOT ZERO THE PROJECTION DATA; ADDS THE RESULTS TO IT.
        #Allocate the projection and volume arrays on the GPU before beginning
        if cpuInfo.isG:
                vol_G = np.zeros([volInfo.chunkDims[1], volInfo.chunkDims[2], volInfo.chunkDims[3]], dtype=FLOAT_TYPE)
        #Take however many passes at the problem we need. There are ceil(numChunks/numGPUs) chunks per GPU, with the final GPU having fewer. GPU 1 handles chunks [0,chunksPerGPU)
        chunksPerGPU = (volInfo.onCPU.size+cpuInfo.numG-1) // cpuInfo.numG
        for ii in range(0, chunksPerGPU):
                #Get all the volume chunks in position. We're using the FOR loop to avoid any possible send/recieve deadlocks.
                for chunk in range(ii, volInfo.onCPU.size, chunksPerGPU):
                        sourceRank = volInfo.onCPU[chunk]
                        destRank = cpuInfo.rankOfGPU[chunk/chunksPerGPU]
                        for zz_G in range(volInfo.chunkDims[1]):
                            zz = zz_G + volInfo.localOffset[chunk,1]
                            if ((cpuInfo.rank == sourceRank) and (cpuInfo.rank == destRank)):
                                if (zz_G == 0): 
                                  if verbose: 
                                    print("MPI Project: Rank ", cpuInfo.rank, "Sending and recieving volume data for chunk", chunk)
                                vol_G[zz_G,:,:] = vol_V[volInfo.localOffset[chunk,0]:(volInfo.localOffset[chunk,0]+volInfo.chunkDims[0]),
                                                zz,
                                                volInfo.localOffset[chunk,2]:(volInfo.localOffset[chunk,2]+volInfo.chunkDims[2]),
                                                volInfo.localOffset[chunk,3]:(volInfo.localOffset[chunk,3]+volInfo.chunkDims[3])]
                            elif ((cpuInfo.rank == sourceRank) and (cpuInfo.rank != destRank)):
                                if (zz_G == 0): 
                                  if verbose: 
                                    print("MPI Project: Rank ", cpuInfo.rank, "Sending volume data for chunk", chunk)
                                cpuInfo.comm.send(vol_V[volInfo.localOffset[chunk,0]:(volInfo.localOffset[chunk,0]+volInfo.chunkDims[0]),
                                                                zz,
                                                                volInfo.localOffset[chunk,2]:(volInfo.localOffset[chunk,2]+volInfo.chunkDims[2]),
                                                                volInfo.localOffset[chunk,3]:(volInfo.localOffset[chunk,3]+volInfo.chunkDims[3])], dest=destRank, tag=chunk)
                            elif ((cpuInfo.rank != sourceRank) and (cpuInfo.rank == destRank)):
                                if (zz_G == 0):
                                  if verbose: 
                                    print("MPI Project: Rank ", cpuInfo.rank, "Recieving volume data for chunk", chunk)
                                vol_G[zz_G,:,:] = cpuInfo.comm.recv(source=sourceRank, tag=chunk)	
                #Project the volume chunk
                for chunk in range(ii, volInfo.onCPU.size, chunksPerGPU):
                        GPUrank = cpuInfo.localIndexOfGPU[chunk/chunksPerGPU] #Should work, because these both should be integers
                        processingRank = cpuInfo.rankOfGPU[chunk/chunksPerGPU]
                        if (cpuInfo.rank == processingRank):
                                if verbose: print("MPI Project: Rank ", cpuInfo.rank, "Processing data for chunk", chunk)
                                proj_G = np.zeros([geomInfo.pSize[chunk], geomInfo.hSize[chunk,:].max(), geomInfo.wSize[chunk,:].max()], dtype=FLOAT_TYPE)
                                angles_G = geomInfo.angle[geomInfo.pKey[chunk][0:geomInfo.pSize[chunk]]]
                                zSourceIndex_G = geomInfo.zSourceIndex[geomInfo.pKey[chunk][0:geomInfo.pSize[chunk]]]
                                wOffset_G = geomInfo.wStart[chunk,0:geomInfo.pSize[chunk]]
                                hOffset_G = geomInfo.hStart[chunk,0:geomInfo.pSize[chunk]]
                                pDims_G = np.array([geomInfo.pSize[chunk], projInfo.dims[1], projInfo.dims[2]], dtype=INT_TYPE)
                                weight_G = geomInfo.projWeightings[chunk,0:geomInfo.pSize[chunk]]
                                cone.project(vol_G, proj_G, angles_G, geomInfo.sampleDistance, zSourceIndex_G, volStart=volInfo.offset[chunk,1:4], volDims=volInfo.dims[1:4], projWOffset=wOffset_G, projHOffset=hOffset_G, projDims=pDims_G, norm=weight_G, misAlignments=misAlignments, GPUID=GPUrank)
                                #print "Done with projection on CPU ", cpuInfo.rank, "\n"
                #move the projections to the right processors. Again, using the FOR loop to avoid any send/recieve deadlocks.
                for chunk in range(ii, volInfo.onCPU.size, chunksPerGPU):
                        sourceRank = cpuInfo.rankOfGPU[chunk/chunksPerGPU]
                        for pp_G in range(0, geomInfo.pSize[chunk], 1):
                                pp = geomInfo.pKey[chunk][pp_G]
                                destRank = projInfo.onCPU[pp]
                                if ((cpuInfo.rank == sourceRank) or (cpuInfo.rank == destRank)):
                                        transferTag = chunk*geomInfo.pSize.max() + pp_G
                                        hMin = geomInfo.hStart[chunk, pp_G]
                                        hMax = geomInfo.hStart[chunk, pp_G] + geomInfo.hSize[chunk, pp_G]
                                        wMin = geomInfo.wStart[chunk, pp_G]
                                        wMax = geomInfo.wStart[chunk, pp_G] + geomInfo.wSize[chunk, pp_G]
                                if ((cpuInfo.rank == sourceRank) and (cpuInfo.rank == destRank)):
                                        if (pp_G == 0):
                                          if verbose:
                                            print("MPI Project: Rank ", cpuInfo.rank, "Sending and Recieving projection data for chunk", chunk)
                                        proj_P[projInfo.localIndex[pp], hMin:hMax, wMin:wMax] += proj_G[pp_G, 0:geomInfo.hSize[chunk,pp_G], 0:geomInfo.wSize[chunk,pp_G]]
                                elif ((cpuInfo.rank == sourceRank) and (cpuInfo.rank != destRank)):
                                        if (pp_G == 0):
                                          if verbose:
                                            print("MPI Project: Rank ", cpuInfo.rank, "Sending projection data for chunk", chunk)
                                        cpuInfo.comm.send(proj_G[pp_G, 0:geomInfo.hSize[chunk,pp_G], 0:geomInfo.wSize[chunk,pp_G]], dest=destRank, tag=transferTag)
                                elif ((cpuInfo.rank != sourceRank) and (cpuInfo.rank == destRank)):
                                        if (pp_G == 0):
                                          if verbose:
                                            print("MPI Project: Rank ", cpuInfo.rank, "Recieving projection data for chunk", chunk)
                                        proj_P[projInfo.localIndex[pp], hMin:hMax, wMin:wMax] += cpuInfo.comm.recv(source=sourceRank, tag=transferTag)
        return 1
	
def backproject(cpuInfo, volInfo, projInfo, geomInfo, vol_V, proj_P, misAlignments=None, kernelMode=2, verbose=True):
        #DOES NOT ZERO THE VOLUME DATA; ADDS THE RESULTS TO IT.
        #Allocate the projection and volume arrays on the GPU before beginning
        if cpuInfo.isG:
                vol_G = np.zeros([volInfo.chunkDims[1], volInfo.chunkDims[2], volInfo.chunkDims[3]], dtype=FLOAT_TYPE)
        #Take however many passes at the problem we need
        chunksPerGPU = (volInfo.onCPU.size+cpuInfo.numG-1) // cpuInfo.numG
        for ii in range(0, chunksPerGPU):
                #Move the projections to the right processors. Again, using the FOR loop to avoid any send/recieve deadlocks.
                for chunk in range(ii, volInfo.onCPU.size, chunksPerGPU):
                        destRank = cpuInfo.rankOfGPU[chunk // chunksPerGPU]
                        if (cpuInfo.rank == destRank):
                                proj_G = np.zeros([geomInfo.pSize[chunk], geomInfo.hSize[chunk,:].max(), geomInfo.wSize[chunk,:].max()], dtype=FLOAT_TYPE)
                        for pp_G in range(0, geomInfo.pSize[chunk], 1):
                                pp = geomInfo.pKey[chunk][pp_G]
                                sourceRank = projInfo.onCPU[pp]
                                if ((cpuInfo.rank == sourceRank) or (cpuInfo.rank == destRank)):
                                        transferTag = chunk*geomInfo.pSize.max() + pp_G
                                        hMin = geomInfo.hStart[chunk, pp_G]
                                        hMax = geomInfo.hStart[chunk, pp_G] + geomInfo.hSize[chunk, pp_G]
                                        wMin = geomInfo.wStart[chunk, pp_G]
                                        wMax = geomInfo.wStart[chunk, pp_G] + geomInfo.wSize[chunk, pp_G]
                                        #print("WARNING: HARD-CODED hMin, hMax, wMin and wMax")
                                        #print("WARNING: changed next few lines because python a:b indexing is INCLUSIVE")
                                if ((cpuInfo.rank == sourceRank) and (cpuInfo.rank == destRank)):
                                        if (pp_G == 0):
                                          if verbose: print("MPI Backproject: Rank ", cpuInfo.rank, "Recieving and sending projection data for chunk", chunk)
                                        proj_G[pp_G, 0:(geomInfo.hSize[chunk,pp_G]-0), 0:(geomInfo.wSize[chunk,pp_G]-0)] = proj_P[projInfo.localIndex[pp], hMin:hMax, wMin:wMax]
                                elif ((cpuInfo.rank == sourceRank) and (cpuInfo.rank != destRank)):
                                        if (pp_G == 0):
                                          if verbose: print("MPI Backproject: Rank ", cpuInfo.rank, "Sending projection data for chunk", chunk)
                                        cpuInfo.comm.send(proj_P[projInfo.localIndex[pp], hMin:hMax, wMin:wMax], dest=destRank, tag=transferTag)
                                elif ((cpuInfo.rank != sourceRank) and (cpuInfo.rank == destRank)):
                                        if (pp_G == 0):
                                          if verbose: print("MPI Backproject: Rank ", cpuInfo.rank, "Recieving projection data for chunk", chunk)
                                        proj_G[pp_G, 0:(geomInfo.hSize[chunk,pp_G]-0), 0:(geomInfo.wSize[chunk,pp_G]-0)] = cpuInfo.comm.recv(source=sourceRank, tag=transferTag)
                for chunk in range(ii, volInfo.onCPU.size, chunksPerGPU):
                        GPUrank = cpuInfo.localIndexOfGPU[chunk // chunksPerGPU] #Should work, because these both should be integers
                        processingRank = cpuInfo.rankOfGPU[chunk // chunksPerGPU]
                        if (cpuInfo.rank == processingRank):
                                if verbose: print("MPI Backproject: Rank ", cpuInfo.rank, "Processing chunk ", chunk, " on GPU.")
                                vol_G[...] = 0.0
                                angles_G = geomInfo.angle[geomInfo.pKey[chunk][0:geomInfo.pSize[chunk]]]
                                zSourceIndex_G = geomInfo.zSourceIndex[geomInfo.pKey[chunk][0:geomInfo.pSize[chunk]]]
                                wOffset_G = geomInfo.wStart[chunk,0:geomInfo.pSize[chunk]]
                                hOffset_G = geomInfo.hStart[chunk,0:geomInfo.pSize[chunk]]
                                pDims_G = np.array([geomInfo.pSize[chunk], projInfo.dims[1], projInfo.dims[2]], dtype=INT_TYPE)
                                weight_G = geomInfo.backWeightings[chunk,0:geomInfo.pSize[chunk]]
                                cone.backproject(vol_G, proj_G, angles_G, geomInfo.sampleDistance, zSourceIndex_G, volStart=volInfo.offset[chunk,1:4], volDims=volInfo.dims[1:4], projWOffset=wOffset_G, projHOffset=hOffset_G, projDims=pDims_G, norm=weight_G, GPUID=GPUrank, kernelMode=kernelMode, misAlignments=misAlignments) 
                #Send the volume data home.
                for chunk in range(ii, volInfo.onCPU.size, chunksPerGPU):
                        destRank = volInfo.onCPU[chunk]
                        sourceRank = cpuInfo.rankOfGPU[chunk // chunksPerGPU]
                        for zz_G in range(volInfo.chunkDims[1]):
                            zz = zz_G + volInfo.localOffset[chunk,1]
                            if ((cpuInfo.rank == sourceRank) and (cpuInfo.rank == destRank)):
                                if (zz_G == 0): 
                                  if verbose: 
                                    print("MPI Backproject: Rank ", cpuInfo.rank, "Recieving and sending processed data for chunk", chunk)
                                vol_V[volInfo.localOffset[chunk,0]:(volInfo.localOffset[chunk,0]+volInfo.chunkDims[0]),
                                      zz,
                                      volInfo.localOffset[chunk,2]:(volInfo.localOffset[chunk,2]+volInfo.chunkDims[2]),
                                      volInfo.localOffset[chunk,3]:(volInfo.localOffset[chunk,3]+volInfo.chunkDims[3])] += vol_G[zz_G,:,:]
                            elif ((cpuInfo.rank == sourceRank) and (cpuInfo.rank != destRank)):
                                if (zz_G == 0): 
                                  if verbose: 
                                    print("MPI Backproject: Rank ", cpuInfo.rank, "Sending processed data for chunk", chunk)
                                cpuInfo.comm.send(vol_G[zz_G,:,:], dest=destRank, tag=chunk)
                            elif ((cpuInfo.rank != sourceRank) and (cpuInfo.rank == destRank)):
                                if (zz_G == 0): 
                                  if verbose: 
                                    print("MPI Backproject: Rank ", cpuInfo.rank, "Recieving processed data for chunk", chunk)
                                vol_V[volInfo.localOffset[chunk,0]:(volInfo.localOffset[chunk,0]+volInfo.chunkDims[0]),
                                      zz,
                                      volInfo.localOffset[chunk,2]:(volInfo.localOffset[chunk,2]+volInfo.chunkDims[2]),
                                      volInfo.localOffset[chunk,3]:(volInfo.localOffset[chunk,3]+volInfo.chunkDims[3])] += cpuInfo.comm.recv(source=sourceRank, tag=chunk)
        return 1

def simpleProject(comm, proj_P, vol_V, angles, volOffset, volDims, zSourceIndex, sampleDistance, chunk=np.array([1,512,512,512], dtype=INT_TYPE), misAlignments=np.array([0,0,0,0,0,0], dtype=FLOAT_TYPE)):
        """
        A simple, mpi projection routine. Doesn't zero the projection data before projecting to it; adds the projected volume to whatever is already there.
        comm = an MPI communicator.
        proj_P[theta, height, width] = the projection data on this CPU. Must be a stack of n complete projections. Set to None if no projection data on this CPU.
        vol_V[t, z, y, x] = the volume data on this CPU. Set to None if no projection data. Must be 4D. Set to None if no volume data is on this CPU. If volume data is present, must be able to be divided into an integer number of "chunks" - see below. Complete coverage is not needed; i.e. the vol_V across the entire MPI_world need not completely cover the volume data set; and missing regions are assumed to be zero.
        angles[theta] = the projection angles of the data on this CPU.
        volOffset[t, z, y, x] = the starting coordinates of vol_V, relative to the total volume reconstruction problem. For example, vol_V.shape = [1,256,512,512], volOffset = [0, 256, 0, 0], and volDims = [1, 512, 512, 512], means that this CPU has the bottom half of the volume data from a 512-ccubed reconstruction.
        volDims[t, z, y, x] = the dimensions of the total volume for the reconstruciton problem.
        zSourceIndex[local projection index] = the z index (measured from element [0,0,0] of the volume) of the source AND detector centre.
        sampleDistance = the distance between the source, and the origin of the reconstruction problem (total problem, NOT centre of vol_V), measured in pixels.
        misAlignments - Optional misalignment parameters. Parameters are as per Andrew's scheme, and are applied in the order they are specified: Dw, Dh, Dl, Dphi, Dtheta, Dpsi. First three terms are in units of "pixels". Last 3 are in units of radians. Dphi is a rotation about the w axis, Dtheta a rotation about the h axis, and Dpsi a rotation in the w-h plane.
        """
        if (proj_P == None):
                lProjDims = np.array([0,0,0], dtype=INT_TYPE)
        else:
                lProjDims = np.array(proj_P.shape, dtype=INT_TYPE)
        if (vol_V == None):
                lVolDims = np.array([0,0,0,0], dtype=INT_TYPE)
        else:
                lVolDims = np.array(vol_V.shape, dtype=INT_TYPE)
        cpuInfo = infoMPI(comm)
        projInfo = infoProj(lProjDims, cpuInfo)
        volInfo = infoVol(lVolDims, volOffset, volDims, cpuInfo, chunk=chunk)
        geomInfo = infoGeometry(projInfo, volInfo, cpuInfo, angles, zSourceIndex, sampleDistance, misAlignments=misAlignments)
        project(cpuInfo, volInfo, projInfo, geomInfo, vol_V, proj_P)
        return 1

def simpleBackproject(comm, proj_P, vol_V, angles, volOffset, volDims, zSourceIndex, sampleDistance, chunk=np.array([1,512,512,512], dtype=INT_TYPE), misAlignments=None, kernelMode=2):
        """
        A simple, mpi backprojection routine. Doesn't zero the volume data before backprojecting; adds the backprojection to whatever is there.
        comm = an MPI communicator.
        proj_P[theta, height, width] = the projection data on this CPU. Must be a stack of n complete projections. Set to None if no projection data on this CPU.
        vol_V[t, z, y, x] = the volume data on this CPU. Set to None if no projection data. Must be 4D. Set to None if no volume data is on this CPU. If volume data is present, must be able to be divided into an integer number of "chunks" - see below. Complete coverage is not needed;
 i.e. the vol_V across the entire MPI_world need not completely cover the volume data set; and missing regions are assumed to be zero.
        angles[theta] = the projection angles of the data on this CPU.
        volOffset[t, z, y, x] = the starting coordinates of vol_V, relative to the total volume reconstruction problem. For example, vol_V.shape = [1,256,512,512], volOffset = [0, 256, 0, 0], and volDims = [1, 512, 512, 512], means that this CPU has the bottom half of the volume data from a 512-ccubed reconstruction.
        volDims[t, z, y, x] = the dimensions of the total volume for the reconstruciton problem.
        zSourceIndex[local projection index] = the z index (measured from element [0,0,0] of the volume) of the source AND detector centre.
        sampleDistance = the distance between the source, and the origin of the reconstruction problem (total problem, NOT centre of vol_V), m
easured in pixels.
        misAlignments - Optional misalignment parameters. Parameters are as per Andrew's scheme, and are applied in the order they are specified: Dw, Dh, Dl, Dphi, Dtheta, Dpsi. First three terms are in units of "pixels". Last 3 are in units of radians. Dphi is a rotation about the w axis, Dtheta a rotation about the h axis, and Dpsi a rotation in the w-h plane.
        kernelMode - The type of kernel used to do the backprojection. Mode 2 is a Feldkamp-type kernel, Mode 1 is a Katsevich-type kernel, Mode 0 is an unweighted-type kernel, suitable for iterative reconstruction like SIRT. Misalignments make no sense with anything other than a type 0 filter.
        """
        #kernelMode = 2
        #zSourceIndex[:] = 1200 - zSourceIndex[:]
        #if (proj_P != None):
        #  proj_P[:,:,:] = proj_P[:,::-1,:].copy()
        #print("Hard-coded kernel mode = ", kernelMode) #, ", zSourceIndex and proj mirrored vertically.")
        if (proj_P == None):
                lProjDims = np.array([0,0,0], dtype=INT_TYPE)
        else:
                lProjDims = np.array(proj_P.shape, dtype=INT_TYPE)
        if (vol_V == None):
                lVolDims = np.array([0,0,0,0], dtype=INT_TYPE)
        else:
                lVolDims = np.array(vol_V.shape, dtype=INT_TYPE)
        #print("SimpleBackproject parameters:", comm.rank, angles.shape, angles, volOffset, volDims, lProjDims, lVolDims, zSourceIndex.shape, zSourceIndex, sampleDistance.shape, sampleDistance, chunk)
        cpuInfo = infoMPI(comm)
        projInfo = infoProj(lProjDims, cpuInfo)
        volInfo = infoVol(lVolDims, volOffset, volDims, cpuInfo, chunk=chunk)
        geomInfo = infoGeometry(projInfo, volInfo, cpuInfo, angles, zSourceIndex, sampleDistance, misAlignments=misAlignments)
        backproject(cpuInfo, volInfo, projInfo, geomInfo, vol_V, proj_P, misAlignments=misAlignments, kernelMode=kernelMode)
        return 1

def createMpiVol(comm, volDims, volLocalDims=None):
        """
        This is a utility for quickly allocating volume arrays for mpi_fastcone. It ensures complete coverage of the volume, which isn't necessary for mpi_fastcone to function.
        Nevertheless, it's the most common execution configuration, and this considerably simplifies the task of figuring out offsets.
        As always, both volLocalDims and volDims need to be 4D.
        volLocalDims and volDims are both broadcast from process 0 to all other processes.
        If volLocalDims is supplied, this figures out how many processes need to store volume data for full coverage, and allocates the arrays appropriately.
        If volLocalDims is not supplied, this figures out what value of volLocalDims gives the most even spread of data, and allocates the arrays.
        In both cases, the volOffsets are also calculated and returned.
        """
        #MPI properties
        rank = comm.Get_rank()
        size = comm.Get_size()
        #Spread arguments from process 0
        volDims = comm.bcast(volDims, root=0)
        volLocalDims = comm.bcast(volLocalDims, root=0)
        #If volLocalDims wasn't specified, create it.
        if (volLocalDims == None):
                #Initial guess
                volLocalDims = volDims.copy()
                volLocalDims[0] = (volDims[0]+size-1) // size
                numSubVols = (volDims[1:4]/volLocalDims[1:4]).prod() * (volDims[0]+volLocalDims[0]-1) // volLocalDims[0]
                #Chop by longest dimension, as long as it improves the coverage of the CPUs
                while (2*numSubVols <= size):
                        volLocalDims[volLocalDims.argmax()] /= 2
                        numSubVols = (volDims[1:4]/volLocalDims[1:4]).prod() * (volDims[0]+volLocalDims[0]-1) // volLocalDims[0]
        #Check number of sub-problems is valid
        numSubVols = (volDims[1:4]/volLocalDims[1:4]).prod() * (volDims[0]+volLocalDims[0]-1) // volLocalDims[0]
        if (numSubVols > size):
                print("local dims, total dims, subvolumes, and size:", volLocalDims, volDims, numSubVols, size)
                sys.exit("Complete coverage of volume space impossible with specified dimensions")
        #Create volume data
        volOffset = np.zeros(4, dtype=INT_TYPE)
        print("CPU:", rank, "volLocalDims:", volLocalDims, "volDims:", volDims)
        if (rank < numSubVols):
                temp = np.indices((volDims+volLocalDims-1) // volLocalDims)
                temp = temp.reshape([4, temp.size/4])
                volOffset[:] = temp[:,rank] * volLocalDims[:]
                #Account for the fact that the final volume array may need to be a different size along the time axis (e.g. 15 timeframes across 8 CPUs)
                if (rank == numSubVols-1):
                        volLocalDims[0] = volDims[0] - (numSubVols-1) * volLocalDims[0]
                vol_V = np.zeros(volLocalDims, dtype=FLOAT_TYPE)
        else:
                vol_V = None
        #Synchronise before return
        comm.Barrier()
        return vol_V, volOffset

