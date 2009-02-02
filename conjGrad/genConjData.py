#!/usr/bin/env python

""" 
 Data generator

"""
import numpy
from numpy import *

from optparse import OptionParser
from math import sqrt

print "START A.X=B matrix generator ..."

# -- command line options
usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("-d", "--dim",
                  dest="dimX", 
                  type="int",
                  default = 2)

(options,args) = parser.parse_args()

#################################
#  MAIN
#################################

nSol=1 # of problems to solve: {A,X,B} * nSol

# define (square) matrix size
dimX=options.dimX
print "dimX=%d" % dimX
# create random square a,b matrices
# symmetrize it to fullfil conjGrad requirements

#pathOut="../../bin/linux/release/dataC/"
pathOut="dataA/"

print "save matrices ...." 
coreN=pathOut+"conjData_%d.bin" % dimX 
print "=",coreN,"= nSol=" , nSol 
fd=open(coreN,'w+')
for k in range(nSol):
    A = numpy.random.randn(dimX,dimX).astype(numpy.float32)
    A=dot(A.T,A)
    A.tofile(fd)
    X0 = numpy.random.randn(dimX).astype(numpy.float32) #solution
    B=dot(A,X0) # constant vector
    B.tofile(fd)
    X0.tofile(fd)

if (dimX<5) : # print them if small
    print "A=", A
    print "B=", B
    print "X0=", X0



