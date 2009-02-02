#!/usr/bin/env python

""" 
 Data generator

"""
import numpy
from numpy import *

from optparse import OptionParser
from math import sqrt
import timeit 

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

# define (square) matrix size
dimX=options.dimX
print "dimX=%d" % dimX
# create random square a,b matrices
# symmetrize it to fullfil conjGrad requirements

A = numpy.random.randn(dimX,dimX).astype(numpy.float32)
A=dot(A.T,A)
X0 = numpy.random.randn(dimX).astype(numpy.float32) #solution
B=dot(A,X0) # constant vector

if (dimX<5) : # print them if small
    print "A=", A
    print "B=", B
    print "X0=", X0
print "A,B generated ,  linalg.lstsq(A,B)..."
nCycle=1
if dimX>16:
    nCycle += 2048*2048/dimX/dimX;
else:
    nCycle=10000;
 
t1=timeit.time.time() 
# execute the code you want to time here
for i in range(nCycle):
    X,R,rnk,s=linalg.lstsq(A,B)
t2=timeit.time.time() 
elapsedT=t2-t1 
tBtot=elapsedT*1000./nCycle

# does not work ???
#t = timeit.Timer( 'X,R,rnk,s=linalg.lstsq(A,B)', 'from  __main__  import linalg')
#t.timeit()

print "Execution took %f seconds" % elapsedT
print "tBtot=%f"% tBtot
# compute epsilon
dX=X-X0
eps=sqrt(dot(dX,dX))
print " eps(X0-X)=%.3f" %eps

print "method,  size, dum ,Tmem(ms),Tmath(ms),lastEps,dum,nCycle,dum, dum"
print "#CPU+lapack, %d, " %dimX + "0, 0.0,   %.3f," %  tBtot + "  %.3f, 0, " %eps + " %d, kraken1, NA" %nCycle


"""
linalg.lstsq(a, b, rcond=-1)

Compute least-squares solution to equation :math:`a x = b`

Returns
    -------
    x : array, shape (N,) or (N, K) depending on shape of b
        Least-squares solution
    residues : array, shape () or (1,) or (K,)
        Sums of residues, squared 2-norm for each column in :math:`b - a x`
        If rank of matrix a is < N or > M this is an empty array.
        If b was 1-d, this is an (1,) shape array, otherwise the shape is (K,)
    rank : integer
        Rank of matrix a
    s : array, shape (min(M,N),)
        Singular values of a
"""
