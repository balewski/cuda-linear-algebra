#!/usr/bin/python

import sys
from pylab import *

#infname = sys.argv[1]

machineN=' kraken1 + Tesla C1060 w/ 4GB'
infname='resultTesla.csv'


flag1=4 # 4=total, 7=# of cycles


print infname
lines = [line[:-1].split(',') for line in open(infname).readlines()[1:]]

# now I init hash-table== dictionary, now is empty
data = {}
nLine=0
for line in lines:
   nLine+=1
   if nLine>5 :   
      print line
   impl = line[0]
   if impl not in data:
       data[impl] = {'x':[],'y':[]} 
   x, y = int(line[1]), float(line[flag1])
   data[impl]['x'] += [x] # force x to be a list element instead of float
   data[impl]['y'] += [y]

print " starting graphics ..."
figure()

legend_l = [] # this is empty list 
styl='-'
k=0
stylA=['-D','-d','-x','-o','-v']
print stylA
for impl in data: # iterating over dictionary, index runs over the key of dict.
   styl=stylA[k]
   k=k+1

   print impl," style="+ styl+"=",k
   
   x, y = data[impl]['x'], data[impl]['y']
   plot(x, y, styl)
   legend_l += [impl]


#quit() #---------------
ax = gca()
ax.set_xscale('log')
ax.set_yscale('log')
if flag1==7 :
   legend(legend_l)
else :
   legend(legend_l, loc='upper left')

axis((1,4e4,5e-4,9e7))

xlabel('dimension of array A ')
ylabel('time (ms)')
if  flag1==7 :
   ylabel('# of cycles ')
grid(True)

if flag1==4 :
   title("Total computing time,"+machineN)

if flag1==7 :
   title("number of repeated fitting (the same for all methods)")
show()
print data
