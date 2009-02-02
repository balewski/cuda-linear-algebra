""" 

Executing shell commande depending on params

"""
import os
from optparse import OptionParser

# -- command line options
usage = "usage: %prog [options]"
parser = OptionParser(usage)

parser.add_option("-r","--run", dest="TASK",type="int",  default = 0)

(options,args) = parser.parse_args()


TASK=options.TASK

print "possible: 1=run C-code,  2=genData, 3=genSolve_lapack"
print ">>> task = %d" % TASK
 
size=1
for i in range(1,15):
    size*=2
#    if i<14 : continue
    if TASK==1:
        for meth in range(4,0,-1):
           logName="pass3/Log_m%d" %meth+"_s%d" %size 
           cmd_string = "./janConjGrad %d "% size+" %d "% meth+ " 1  >&" +logName 
           print "echo ========= "+ cmd_string
           print  cmd_string 
           #os.system(cmd_string)
           cmd_string = "grep \# " + logName
           print  cmd_string 
           # os.system(cmd_string)
    if TASK==2:
        cmd_string = "python genConjData.py -d  %d" % size   
        print cmd_string
        #if i>12 :
        #os.system(cmd_string)
    if TASK==3:
        cmd_string = "python genSolvLinEq.py -d  %d" % size   
        print cmd_string
        if i>12 :
            os.system(cmd_string) 




#
