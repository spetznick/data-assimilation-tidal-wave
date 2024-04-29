#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:45:10 2018

@author: verlaanm
"""

import dateutil

def read_series(filename):
    infile=open(filename,'r')
    times=[]
    values=[]
    for line in infile:
        #print(">%s<%d"%(line,len(line)))
        if line.startswith("#") or len(line)<=1:
            continue
        parts=line.split()
        times.append(dateutil.parser.parse(parts[0]))
        values.append(float(parts[1]))
    infile.close()
    return (times,values)
      
#
# Tests
#
if __name__ == "__main__":
    (times,values)=read_series('tide_vlissingen.txt');
    assert len(times)==len(values) #should be matching pairs
    assert len(times)==289
    assert abs(values[0]-0.07)<0.01
