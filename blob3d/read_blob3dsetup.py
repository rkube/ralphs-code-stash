#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np


def read_blob3dsetup(path):
    try:
        sfile = open('%s/setup.txt' % path, 'r')
    except:
        print 'Unable to open file %s/setup.txt' % path
        return
        
        
    lines = sfile.readlines()
    
    setup = {}
    
    for line in lines:
        seps = line.rstrip().partition('=')
        
        # All values in the config file are either float or strings.
        # Try to cast each value to a string first. If this fails,
        # it must be a string
        try:   
            setup[seps[0]] = float(seps[-1])
        except ValueError:
            setup[seps[0]] = seps[-1]
 
    return setup
