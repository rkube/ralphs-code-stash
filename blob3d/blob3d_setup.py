#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

import numpy as np
from collections import OrderedDict

class blob3d_setup:
    def __init__(self, path = None):
    
        # List of keys needed for blob3d setup file
        self.key_list = ['nx', 'ny', 'nz', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', \
            't_start', 't_end', 'dt', 'outputinterval', 'diaginterval', 'kappa_perp', \
            'kappa_par', 'mu', 'theta', 'gaussamp', 'gaussxwidth', 'gaussywidth', 'gausszwidth', \
            'gaussxcenter', 'gaussycenter', 'gausszcenter', 'gaussbglevel', \
            'bc_left_n', 'bc_right_n', 'value_bc_left_n', 'value_bc_right_n', \
            'bc_left_omega', 'bc_right_omega', 'value_bc_left_omega', 'value_bc_right_omega'] 
        # Keys that are of type float
        self.key_float_val = ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', \
            't_start', 't_end', 'dt', \
            'kappa_perp', 'kappa_par', 'mu', 'theta', 'gaussamp', 'gaussxwidth', 'gaussywidth', 'gausszwidth', \
            'gaussxcenter', 'gaussycenter', 'gausszcenter', 'gaussbglevel', \
            'value_bc_left_n', 'value_bc_right_n', 'value_bc_left_omega', 'value_bc_right_omega']  
        # Keys that are of type int
        self.key_int_val = ['nx', 'ny', 'nz', 'outputinterval', 'diaginterval']
        # Keys that are of type string  
        self.key_string_val = [ 'bc_left_n', 'bc_right_n', 'bc_left_omega', 'bc_right_omega' ]
        
    
        # Read parameters from file
        try:
            setupfile = open('%s/setup.txt' % path, 'r')            
            lines = setupfile.readlines()
            self.params = OrderedDict()

            for line in lines:
                seps = line.rstrip().partition('=')
                
                # All values in the config file are either float or strings.
                # Try to cast each value to a string first. If this fails,
                # it must be a string
                try:   
                    #setup[seps[0]] = float(seps[-1])
                    self.params[seps[0]] = float(seps[-1])
                except ValueError:
                    #setup[seps[0]] = seps[-1]
                    self.params[seps[0]] = seps[-1]
            
        # Or, if the file does not exist, create a standard setup
        except:
            print 'Unable to open file %s/setup.txt for reading' % path
            print 'Creating standard blob3d setups'
            
            self.params = OrderedDict( \
            [ ('nx', 512), ('ny', 512), ('nz', 64), \
              ('xmin', -30.0), ('xmax', +30.0), ('ymin', -30.0), ('ymax', +30.0), ('zmin', -10.0), ('zmax', +10.0), \
              ('t_start', 0), ('t_end', 10.0), ('dt', 0.001), ('outputinterval', 100), ('diaginterval', 10), \
              ('kappa_perp', 0.001), ('kappa_par', 0.001), ('mu', 0.001), ('theta', 1.0), \
              ('gaussamp', 1.0), ('gaussxwidth', 1.0), ('gaussywidth', 1.0), ('gausszwidth', 1.0), \
              ('gaussxcenter', 0.0), ('gaussycenter', 0.0), ('gausszcenter', 0.0), ('gaussbglevel', 1.0), \
              ('bc_left_n', 'neumann'), ('bc_right_n', 'neumann'), ('value_bc_left_n', 0.0), ('value_bc_right_n', 0.0), \
              ('bc_left_omega', 'neumann'), ('bc_right_omega', 'neumann'), ('value_bc_left_omega', 0.0), ('value_bc_right_omega', 0.0)])


#            self.params =  { 'nx' : 512, 'ny' : 512, 'nz': 64,\
#                            'xmin': -30.0, 'ymin': -30.0, 'zmin': -10.0, \
#                            'xmax': +30.0, 'ymax' : +30.0, 'zmax': +10.0, \
#                            't_start': 0, 't_end': 10.0, 'dt': 0.001, \
#                            'outputinterval': 100, 'diaginterval': 10, \
#                            'kappa_perp': 0.001, 'kappa_par': 0.001, 'mu': 0.001, 'theta': 1.0, \
#                            'gaussamp': 1.0, 'gaussxwidth': 1.0, 'gaussywidth': 1.0, 'gausszwidth': 1.0, \
#                            'gaussxcenter': 0.0, 'gaussycenter': 0.0, 'gausszcenter': 0.0, 'gaussbglevel': 1.0, \
#                            'bc_left_n': 'neumann', 'bc_right_n': 'neumann', 'value_bc_left_n': 0.0, 'value_bc_right_n': 0.0, \
#                            'bc_left_omega': 'neumann', 'bc_right_omega': 'neumann', 'value_bc_left_omega': 0.0, 'value_bc_right_omega': 0.0 } 

        # Either way, ensure that all parameters in the setup structure have valid values
        assert any(x in self.params for x in self.key_list)
        
    
    def get_dict(self):
        return self.params

        
    def get_key(self, key):
        try:
            return self.params[key]
        except KeyError:
            print 'No such key: %s' % key

    def __getitem__(self, key):
        try:
            return self.params[key]
        except KeyError:
            print 'No such key: %s' % key
            


    def set_key(self, key, value):
        # Ensure value is of right type for key
        if key in self.key_string_val:
            assert type(value) is str
        elif key in self.key_int_val:
            assert type(value) is int
        elif key in self.key_float_val:
            assert type(value) is float
            
        self.params[key] = value


    def __setitem__(self, key, value):
        # Ensure value is of right type for th ekey
        if key in self.key_string_val:
            assert type(value) is str
        elif key in self.key_int_val:
            assert type(value) is int
        elif key in self.key_float_val:
            assert type(value) is float

        self.params[key] = value 


 
    def read_blob3dsetup(self, path):
        try:
            sfile = open('%s/setup.txt' % path, 'r')
        except:
            print 'Unable to open file %s/setup.txt' % path
            return
            
            
        lines = sfile.readlines()
        
        params = {}
        
        for line in lines:
            seps = line.rstrip().partition('=')
            
            # All values in the config file are either float or strings.
            # Try to cast each value to a string first. If this fails,
            # it must be a string
            try:   
                params[seps[0]] = float(seps[-1])
            except ValueError:
                params[seps[0]] = seps[-1]
     
        return params
    
        
    def write(self, path):
        try:
            setupfile = open('%s/setup.txt' % path, 'w')
        except:
            print 'Unable to open file %s/setup.txt for writing' % path
            return
        
        for key in self.params:
            if key in self.key_float_val:
                setupfile.write('%s=%.3f\n' % ( key, self.params[key] ) )
            elif key in self.key_int_val:
                setupfile.write('%s=%d\n' % ( key, int(self.params[key]) ) )
            elif key in self.key_string_val:
                setupfile.write('%s=%s\n' % ( key, self.params[key] ) )
                
        setupfile.close()
