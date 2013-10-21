#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

from os.path import join


class strval_pair:
    """Type - value pair"""
    def __init__(self, typeval=None, val=None):
        # Initialize with type and value
        self.typeval = typeval
        self.val = val
        print 'new strval, typeval:', typeval, 'val=', val

    def update(self, newval):
        """Update, with type conversion"""
        if isinstance(newval, self.typeval):
            #Same type, no conversion necessary
            self.val = newval
        else:
            # Explicit type conversion
            try:
                self.val = self.typeval(newval)
            except:
                raise TypeError('%s is not of type %s' % (type(newval),
                                                        self.typeval))

    def gettype(self):
        return self.typeval

    def __str__(self):
        """string representation as to be put in output file"""
        if self.typeval == str:
            return '%s' % self.val
        elif self.typeval == int:
            return '%d' % self.val
        elif self.typeval == float:
            return '%f' % self.val
        elif self.typeval == list:
            return self.typeval.__str__()


class input:
    """ Class interface to 2dads input.ini"""

    def __init__(self, simdir=None):

        self.keys = {'runnr': strval_pair(int, 1),
                     'xleft':  strval_pair(float, -10.0),
                     'xright': strval_pair(float, +10.0),
                     'ylow': strval_pair(float, -10.0),
                     'yup': strval_pair(float, +10.0),
                     'Nx': strval_pair(int, 64),
                     'My': strval_pair(int, 64),
                     'scheme': strval_pair(str, 'ss4'),
                     'tlevs': strval_pair(int, 4),
                     'deltat': strval_pair(float, 1e-3),
                     'tend': strval_pair(float, 1e1),
                     'tdiag': strval_pair(float, 1e-2),
                     'tout': strval_pair(float, 1e-2),
                     'do_particle_tracking': strval_pair(int, 0),
                     'nprobes': strval_pair(int, 8),
                     'theta_rhs': strval_pair(str, 'theta_rhs_log'),
                     'omega_rhs': strval_pair(str, 'omega_rhs'),
                     'log_theta': strval_pair(int, 1),
                     'init_function': strval_pair(str, 'theta_gaussian'),
                     'initial_conditions': strval_pair(list, [1, 1, 0, 0, 1, 1]),
                     'model_params': strval_pair(list, [1, 1e-3, 1e-3, 0, 0, 1]),
                     'output': strval_pair(list, ['theta', 'omega', 'strmf']),
                     'diagnostics': strval_pair(list, ['energy', 'blobs', 'probes']),
                     'nthreads': strval_pair(int, 1)}

        if simdir is not None:
            # Create dictionary from lines in file
            filename = join(simdir, 'input.ini')
            print 'Reading input.ini from %s' % (filename)
            with open(filename) as infile:
                for line in infile.readlines():
                    self.update_dict(self.keys, line)
            print 'Done parsing input'
        else:
            print 'No simdir passed'

    def update_dict(self, keys, line):
        update_key = line[:line.index('=')].strip()
        if update_key not in self.keys.keys():
            raise NameError('%s it not a valid key' % (update_key))
        update_val = line[line.index('=') + 1:].strip()

        # Cast string to correct type
        self.keys[update_key].update(update_val)


    def __getitem__(self, key):
        return self.keys[key]

# End of file twodasd.py
