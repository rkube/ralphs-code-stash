#!/usr/bin/env python
#-*- Encoding: UTF-8 -*-

from os.path import join

# Dictionary, restricted to a few items
# Gives string representation of dictionary values
# Does typecasting depending on string
class strval_pair:
    """Type - value pair"""
    def __init__(self, typeval=None, val=None):
        # Initialize with type and value
        self.typeval = typeval
        self.val = val

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

    def getval(self):
        return self.val

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


class input2d:
    """ Class interface to 2dads input.ini"""

    def __init__(self, simdir=None, fname=None):

        self.keys = {'runnr': 1,
                     'xleft': -10.0,
                     'xright': 10.0,
                     'ylow': -10.0,
                     'yup': 10.0,
                     'Nx': 64,
                     'My': 64,
                     'scheme': 'ss4',
                     'Lx': 0.0,
                     'Ly': 0.0,
                     'deltax': 0.0,
                     'deltay': 0.0,
                     'tlevs': 4,
                     'deltat': 1e-3,
                     'tend': 10.0,
                     'tdiag': 0.01,
                     'tout': 0.1,
                     'log_theta': 1,
                     'do_particle_tracking': 0,
                     'nprobes': 8,
                     'theta_rhs': 'theta_rhs_log',
                     'omega_rhs': 'omega_rhs_ic',
                     'strmf_solver': 'spectral',
                     'init_function': 'theta_gaussian',
                     'initial_conditions': [1, 1, 0, 0, 1, 1],
                     'model_params':  [1, 1e-3, 1e-3, 0, 0, 1],
                     'output': ['theta', 'omega', 'strmf'],
                     'diagnostics': ['energy', 'blobs', 'probes'],
                     'nthreads': 1}

        self.value_type = {"runnr": int,
                           "xleft": float,
                           'xright': float,
                           'ylow': float,
                           'yup': float,
                           'Nx': int,
                           'My': int,
                           'scheme': str,
                           'Lx': float,
                           'Ly': float,
                           'deltax': float,
                           'deltay': float,
                           'tlevs': int,
                           'deltat': float,
                           'tend': float,
                           'tdiag': float,
                           'tout': float,
                           'log_theta': bool,
                           'do_particle_tracking': bool,
                           'nprobes': int,
                           'theta_rhs': str,
                           'omega_rhs': str,
                           'strmf_solver': str,
                           'init_function': str,
                           'initial_conditions': list,
                           'model_params':  list,
                           'output': list,
                           'diagnostics': list,
                           'nthreads': int}

        if simdir is not None:
            # Create dictionary from lines in file
            if fname is None:
                filename = join(simdir, 'input.ini')
            else:
                filename = join(simdir, fname)
            print 'Reading input.ini from %s' % (filename)
            # Populate dictionary from input file
            with open(filename) as infile:
                for line in infile.readlines():
                    # Convert  line from input file to correct data type
                    self.update_dict(self.keys, line)
            print 'Done parsing input'
        else:
            print 'simdir not set. Creating input with default values'
        # If simdir is not set, we have created a dictionary with default
        # values
        self.keys['Lx'] = (self.keys['xright'] - self.keys['xleft'])
        self.keys['Ly'] = (self.keys['yup'] - self.keys['ylow'])
        self.keys['deltax'] = (self.keys['Lx'] / self.keys['Nx'])
        self.keys['deltay'] = (self.keys['Ly'] / self.keys['My'])


    def update_dict(self, keys, line):
        # Split line in two at first occurance of '='
        update_key = line[:line.index('=')].strip()
        val_line = line[line.index('=') + 1:]

        if update_key not in self.keys.keys():
            print '%s is not a valid key... skipping' % (update_key)
            return
            #raise NameError('%s it not a valid key' % (update_key))

        if update_key in ['initial_conditions', 'model_params']:
            # keys initial_conditions and model_params are
            # vector valued. Strip the line after the '=' and
            # create a list with all occuring values

            # Strip trailing whitespaces, split at ' '
            update_val = [float(s) for s in val_line.strip().split(' ')]
        elif update_key in ['scheme', 'theta_rhs', 'omega_rhs', 'init_function']:
            update_val = val_line.strip()
        elif update_key in ['output', 'diagnostics', 'strmf_solver']:
            update_val = val_line.strip().split(' ')
        else:
            # All other keys are scalar valued.
            update_val = float(val_line.strip())

        # Cast string to correct type
        #self.keys[update_key].update(update_val)
        #print 'key: ' + update_key + ', type(val)',  type(update_val)
        self.keys[update_key] = update_val

    def to_file(self, filename):
        with open(filename, 'w') as outfile:
            for k in self.keys.iterkeys():
                if k in ['Lx', 'Ly', 'deltax', 'deltay']:
                    continue
                line = ''
                if self.value_type[k] is int:
                    line = '%s = %d' % (k, self.keys[k])
                elif self.value_type[k] is float:
                    line = '%s = %f' % (k, self.keys[k])
                elif self.value_type[k] is bool:
                    line = '%s = %d' % (k, self.keys[k])
                elif self.value_type[k] is str:
                    line = '%s = %s' % (k, self.keys[k])
                elif self.value_type[k] is list:
                    if k in ['output', 'diagnostics']:
                        line = '%s = %s' % (k,
                                            ' '.join(self.keys[k]))
                    elif k in ['model_params', 'initial_conditions']:
                        line = '%s = ' % (k)
                        for val in self.keys[k]:
                            line += '%7.5f ' % val
                else:
                    raise ValueError('Could not generate line for key %s' % k)
                line += '\n'
                outfile.write(line)

    # Dictionary member functions
    def __getitem__(self, key):
        return self.keys[key]

    def __setitem__(self, update_key, update_val):
        """
        Update update_key with update_val. Allow update only, if update_val
        is of correct data type for update_key. Throws ValueError otherwise.
        """

        # Check if key is known
        if update_key not in self.keys.keys():
            print '%s is not a valid key..' % (update_key)
            return
        # These keys are not to be updated. Computed after updating xl, xr, yl,
        # yu, Nx, My (see, below)
        if update_key in ['Lx', 'Ly', 'deltax', 'deltay']:
            err_msg = 'Updating Lx, Ly, deltax, deltay is ambiguous.'
            err_msg += 'Update xleft/ylow, xright/yup , Nx, My instead'
            raise TypeError(err_msg)

        # When updating, cast to desired types
        elif update_key in ['runnr', 'xleft', 'xright', 'ylow', 'yup', 'Nx', 'My',
                'scheme', 'tlevs', 'deltat', 'tend', 'tdiag', 'tout', 'log_theta',
                'do_particle_tracking', 'nprobes', 'theta_rhs', 'omega_rhs',
                'strmf_solver', 'init_function', 'nthreads']:
            try:
                self.keys[update_key] = self.value_type[update_key](update_val)
            except TypeError:
                err_msg = 'Failed to cast type %s to type %s' % (type(update_val),
                        self.value_type[update_key])
                raise TypeError(err_msg)

        # When updating output, cast to list of strings
        elif update_key is 'output':
                try:
                    dummy = [str(u) for u in update_val]
                except TypeError:
                    err_msg = 'Failed to cast all elements of ', update_val, ' to str'
                    raise TypeError(err_msg)
                self.keys[update_key] = update_val

        # When updating diagnostics, check if update_key is a list of strings:
        elif update_key is 'diagnostics':
            try:
                dummy = [str(u) for u in update_val]
            except TypeError:
                err_msg = 'Failed to cast all elements of ', update_val, ' to str'
                raise TypeError(err_msg)
            self.keys[update_key] = update_val

        # When updating initial_conditions, check if update_key is a list of
        # floats
        elif update_key is 'initial_conditions':
            try:
                dummy = [str(u) for u in update_val]
            except TypeError:
                err_msg = 'Failed to cast all elements of ', update_val, ' to str'
                raise TypeError(err_msg)
            self.keys[update_key] = update_val

        # When updating model_params, check if update_key is a list of
        # floats
        elif update_key is 'model_params':
            try:
                dummy = [str(u) for u in update_val]
            except TypeError:
                err_msg = 'Failed to cast all elements of ', update_val, ' to str'
                raise TypeError(err_msg)
            self.keys[update_key] = update_val



        elif update_key in ['xright', 'xleft']:
            self.keys['Lx'] = (self.keys['xright'] - self.keys['xleft'])
            self.keys['deltax'] = (self.keys['Lx'] / self.keys['Nx'])

        elif update_key in ['ylow', 'yup']:
            self.keys['Ly'] = (self.keys['yup'] - self.keys['ylow'])
            self.keys['deltay'] = (self.keys['Ly'] / self.keys['My'])


        if update_key is 'Nx':
            self.keys['deltax'] = self.keys['Lx'] / self.keys['Nx']

        elif update_key is 'My':
            self.keys['deltay'] = self.keys['Ly'] / self.keys['My']


    def items(self):
        return self.keys.items()


    def has_key(key):
        return self.keys.has_key(key)


    def values(self):
        return self.keys.values()


    def iteritems():
        return self.keys.iteritems()


    def iterkeys(self):
        return self.keys.iterkeys()


    def itervalues(self):
        return self.keys.itervalues()


    def keys(self):
        return self.keys.keys()


    def pop(self):
        self.keys.pop()


    def viewitems(self):
        return self.keys.viewitems()


    def viewkeys(self):
        return self.keys.viewkeys()


    def viewvalues(self):
        return self.keys.viewvalues()


class input2d_3:
    """ Class interface to 2dads input.ini for three-field version"""

    def __init__(self, simdir=None, fname=None):

        self.keys = {'runnr': 1,
                     'xleft': -10.0,
                     'xright': 10.0,
                     'ylow': -10.0,
                     'yup': 10.0,
                     'Nx': 64,
                     'My': 64,
                     'scheme': 'ss4',
                     'Lx': 0.0,
                     'Ly': 0.0,
                     'deltax': 0.0,
                     'deltay': 0.0,
                     'tlevs': 4,
                     'deltat': 1e-3,
                     'tend': 10.0,
                     'tdiag': 0.01,
                     'tout': 0.1,
                     'log_theta': 1,
                     'log_tau': 1,
                     'do_particle_tracking': 0,
                     'nprobes': 8,
                     'theta_rhs': 'theta_rhs_log',
                     'omega_rhs': 'omega_rhs_ic',
                     'tau_rhs': 'tau_rhs_null',
                     'strmf_solver': 'spectral',
                     'init_function_theta': 'gaussian',
                     'initial_conditions_theta': [1, 1, 0, 0, 1, 1],
                     'init_function_tau': '_gaussian',
                     'initial_conditions_tau': [1, 1, 0, 0, 1, 1],
                     'init_function_omega': 'constant',
                     'initial_conditions_omega': [0],
                     'model_params':  [1, 1e-3, 1e-3, 0, 0, 1],
                     'output': ['theta', 'omega', 'strmf', 'tau'],
                     'diagnostics': ['energy', 'blobs', 'probes'],
                     'nthreads': 1}

        self.value_type = {"runnr": int,
                           "xleft": float,
                           'xright': float,
                           'ylow': float,
                           'yup': float,
                           'Nx': int,
                           'My': int,
                           'scheme': str,
                           'Lx': float,
                           'Ly': float,
                           'deltax': float,
                           'deltay': float,
                           'tlevs': int,
                           'deltat': float,
                           'tend': float,
                           'tdiag': float,
                           'tout': float,
                           'log_theta': bool,
                           'log_tau': bool,
                           'do_particle_tracking': bool,
                           'nprobes': int,
                           'theta_rhs': str,
                           'omega_rhs': str,
                           'tau_rhs': str,
                           'strmf_solver': str,
                           'init_function_theta': str,
                           'initial_conditions_theta': list,
                           'init_function_tau': str,
                           'initial_conditions_tau': list,
                           'init_function_omega': str,
                           'initial_conditions_omega': list,
                           'model_params':  list,
                           'output': list,
                           'diagnostics': list,
                           'nthreads': int}

        if simdir is not None:
            # Create dictionary from lines in file
            if fname is None:
                filename = join(simdir, 'input.ini')
            else:
                filename = join(simdir, fname)
            print 'Reading input.ini from %s' % (filename)
            # Populate dictionary from input file
            with open(filename) as infile:
                for line in infile.readlines():
                    # Convert  line from input file to correct data type
                    self.update_dict(self.keys, line)
            print 'Done parsing input'
        else:
            print 'simdir not set. Creating input with default values'
        # If simdir is not set, we have created a dictionary with default
        # values
        self.keys['Lx'] = (self.keys['xright'] - self.keys['xleft'])
        self.keys['Ly'] = (self.keys['yup'] - self.keys['ylow'])
        self.keys['deltax'] = (self.keys['Lx'] / self.keys['Nx'])
        self.keys['deltay'] = (self.keys['Ly'] / self.keys['My'])


    def update_dict(self, keys, line):
        # Split line in two at first occurance of '='
        update_key = line[:line.index('=')].strip()
        val_line = line[line.index('=') + 1:]

        if update_key not in self.keys.keys():
            print '%s is not a valid key... skipping' % (update_key)
            return
            #raise NameError('%s it not a valid key' % (update_key))

        if update_key in ['initial_conditions_theta',
                          'initial_conditions_tau',
                          'initial_conditions_omega',
                          'model_params']:
            # keys initial_conditions and model_params are
            # vector valued. Strip the line after the '=' and
            # create a list with all occuring values

            # Strip trailing whitespaces, split at ' '
            update_val = [float(s) for s in val_line.strip().split(' ')]
        elif update_key in ['scheme', 'theta_rhs', 'omega_rhs', 'tau_rhs',
                            'init_function_theta', 'init_function_tau', 'init_function_omega']:
            update_val = val_line.strip()
        elif update_key in ['output', 'diagnostics', 'strmf_solver']:
            update_val = val_line.strip().split(' ')
        else:
            # All other keys are scalar valued.
            update_val = float(val_line.strip())

        # Cast string to correct type
        #self.keys[update_key].update(update_val)
        #print 'key: ' + update_key + ', type(val)',  type(update_val)
        self.keys[update_key] = update_val

    def to_file(self, filename):
        print '------called to_file()'
        with open(filename, 'w') as outfile:
            for k in self.keys.iterkeys():
                if k in ['Lx', 'Ly', 'deltax', 'deltay']:
                    continue
                line = ''
                if self.value_type[k] is int:
                    line = '%s = %d' % (k, self.keys[k])
                elif self.value_type[k] is float:
                    line = '%s = %f' % (k, self.keys[k])
                elif self.value_type[k] is bool:
                    line = '%s = %d' % (k, self.keys[k])
                elif self.value_type[k] is str:
                    line = '%s = %s' % (k, self.keys[k])
                elif self.value_type[k] is list:
                    if k in ['output', 'diagnostics']:
                        line = '%s = %s' % (k,
                                            ' '.join(self.keys[k]))
                    elif k in ['model_params', 'initial_conditions_theta',
                               'initial_conditions_omega', 'initial_conditions_tau']:
                        line = '%s = ' % (k)
                        for val in self.keys[k]:
                            line += '%7.5f ' % val
                else:
                    raise ValueError('Could not generate line for key %s' % k)
                line += '\n'
                outfile.write(line)

    # Dictionary member functions
    def __getitem__(self, key):
        return self.keys[key]

    def __setitem__(self, update_key, update_val):
        """
        Update update_key with update_val. Allow update only, if update_val
        is of correct data type for update_key. Throws ValueError otherwise.
        """

        # Check if key is known
        if update_key not in self.keys.keys():
            print '%s is not a valid key..' % (update_key)
            return
        # These keys are not to be updated. Computed after updating xl, xr, yl,
        # yu, Nx, My (see, below)
        if update_key in ['Lx', 'Ly', 'deltax', 'deltay']:
            err_msg = 'Updating Lx, Ly, deltax, deltay is ambiguous.'
            err_msg += 'Update xleft/ylow, xright/yup , Nx, My instead'
            raise TypeError(err_msg)

        # When updating, cast to desired types
        elif update_key in ['runnr', 'xleft', 'xright', 'ylow', 'yup', 'Nx', 'My',
                            'scheme', 'tlevs', 'deltat', 'tend', 'tdiag', 'tout',
                            'log_theta', 'do_particle_tracking', 'nprobes', 'theta_rhs',
                            'omega_rhs', 'tau_rhs', 'strmf_solver', 'init_function_theta',
                            'init_function_tau', 'init_function_omega', 'nthreads']:
            try:
                self.keys[update_key] = self.value_type[update_key](update_val)
            except TypeError:
                err_msg = 'Failed to cast type %s to type %s' % (type(update_val),
                        self.value_type[update_key])
                raise TypeError(err_msg)

        # When updating output, cast to list of strings
        elif update_key is 'output':
                try:
                    dummy = [str(u) for u in update_val]
                except TypeError:
                    err_msg = 'Failed to cast all elements of ', update_val, ' to str'
                    raise TypeError(err_msg)
                self.keys[update_key] = update_val

        # When updating diagnostics, check if update_key is a list of strings:
        elif update_key is 'diagnostics':
            try:
                dummy = [str(u) for u in update_val]
            except TypeError:
                err_msg = 'Failed to cast all elements of ', update_val, ' to str'
                raise TypeError(err_msg)
            self.keys[update_key] = update_val

        # When updating initial_conditions, check if update_key is a list of
        # floats
        elif update_key in ['initial_conditions_theta', 'initial_conditions_tau', 'initial_conditions_omega']:
            try:
                dummy = [str(u) for u in update_val]
            except TypeError:
                err_msg = 'Failed to cast all elements of ', update_val, ' to str'
                raise TypeError(err_msg)
            self.keys[update_key] = update_val

        # When updating model_params, check if update_key is a list of
        # floats
        elif update_key is 'model_params':
            try:
                dummy = [str(u) for u in update_val]
            except TypeError:
                err_msg = 'Failed to cast all elements of ', update_val, ' to str'
                raise TypeError(err_msg)
            self.keys[update_key] = update_val



        elif update_key in ['xright', 'xleft']:
            self.keys['Lx'] = (self.keys['xright'] - self.keys['xleft'])
            self.keys['deltax'] = (self.keys['Lx'] / self.keys['Nx'])

        elif update_key in ['ylow', 'yup']:
            self.keys['Ly'] = (self.keys['yup'] - self.keys['ylow'])
            self.keys['deltay'] = (self.keys['Ly'] / self.keys['My'])


        if update_key is 'Nx':
            self.keys['deltax'] = self.keys['Lx'] / self.keys['Nx']

        elif update_key is 'My':
            self.keys['deltay'] = self.keys['Ly'] / self.keys['My']


    def items(self):
        return self.keys.items()


    def has_key(key):
        return self.keys.has_key(key)


    def values(self):
        return self.keys.values()


    def iteritems():
        return self.keys.iteritems()


    def iterkeys(self):
        return self.keys.iterkeys()


    def itervalues(self):
        return self.keys.itervalues()


    def keys(self):
        return self.keys.keys()


    def pop(self):
        self.keys.pop()


    def viewitems(self):
        return self.keys.viewitems()


    def viewkeys(self):
        return self.keys.viewkeys()


    def viewvalues(self):
        return self.keys.viewvalues()


# End of file twodads.py
