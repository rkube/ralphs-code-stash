#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from phantom_helper import make_rz_array
from scipy.signal import correlate
from helper_functions import blob_in_sol

"""
Convenient scripts that look at the statistics of blob trails from a show.
Require frames from phantom camera
blobtrails from blob_tracking
"""


def statistics_blob_sol(shotnr, blobtrails, frame_info, frames = None, good_domain = None, logger = None):
    """
    Do blob statistics for blobs that travel through the SOL
    """

    np.set_printoptions(linewidth=999999)
    min_sol_px = 4

#    # Get R,z projection, grid data
    rz_array, transform_data = make_rz_array(frame_info)
    xxi, yyi = np.meshgrid( np.linspace( np.min(rz_array[:,:,0] ), np.max( rz_array[:,:,0] ),64 ), np.linspace( np.min( rz_array[:,:,1] ), np.max( rz_array[:,:,1] ),64 ) )
    xyi = np.concatenate( (xxi[:,:,np.newaxis], yyi[:,:,np.newaxis]), axis=2 )
    

    num_events = len(blobtrails)    
    failcount = 0
    blob_amps = np.zeros([num_events])
    blob_ell  = np.zeros([num_events, 2])
    blob_vel  = np.zeros([num_events, 2])
    blob_shape= np.zeros([24, 24])
    
    blobs_used_good_domain = np.zeros(num_events, dtype='bool')
    fail_list = []
    
    blob_shape_count = 0
    
    for idx, trail in enumerate(blobtrails):
    
        good_pos_idx = blob_in_sol(trail, good_domain, logger)
        
        # Skip this blob if it is not detected between LCFS and LS
        if ( good_pos_idx.sum() < min_sol_px ):
            if ( logger != None ):
                logger.debug('Blob %d/%d was not detected in the SOL, rejecting' % (idx, num_events) )
            else:
                print 'Blob %d/%d was not detected in the SOL, rejecting' % (idx, num_events)
            continue

        if ( logger != None ):
            logger.debug('Blob %d is recorded at %d/%d positions in the SOL' % (idx, np.sum(good_pos_idx), np.size(trail.get_tau()) ) )
        else:
            print 'Blob %d is recorded at %d/%d positions in the SOL' % (idx, np.sum(good_pos_idx), np.size(trail.get_tau()) )
    
        try:
            # Do not use the last position since no velocity can be computed for it.
            blob_vel[idx,:] = trail.get_velocity_com(rz_array)[good_pos_idx[:-1]].mean( axis = 0 )#[newtrail.get_frame0(),0]#.mean( axis=0 )
            blob_amps[idx]  = trail.get_amp()[good_pos_idx].mean()#[newtrail.get_frame0()]
            blob_ell[idx,0] = trail.get_ell_rad()[good_pos_idx].mean()#[[newtrail.get_frame0()]]
            blob_ell[idx,1] = trail.get_ell_pol()[good_pos_idx].mean()#[[newtrail.get_frame0()]]
            
        except AssertionError:
            fail_list.append(idx)
            failcount += 1
            continue
    
        # Collect blob shapes at time where the maximum was recorded
        try:
            blob_shape += trail.get_blob_shape(frames, position = 'MAX', frameno = np.arange(1)).mean(axis=0)
            blob_shape_count += 1
        except:
            logger.debug('Blob %d: Error adding blob shape' % idx)
            
        # The blob survived :)
        blobs_used_good_domain[idx] = 1
        
        trail.plot_trail(frames, rz_array = rz_array, xyi = xyi, plot_com = True, plot_shape = True, save_frames = True)
 
    print 'failcount = %d ' % failcount

    blob_shape /= float(blob_shape_count) 
    
    if ( logger != None ):
        logger.info('Blobs between LCFS and LS: %d' % blobs_used_good_domain.sum() )

    if ( blobs_used_good_domain.any() == 0 ):
        raise ValueError('No blobs detected between LCFS and LS')

    return blob_amps[blobs_used_good_domain], blob_ell[blobs_used_good_domain,:], \
        blob_vel[blobs_used_good_domain,:], blob_shape, blobs_used_good_domain, fail_list



def plot_blob_stat1(blob_amps, blob_ell, blob_vel, blob_shape, fail_list, shotnr, num_events, failcount, save = False, logger = None):
    """
    Plot histograms of blob amplitude, length, velocity and average shape
    """


    F = plt.figure( figsize = (12,12) )
    F.text(0.5, 0.95, 'shot #%d, %d blob events, failcount = %d' % (shotnr, num_events, failcount), ha='center')
    plt.subplot(221)
    plt.xlabel('$\\bar{I}$ / a.u.')
    plt.ylabel('Counts')
    plt.hist( blob_amps, bins = 15 )
    
    plt.subplot(222)
    plt.title('Length at normalized amplitude' )
    plt.hist( blob_ell[:, 0], bins = 15, histtype = 'step', label='radial', linewidth = 2 )
    plt.hist( blob_ell[:, 1], bins = 15, histtype = 'step', label='poloidal', linewidth = 2)
    plt.errorbar( blob_ell[:, 0].mean(), 3, xerr = blob_ell[:,0].std(), color = 'b', marker = '>', markersize = 8)
    plt.errorbar( blob_ell[:, 1].mean(), 6, xerr = blob_ell[:,1].std(), color = 'g', marker = '^', markersize = 8)
    plt.xlabel('Length / cm')
    plt.ylabel('Counts')
    plt.legend(loc = 'upper left')
    
    plt.subplot(223)
    plt.hist( blob_vel[:, 0], bins = 15, histtype = 'step', label='radial', linewidth = 2 )
    plt.hist( blob_vel[:, 1], bins = 15, histtype = 'step', label='poloidal', linewidth = 2 )
    plt.errorbar( blob_vel[:, 0].mean(), 3, xerr = blob_vel[:,0].std(), color = 'b', marker = '>', markersize = 8)
    plt.errorbar( blob_vel[:, 1].mean(), 6, xerr = blob_vel[:,1].std(), color = 'g', marker = '^', markersize = 8)
    plt.ylim(0, plt.ylim()[1])
    plt.xlabel('COM velocity / m/s')
    plt.ylabel('Counts')
    plt.legend(loc = 'upper left')
    
    
    plt.subplot(224)
    plt.title('Mean structure at $\\max{\\bar{I}}$')
    plt.contour (blob_shape, 15, colors='k')
    plt.contourf(blob_shape, 15, cmap = plt.cm.hot)
    plt.colorbar()

    try:
        logger.info('Mean radial velocity: %6f +- %6f' % ( blob_vel[:, 0].mean(), blob_vel[:,0].std() ) )
        logger.info('Mean poloidal velocity: %6f +- %6f' % ( blob_vel[:, 1].mean(), blob_vel[:,1].std() ) )
        logger.info('Mean radial length: %6f +- %6f' % ( blob_ell[:, 0].mean(), blob_ell[:,0].std() ) )
        logger.info('Mean poloidal length: %6f +- %6f' % ( blob_ell[:, 1].mean(), blob_ell[:,1].std() ) )
    except: 
        print 'Mean radial velocity: %6f +- %6f' % ( blob_vel[:, 0].mean(), blob_vel[:,0].std() ) 
        print 'Mean poloidal velocity: %6f +- %6f' % ( blob_vel[:, 1].mean(), blob_vel[:,1].std() )
        print 'Mean radial length: %6f +- %6f' % ( blob_ell[:, 0].mean(), blob_ell[:,0].std() )
        print 'Mean poloidal length: %6f +- %6f' % ( blob_ell[:, 1].mean(), blob_ell[:,1].std() )


    if ( save == True ):
        F = plt.gcf()
        filename = '%d/results/%d_thresh20_com_blobs_meaninsol_2.eps' % ( shotnr, shotnr )
        print 'Saving file %s' % filename
        F.savefig( filename )
        plt.close()
 
 
def plot_blob_stat2(blob_amps, blob_ell, blob_vel, blob_shape, fail_list, shotnr, num_events, failcount, save = False, logger = None):
    """
    Plot histograms of blob amplitude, length, velocity and average shape.
    Exclude blobs with low velocity
    """

    low_vrad_idx = np.argwhere( blob_vel[:, 0] < 100 )
    print np.size(low_vrad_idx)

    F = plt.figure( figsize = (14,7) )
    F.text(0.5, 0.95, 'shot #%d, %d blob events, failcount = %d' % (shotnr, num_events, failcount), ha='center')
    plt.subplot(131)
    plt.xlabel('$\\bar{I}$ / a.u.')
    plt.ylabel('Counts')
    plt.hist( blob_amps, bins = 15 )
    
    plt.subplot(132)
    plt.title('Length at normalized amplitude' )
    plt.hist( blob_ell[:, 0], bins = 15, histtype = 'step', label='radial', linewidth = 2 )
    plt.hist( blob_ell[:, 1], bins = 15, histtype = 'step', label='poloidal', linewidth = 2)
    plt.errorbar( blob_ell[:, 0].mean(), 3, xerr = blob_ell[:,0].std(), color = 'b', marker = '>', markersize = 8)
    plt.errorbar( blob_ell[:, 1].mean(), 6, xerr = blob_ell[:,1].std(), color = 'g', marker = '^', markersize = 8)
    plt.xlabel('Length / cm')
    plt.ylabel('Counts')
    plt.legend(loc = 'upper left')
    
    plt.subplot(133)
    plt.title('Blob velocities')
    plt.hist( blob_vel[:, 0], bins = 15, histtype = 'step', label='radial', linewidth = 2 )
    plt.hist( blob_vel[:, 1], bins = 15, histtype = 'step', label='poloidal', linewidth = 2 )
    plt.errorbar( blob_vel[:, 0].mean(), 3, xerr = blob_vel[:,0].std(), color = 'b', marker = '>', markersize = 8)
    plt.errorbar( blob_vel[:, 1].mean(), 6, xerr = blob_vel[:,1].std(), color = 'g', marker = '^', markersize = 8)
    plt.ylim(0, plt.ylim()[1])
    plt.xlabel('COM velocity / m/s')
    plt.ylabel('Counts')
    plt.legend(loc = 'upper left')
        
    try:
        logger.info('Mean radial velocity: %6f +- %6f' % ( blob_vel[:, 0].mean(), blob_vel[:,0].std() ) )
        logger.info('Mean poloidal velocity: %6f +- %6f' % ( blob_vel[:, 1].mean(), blob_vel[:,1].std() ) )
        logger.info('Mean radial length: %6f +- %6f' % ( blob_ell[:, 0].mean(), blob_ell[:,0].std() ) )
        logger.info('Mean poloidal length: %6f +- %6f' % ( blob_ell[:, 1].mean(), blob_ell[:,1].std() ) )
    except: 
        print 'Mean radial velocity: %6f +- %6f' % ( blob_vel[:, 0].mean(), blob_vel[:,0].std() ) 
        print 'Mean poloidal velocity: %6f +- %6f' % ( blob_vel[:, 1].mean(), blob_vel[:,1].std() )
        print 'Mean radial length: %6f +- %6f' % ( blob_ell[:, 0].mean(), blob_ell[:,0].std() )
        print 'Mean poloidal length: %6f +- %6f' % ( blob_ell[:, 1].mean(), blob_ell[:,1].std() )


    if ( save == True ):
        F = plt.gcf()
        filename = '%d/results/%d_thresh20_com_blobs_meaninsol_2.eps' % ( shotnr, shotnr )
        print 'Saving file %s' % filename
        F.savefig( filename )
        plt.close() 
 
#    plt.figure()
#    plt.plot(blob_vel[:,0], 'ko')
#    print np.shape(blob_vel)


 
 
def plot_blob_scatter1(blob_amps, blob_ell, blob_vel, blob_shape, shotnr, fail_list, save = False, logger = None):    
    
    plt.figure( figsize = (14,7) )
    plt.subplot(131)
    plt.title('Amplitude vs. blob length')
    plt.plot(blob_amps, blob_ell[:, 0], '>', markersize = 6, label='radial')
    plt.plot(blob_amps, blob_ell[:, 1], 'g^', markersize = 6, label='poloidal')
    plt.legend()
    plt.xlabel('Intensity / a.u.')
    plt.ylabel('Length / cm')
    
    plt.subplot(132)
    plt.title('Amplitude vs. blob velocity')
    plt.plot(blob_amps, blob_vel[:, 0], '>', markersize = 6, label='radial')
    plt.plot(blob_amps, blob_vel[:, 1], 'g^', markersize = 6, label='poloidal')
    plt.ylim(-1000., 1000.)
    plt.legend()
    plt.xlabel('Intensity / a.u.')
    plt.ylabel('Velocity / ms^-1')
    
    amp0_idx = np.argwhere(blob_amps < 2.0)
    amp1_idx = np.argwhere( (blob_amps > 2.0) & (blob_amps < 3.0) )
    amp2_idx = np.argwhere( (blob_amps > 3.0) & (blob_amps < 4.0) )
    amp3_idx = np.argwhere(blob_amps < 4.0)
  
     # Correlate velocity against filament size
    corr_vr_ell = (( blob_vel[:,0] - blob_vel[:,0].mean() ) * ( blob_ell[:,1] - blob_ell[:,1].mean() )).mean() / \
        ( blob_vel[:,0].std() * blob_ell[:,1].std() )

    plt.subplot(133)
    plt.title('Length vs Velocity.\\ N = %d, Correlation: %f' % ( np.size(blob_vel[:,0]), corr_vr_ell ))
    plt.plot( blob_ell[amp0_idx,1], blob_vel[amp0_idx, 0], 'bv', markersize = 6 )
    plt.plot( blob_ell[amp1_idx,1], blob_vel[amp1_idx, 0], 'gs', markersize = 4 )
    plt.plot( blob_ell[amp2_idx,1], blob_vel[amp2_idx, 0], 'ro', markersize = 3 )
    plt.plot( blob_ell[amp3_idx,1], blob_vel[amp3_idx, 0], 'k^', markersize = 2 )
    plt.xlabel('Poloidal length / cm')
    plt.ylabel('Radial velocity / ms^-1')

    try:
        logger.info('Correlation v_rad vs ell_p = %f' % corr_vr_ell )
    except:
        print 'Correlation v_rad vs ell_p = %f' % corr_vr_ell

    if ( save == True ):
        F = plt.gcf()
        filename = '%d/results/%d_thresh20_com_scatter_meaninsol_2.eps' % ( shotnr, shotnr )
        print 'Saving file %s' % filename
        F.savefig( filename )
        plt.close()

    plt.show()    
    return 0





def statistics_velocity(shotnr, blobtrails, frames, frame_info, good_domain):
    """
    More elaborate velocity analysis.
    We estimate the blob velocity only in the specified trigger domain. Ideally, this is the
    blob velocity in the scrape-off layer. Then the trigger domain excludes the are before the
    separatrix and behind the LCFS.
    """
    np.set_printoptions(linewidth=999999)


#    # Get R,z projection, grid data
#    rz_array, transform_data = make_rz_array(frame_info)
#    xxi, yyi = np.meshgrid( np.linspace( np.min(rz_array[:,:,0] ), np.max( rz_array[:,:,0] ),64 ), np.linspace( np.min( rz_array[:,:,1] ), np.max( rz_array[:,:,1] ),64 ) )
#    xyi = np.concatenate( (xxi[:,:,np.newaxis], yyi[:,:,np.newaxis]), axis=2 )
    
    num_events = len(blobtrails)    
    
    velocities = np.zeros( num_events )
    amplitudes = np.zeros( num_events )
    
    for idx, t in enumerate(blobtrails):
        trail = t.get_trail_com()
        vel   = t.get_velocity_com()  
        amp   = t.get_amp()
        
        vtrail = []
        amps   = []
        
        print np.shape(vel), np.shape(trail), np.shape(amp)
        
        for pos_idx, pos in enumerate(trail[:1]):
            #print 'Blob com at ', trail[pos_idx].round().astype('int').tolist()
            # Check if the position is in good_domain
            
            print np.array([i in good_domain for i in trail.round().astype('int').tolist()])
            
            if trail[pos_idx].round().astype('int').tolist() in good_domain:
#                print trail[pos_idx].round().astype('int').tolist(), ' is good '            
                vtrail.append( vel[pos_idx] )
                amps.append( amp[pos_idx] )
                
        print 'Blob %d, %d positions in good domain' % ( idx, len(amps) )
        
        vtrail = np.array(vtrail)
        amps   = np.array(vtrail)
        velocities[idx] = vtrail.mean()
        amplitudes[idx] = amps.max()


    plt.figure()
    plt.scatter( amplitudes, velocities )
    
    
#    plt.figure(0)
#    for i in np.arange(1,4):
#        plt.subplot(1,3,i)
#    plt.xlabel('Frame in blob evolution')
#    plt.ylabel('V_COM')
#    plt.title('Comparison of blob velocity evolution')
#    
#    plt.figure(1)
#    for i in np.arange(1,4):
#        plt.subplot(1,3,i)
#    plt.xlabel('Radial blob position')
#    plt.ylabel('V_COM')
#        
#    for idx, trail in enumerate(blobtrails[:50]):
#        symbol = 'b-'   
#        if ( trail.get_amp().max() < 3.5 ):
#            symbol = 'b-'
#            spnr = 1
#        elif ( trail.get_amp().max() > 3.5 and trail.get_amp().max() < 4.5 ):
#            symbol = 'g.-'
#            spnr = 2
#        elif ( trail.get_amp().max > 4.5 ):
#            symbol = 'k--'
#            spnr = 3
#    
#        # Plot the velocity for each blob#
#
#        plt.subplot(1,3,3)
#        if ( trail.get_velocity_com(rz_array)[:,0].std() > 450. ):
#            plt.figure()
#            plt.plot( trail.get_tau()[:-1], trail.get_velocity_com(rz_array)[:,0], label='rms = %5.2f' % trail.get_velocity_com(rz_array)[:,0].std())    
#        plt.legend(loc='upper left')
#    
#        # Plot the velocity against radial position
#        plt.figure(1)
#        plt.subplot(1,3,3)
#        plt.plot( trail.get_trail_com(rz_array)[:-1,0], trail.get_velocity_com(rz_array)[:,0], label='rms = %5.2f' % trail.get_velocity_com(rz_array)[:,0].std())    
#        plt.legend(loc='upper left')

    plt.show()


