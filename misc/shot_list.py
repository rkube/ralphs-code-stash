#!/usr/bin/python
#-*- Encoding: UTF-8 -8-

import pymssql

def shot_list( runs = [], start = -1, stop = -1 ):
    """
    Fetch all shots for a single runday, a list of rundays or between two rundays

    Input:
    ======
        run - The run or list of runs for which to return the shots. Either an int or a list.
        start(optional) - First day of runs to return the shots. Integer.
        stop(optional) - Last day of runs to return the shots. Integer

    Output: 
    =======
    A list of shots.
    
    Optional output parameters:
    ===========================
    
    Side effects:
    =============

    Call:
    =====
    shots = shot_list( 1100803 )
    shots = shot_list( [1100803, 1100804, 1100804] )
    shots = shot_list( start=1100803, stop=1000805 )


    Modification history:
        Ralph Kube 31-Aug-2012  Initial coding

    """

    criteria = ''
    try:
        # Assume run is a list, build the query accordingly
        if ( len(runs) > 0 ):
            criteria = 'RUN = ' + ' or RUN = '.join([ str(r) for r in runs])    
    except TypeError:
        # Oops, run is a single shot
        criteria = 'RUN = %d' % runs

    if ( (start > 0) and (stop > 0) ):
        if ( len(criteria) == 0 ):
            criteria = 'RUN >= %d and RUN <= %d' % (start, stop)
        elif ( len(criteria) > 0 ):
            criteria = criteria + ' or (RUN >= %d and RUN <= %d)' % (start, stop)

    print criteria

    conn = pymssql.connect(host = 'alcdb2', user='rkube', password='pfcworld', database='logbook')
    cur = conn.cursor()
    cur.execute('SELECT shot from shots where %s' % criteria)
    shots = cur.fetchall()

    return [s[0] for s in shots]



if __name__ == '__main__':
    r_shots = shot_list( [1100803, 1100804, 1100805] )
    print r_shots


