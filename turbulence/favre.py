#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-

def favg(field, density, dy):
    """
    Computes the Favre average of a field f.
    Given f, decompose as
    f = bar{f} + tilde{f} where
    bar{f} = <n f> / <n>
    and tilde{f} = f - bar{f}

     and <f> = 1/L \int_{0}^{L} dy f
         
             
    Input:
    ======
    field..... ndarray, float. axis0: y, axis1: x
    density... ndarray, float. axis0: y, axis1: x
    dy........ ndarray, float. axis0: y, axis1: x

    Output:
    =======
    bar_f..... ndarray, float. Favre averaged f
    tilde_f... ndarray, float. Fluctuation on f
    """
                                                             
    res = (density * field * dy).sum(axis=0) / (density * dy).sum(axis=0)
    bar_f = res[np.newaxis, :].repeat(field.shape[0], axis=0)
    tilde_f = field - bar_f
                                                                             
    return bar_f, tilde_f
# End of file favre.py
