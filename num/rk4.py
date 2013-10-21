"""

Runge-Kutta 4 integration.

dxfun	: Right hand side function.
y		: State vector of the system
delta_t	: Timestep

"""


def rk4_step(dxfun, y, t, delta_t, args=None):
#def rk4_step(dxfun, y, t, delta_t, args):
    # Concatenate delta_t and arguments
    #args = [delta_t] + args
    k_1 = dxfun(y, t, args)
    k_2 = dxfun(y + 0.5 * delta_t * k_1, t + 0.5 * delta_t, args)
    k_3 = dxfun(y + 0.5 * delta_t * k_2, t + 0.5 * delta_t, args)
    k_4 = dxfun(y + delta_t * k_3, t + delta_t, args)
    return (delta_t * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6.)

#
