import numpy as np
from flask import Flask
from math import sqrt

from scipy.optimize import minimize

app = Flask(__name__)

PINE = 0
OAK = 1
WOODS = [PINE, OAK]

MIN_THICKNESS = 3 / 16
MAX_THICKNESS = 3 / 4
INC_THICKNESS = 1 / 16

MAX_HEIGHT = 4
MAX_RATIO = 2

P_GOAL = 1250

sig = [0] * 2
tau = [0] * 2
tau_g = [0] * 2
E = [0] * 2

sig[OAK] = 17873
sig[PINE] = 14327
tau[OAK] = 2873
tau[PINE] = 1492
tau_g[OAK] = 1391
tau_g[PINE] = 989


def calc_failure_states(l_v, l_h, t_v, t_ht, t_hb, w_v, w_t, w_b):
    y_g = (t_v * l_v ** 2 + (l_v - t_ht / 2) * l_h * t_ht + t_hb / 2 * l_h * t_hb) / \
          (2 * l_v * t_v + l_h * t_ht + l_h * t_hb)
    i_z = t_v * l_v ** 3 / 6 + l_h * t_hb ** 3 / 12 + l_h * t_ht ** 3 / 12 + 2 * l_v * t_v * (
            l_v / 2 - y_g) ** 2 + l_h * t_hb * (t_hb / 2 - y_g) ** 2 + l_h * t_ht * (l_v - t_ht / 2 - y_g) ** 2

    p_bend = 5 * min(sig[w_v], sig[w_b]) * i_z / (24 * y_g)
    p_glue = 10 * min(tau_g[w_v], tau_g[w_b], tau_g[w_t]) * i_z / (3 * l_h * (y_g - t_hb / 2))
    p_shear = 10 * tau[w_v] * i_z * t_v / (3 * (l_h * t_ht * (l_v - t_ht / 2 - y_g) + (l_v - y_g) ** 2))

    return p_bend, p_glue, p_shear


def optimize_beam(p_goal):
    best = 10 ** 20
    store = None
    fail = None
    top_wood = None
    bot_wood = None
    side_wood = None

    ret = ''

    for w_v in WOODS:
        for w_t in WOODS:
            for w_b in WOODS:
                for t_v in np.arange(MIN_THICKNESS, MAX_THICKNESS, INC_THICKNESS):
                    for t_ht in np.arange(MIN_THICKNESS, MAX_THICKNESS, INC_THICKNESS):
                        for t_hb in np.arange(MIN_THICKNESS, MAX_THICKNESS, INC_THICKNESS):
                            def error(x):
                                l_v, l_h = x
                                failures = calc_failure_states(l_v, l_h, t_v, t_ht, t_hb, w_v, w_t, w_b)

                                return sqrt(sum([(p_goal - failure) ** 2 for failure in failures]))

                            m = minimize(error, bounds=[(0, 4), (0, 1.5)], x0=np.array([2, .5]),
                                         options={'gtol': 1e-2})

                            if m['fun'] < best:
                                best = m['fun']
                                store = [*m['x'], t_v, t_hb, t_ht]
                                fail = calc_failure_states(*m['x'], t_v, t_ht, t_hb, w_v, w_t, w_b)
                                top_wood = w_t
                                bot_wood = w_b
                                side_wood = w_v

    print(best)
    print(store)
    print(fail)

    ret += str(store) + '\n'
    ret += str(fail) + '\n'

    for w in (top_wood, bot_wood, side_wood):
        if w == PINE:
            print("Pine")
            ret += 'PINE '
        else:
            print("Oak")
            ret += 'OAK '

    return ret


@app.route('/<int:p_goal>')
def index(p_goal):
    return str(optimize_beam(p_goal))


if __name__ == '__main__':
    app.run()
