from __future__ import print_function
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


alphabet_ = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z','_','!']

alphabet = []
for i in range(len(alphabet_)):
    for j in range(len(alphabet_)):
        alphabet_bi = alphabet_[i] + alphabet_[j]
        alphabet.append(alphabet_bi)

        ['  ', ' a', ' b', ' c', ' d', ' e', ' f', ' g', ' h', ' i', ' j', ' k', ' l', ' m', ' n', ' o', ' p', ' q',
         ' r', ' s', ' t', ' u', ' v', ' w', ' x', ' y', ' z', ' _', ' !', 'a ', 'aa', 'ab', 'ac', 'ad', 'ae', 'af',
         'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax',
         'ay', 'az', 'a_', 'a!', 'b ', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm',
         'bn', 'bo', 'bp', 'bq', 'br', 'bs', 'bt', 'bu', 'bv', 'bw', 'bx', 'by', 'bz', 'b_', 'b!', 'c ', 'ca', 'cb',
         'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj', 'ck', 'cl', 'cm', 'cn', 'co', 'cp', 'cq', 'cr', 'cs', 'ct',
         'cu', 'cv', 'cw', 'cx', 'cy', 'cz', 'c_', 'c!', 'd ', 'da', 'db', 'dc', 'dd', 'de', 'df', 'dg', 'dh', 'di',
         'dj', 'dk', 'dl', 'dm', 'dn', 'do', 'dp', 'dq', 'dr', 'ds', 'dt', 'du', 'dv', 'dw', 'dx', 'dy', 'dz', 'd_',
         'd!', 'e ', 'ea', 'eb', 'ec', 'ed', 'ee', 'ef', 'eg', 'eh', 'ei', 'ej', 'ek', 'el', 'em', 'en', 'eo', 'ep',
         'eq', 'er', 'es', 'et', 'eu', 'ev', 'ew', 'ex', 'ey', 'ez', 'e_', 'e!', 'f ', 'fa', 'fb', 'fc', 'fd', 'fe',
         'ff', 'fg', 'fh', 'fi', 'fj', 'fk', 'fl', 'fm', 'fn', 'fo', 'fp', 'fq', 'fr', 'fs', 'ft', 'fu', 'fv', 'fw',
         'fx', 'fy', 'fz', 'f_', 'f!', 'g ', 'ga', 'gb', 'gc', 'gd', 'ge', 'gf', 'gg', 'gh', 'gi', 'gj', 'gk', 'gl',
         'gm', 'gn', 'go', 'gp', 'gq', 'gr', 'gs', 'gt', 'gu', 'gv', 'gw', 'gx', 'gy', 'gz', 'g_', 'g!', 'h ', 'ha',
         'hb', 'hc', 'hd', 'he', 'hf', 'hg', 'hh', 'hi', 'hj', 'hk', 'hl', 'hm', 'hn', 'ho', 'hp', 'hq', 'hr', 'hs',
         'ht', 'hu', 'hv', 'hw', 'hx', 'hy', 'hz', 'h_', 'h!', 'i ', 'ia', 'ib', 'ic', 'id', 'ie', 'if', 'ig', 'ih',
         'ii', 'ij', 'ik', 'il', 'im', 'in', 'io', 'ip', 'iq', 'ir', 'is', 'it', 'iu', 'iv', 'iw', 'ix', 'iy', 'iz',
         'i_', 'i!', 'j ', 'ja', 'jb', 'jc', 'jd', 'je', 'jf', 'jg', 'jh', 'ji', 'jj', 'jk', 'jl', 'jm', 'jn', 'jo',
         'jp', 'jq', 'jr', 'js', 'jt', 'ju', 'jv', 'jw', 'jx', 'jy', 'jz', 'j_', 'j!', 'k ', 'ka', 'kb', 'kc', 'kd',
         'ke', 'kf', 'kg', 'kh', 'ki', 'kj', 'kk', 'kl', 'km', 'kn', 'ko', 'kp', 'kq', 'kr', 'ks', 'kt', 'ku', 'kv',
         'kw', 'kx', 'ky', 'kz', 'k_', 'k!', 'l ', 'la', 'lb', 'lc', 'ld', 'le', 'lf', 'lg', 'lh', 'li', 'lj', 'lk',
         'll', 'lm', 'ln', 'lo', 'lp', 'lq', 'lr', 'ls', 'lt', 'lu', 'lv', 'lw', 'lx', 'ly', 'lz', 'l_', 'l!', 'm ',
         'ma', 'mb', 'mc', 'md', 'me', 'mf', 'mg', 'mh', 'mi', 'mj', 'mk', 'ml', 'mm', 'mn', 'mo', 'mp', 'mq', 'mr',
         'ms', 'mt', 'mu', 'mv', 'mw', 'mx', 'my', 'mz', 'm_', 'm!', 'n ', 'na', 'nb', 'nc', 'nd', 'ne', 'nf', 'ng',
         'nh', 'ni', 'nj', 'nk', 'nl', 'nm', 'nn', 'no', 'np', 'nq', 'nr', 'ns', 'nt', 'nu', 'nv', 'nw', 'nx', 'ny',
         'nz', 'n_', 'n!', 'o ', 'oa', 'ob', 'oc', 'od', 'oe', 'of', 'og', 'oh', 'oi', 'oj', 'ok', 'ol', 'om', 'on',
         'oo', 'op', 'oq', 'or', 'os', 'ot', 'ou', 'ov', 'ow', 'ox', 'oy', 'oz', 'o_', 'o!', 'p ', 'pa', 'pb', 'pc',
         'pd', 'pe', 'pf', 'pg', 'ph', 'pi', 'pj', 'pk', 'pl', 'pm', 'pn', 'po', 'pp', 'pq', 'pr', 'ps', 'pt', 'pu',
         'pv', 'pw', 'px', 'py', 'pz', 'p_', 'p!', 'q ', 'qa', 'qb', 'qc', 'qd', 'qe', 'qf', 'qg', 'qh', 'qi', 'qj',
         'qk', 'ql', 'qm', 'qn', 'qo', 'qp', 'qq', 'qr', 'qs', 'qt', 'qu', 'qv', 'qw', 'qx', 'qy', 'qz', 'q_', 'q!',
         'r ', 'ra', 'rb', 'rc', 'rd', 're', 'rf', 'rg', 'rh', 'ri', 'rj', 'rk', 'rl', 'rm', 'rn', 'ro', 'rp', 'rq',
         'rr', 'rs', 'rt', 'ru', 'rv', 'rw', 'rx', 'ry', 'rz', 'r_', 'r!', 's ', 'sa', 'sb', 'sc', 'sd', 'se', 'sf',
         'sg', 'sh', 'si', 'sj', 'sk', 'sl', 'sm', 'sn', 'so', 'sp', 'sq', 'sr', 'ss', 'st', 'su', 'sv', 'sw', 'sx',
         'sy', 'sz', 's_', 's!', 't ', 'ta', 'tb', 'tc', 'td', 'te', 'tf', 'tg', 'th', 'ti', 'tj', 'tk', 'tl', 'tm',
         'tn', 'to', 'tp', 'tq', 'tr', 'ts', 'tt', 'tu', 'tv', 'tw', 'tx', 'ty', 'tz', 't_', 't!', 'u ', 'ua', 'ub',
         'uc', 'ud', 'ue', 'uf', 'ug', 'uh', 'ui', 'uj', 'uk', 'ul', 'um', 'un', 'uo', 'up', 'uq', 'ur', 'us', 'ut',
         'uu', 'uv', 'uw', 'ux', 'uy', 'uz', 'u_', 'u!', 'v ', 'va', 'vb', 'vc', 'vd', 've', 'vf', 'vg', 'vh', 'vi',
         'vj', 'vk', 'vl', 'vm', 'vn', 'vo', 'vp', 'vq', 'vr', 'vs', 'vt', 'vu', 'vv', 'vw', 'vx', 'vy', 'vz', 'v_',
         'v!', 'w ', 'wa', 'wb', 'wc', 'wd', 'we', 'wf', 'wg', 'wh', 'wi', 'wj', 'wk', 'wl', 'wm', 'wn', 'wo', 'wp',
         'wq', 'wr', 'ws', 'wt', 'wu', 'wv', 'ww', 'wx', 'wy', 'wz', 'w_', 'w!', 'x ', 'xa', 'xb', 'xc', 'xd', 'xe',
         'xf', 'xg', 'xh', 'xi', 'xj', 'xk', 'xl', 'xm', 'xn', 'xo', 'xp', 'xq', 'xr', 'xs', 'xt', 'xu', 'xv', 'xw',
         'xx', 'xy', 'xz', 'x_', 'x!', 'y ', 'ya', 'yb', 'yc', 'yd', 'ye', 'yf', 'yg', 'yh', 'yi', 'yj', 'yk', 'yl',
         'ym', 'yn', 'yo', 'yp', 'yq', 'yr', 'ys', 'yt', 'yu', 'yv', 'yw', 'yx', 'yy', 'yz', 'y_', 'y!', 'z ', 'za',
         'zb', 'zc', 'zd', 'ze', 'zf', 'zg', 'zh', 'zi', 'zj', 'zk', 'zl', 'zm', 'zn', 'zo', 'zp', 'zq', 'zr', 'zs',
         'zt', 'zu', 'zv', 'zw', 'zx', 'zy', 'zz', 'z_', 'z!', '_ ', '_a', '_b', '_c', '_d', '_e', '_f', '_g', '_h',
         '_i', '_j', '_k', '_l', '_m', '_n', '_o', '_p', '_q', '_r', '_s', '_t', '_u', '_v', '_w', '_x', '_y', '_z',
         '__', '_!', '! ', '!a', '!b', '!c', '!d', '!e', '!f', '!g', '!h', '!i', '!j', '!k', '!l', '!m', '!n', '!o',
         '!p', '!q', '!r', '!s', '!t', '!u', '!v', '!w', '!x', '!y', '!z', '!_', '!!']
# alphabet_ord = list(sum([ord(i) for i in j]) for j in alphabet)
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))
# num_to_alpha = dict(enumerate(alphabet_ord))

MAX_STROKE_LEN = 420
MAX_CHAR_LEN = 35


def align(coords):
    """
    corrects for global slant/offset in handwriting strokes
    """
    coords = np.copy(coords)
    X, Y = coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1)
    X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
    offset, slope = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y).squeeze()
    theta = np.arctan(slope)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )
    coords[:, :2] = np.dot(coords[:, :2], rotation_matrix) - offset
    return coords


def skew(coords, degrees):
    """
    skews strokes by given degrees
    """
    coords = np.copy(coords)
    theta = degrees * np.pi/180
    A = np.array([[np.cos(-theta), 0], [np.sin(-theta), 1]])
    coords[:, :2] = np.dot(coords[:, :2], A)
    return coords


def stretch(coords, x_factor, y_factor):
    """
    stretches strokes along x and y axis
    """
    coords = np.copy(coords)
    coords[:, :2] *= np.array([x_factor, y_factor])
    return coords


def add_noise(coords, scale):
    """
    adds gaussian noise to strokes
    """
    coords = np.copy(coords)
    coords[1:, :2] += np.random.normal(loc=0.0, scale=scale, size=coords[1:, :2].shape)
    return coords


def encode_ascii(ascii_string):
    """
    encodes ascii string to array of ints
    """
    ascii_string_bi=[ascii_string[i]+ascii_string[i+1] for i in range(len(ascii_string)-1)]
    return np.array(list(map(lambda x: alpha_to_num[x], ascii_string_bi)) + [0])



def interpolate(coords, factor=2):
    """
    interpolates strokes using cubic spline
    """
    coords = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    new_coords = []
    for stroke in coords:

        if len(stroke) == 0:
            continue

        xy_coords = stroke[:, :2]

        if len(stroke) > 3:
            f_x = interp1d(np.arange(len(stroke)), stroke[:, 0], kind='cubic')
            f_y = interp1d(np.arange(len(stroke)), stroke[:, 1], kind='cubic')

            xx = np.linspace(0, len(stroke) - 1, factor*(len(stroke)))
            yy = np.linspace(0, len(stroke) - 1, factor*(len(stroke)))

            x_new = f_x(xx)
            y_new = f_y(yy)

            xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])

        stroke_eos = np.zeros([len(xy_coords), 1])
        stroke_eos[-1] = 1.0
        stroke = np.concatenate([xy_coords, stroke_eos], axis=1)
        new_coords.append(stroke)

    coords = np.vstack(new_coords)
    return coords


def normalize(offsets):
    """
    normalizes strokes to median unit norm
    """
    offsets = np.copy(offsets)
    offsets[:, :2] /= np.median(np.linalg.norm(offsets[:, :2], axis=1))
    return offsets


def coords_to_offsets(coords):
    """
    convert from coordinates to offsets
    """
    offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    offsets = np.concatenate([np.array([[0, 0, 1]]), offsets], axis=0)
    return offsets


def offsets_to_coords(offsets):
    """
    convert from offsets to coordinates
    """
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def draw(
        offsets,
        ascii_seq=None,
        align_strokes=True,
        denoise_strokes=True,
        interpolation_factor=None,
        save_file=None
):
    strokes = offsets_to_coords(offsets)

    if denoise_strokes:
        strokes = denoise(strokes)

    if interpolation_factor is not None:
        strokes = interpolate(strokes, factor=interpolation_factor)

    if align_strokes:
        strokes[:, :2] = align(strokes[:, :2])

    fig, ax = plt.subplots(figsize=(12, 3))

    stroke = []
    for x, y, eos in strokes:
        stroke.append((x, y))
        if eos == 1:
            coords = zip(*stroke)
            ax.plot(coords[0], coords[1], 'k')
            stroke = []
    if stroke:
        coords = zip(*stroke)
        ax.plot(coords[0], coords[1], 'k')
        stroke = []

    ax.set_xlim(-50, 600)
    ax.set_ylim(-40, 40)

    ax.set_aspect('equal')
    plt.tick_params(
        axis='both',
        left='off',
        top='off',
        right='off',
        bottom='off',
        labelleft='off',
        labeltop='off',
        labelright='off',
        labelbottom='off'
    )

    if ascii_seq is not None:
        if not isinstance(ascii_seq, str):
            ascii_seq = ''.join(list(map(chr, ascii_seq)))
        plt.title(ascii_seq)

    if save_file is not None:
        plt.savefig(save_file)
        print('saved to {}'.format(save_file))
    else:
        plt.show()
    plt.close('all')
