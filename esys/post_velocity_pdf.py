import numpy as np
import matplotlib as mpl
from statsmodels.nonparametric.kernel_density import KDEMultivariate
mpl.use("agg")
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.ioff()


def maxwell_boltzman_speed(v, kT, m):
    return \
        (m / (2 * np.pi * kT))**1.5 \
        * 4.0 * np.pi * v**2 \
        * np.exp(- (m * v**2) / (2 * kT))


def save_contour_plot(x, y, z, filename="pdf.png", title=None):
    fig = plt.figure()
    ax = fig.gca()
    cs = ax.contour(x, y, z, cmap=plt.cm.Paired)
    ## cs.set_clim(0, 1.6)
    if title is not None:
        fig.suptitle(title, fontsize=7)

    plt.xlabel("First particle velocity $v_x^{(1)}$")
    plt.ylabel("Second particle velocity $v_x^{(2)}$")

    plt.clabel(cs, inline=1, fontsize=5, fmt="%1.1f")
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def set_plt_params(
        relative_fig_width=1.0, landscape=True, page_width=307.3, rescale_h=1):

    fig_width_pt = page_width * relative_fig_width
    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches

    if landscape:
        fig_height = fig_width * golden_mean  # height in inches
    else:
        fig_height = fig_width / golden_mean  # height in inches

    fig_height = fig_height * rescale_h
    fig_size = [fig_width, fig_height]
    params = {
        'font.family': 'serif',
        'axes.labelsize': 7,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'axes.labelcolor': 'black',
        'ytick.color': 'black',
        'xtick.color': 'black',
        'legend.handlelength': 4,
        'legend.fontsize': 7,
        # 'lines.markersize': 3,
        # 'xtick.labelsize': 7,
        # 'ytick.labelsize': 7,
        'text.usetex': True,
        'text.latex.unicode': True,
        'figure.figsize': fig_size,
        'pgf.texsystem': "xelatex",
        'pgf.rcfonts': False,
    }

    plt.rcParams.update(params)


def speed_graphs(N0=0, N=4500, vmax=3, resolution=300):
    data = dict(
        (
            n,
            np.genfromtxt(
                "pdf/v-{0:04d}.csv".format(n),
                delimiter=' ')
        ) for n in range(N))
    Tdata = np.genfromtxt("bulk.csv", delimiter=' ')
    T = Tdata[:, 2]
    t = Tdata[:, 1]

    x = np.linspace(0, vmax, resolution)

    for n in np.arange(N0, N):
        kde = KDEMultivariate(data[n], bw='normal_reference', var_type='c')
        fig = plt.figure()
        ax = fig.gca()
        fig.subplots_adjust(wspace=0)
        fig.suptitle("Time = {0:.2f} s".format(t[n]), fontsize=7)

        ax.set_ylim(-0.01, 2.5)
        plt.xlabel("Velocity norm")
        plt.ylabel("PDF")
        # Fix the seed for reproducibility
        ax.plot(x, kde.pdf(x), label="Simulation")
        ax.plot(
            x,
            maxwell_boltzman_speed(v=x, m=1, kT=T[n]),
            label="Maxwell-Boltzmann")
        ax.legend(loc='upper right', shadow=True)
        fig.savefig(
            "v-pdf{0:04d}.png".format(n),
            bbox_inches='tight', dpi=300)
        plt.close()


def velocity_graphs(N0=0, N=4500, vmax=1, resolution=0.05):
    data = dict(
        (
            n,
            np.genfromtxt(
                "pdf/VX-{0:04d}.csv".format(n),
                delimiter=' ')
        ) for n in range(N))
    Tdata = np.genfromtxt("bulk.csv", delimiter=' ')
    # T = Tdata[:, 2]
    t = Tdata[:, 1]

    x, y = np.mgrid[-vmax:vmax:resolution, -vmax:vmax:resolution]

    for n in np.arange(N0, N):
        kde = KDEMultivariate(
            data=data[n][:, 3:5],
            bw='normal_reference', var_type='cc')
        fig = plt.figure()
        ax = fig.gca()
        fig.subplots_adjust(wspace=0)
        fig.suptitle("Time = {0:.2f} s".format(t[n]), fontsize=7)

        plt.xlabel("$x$-velocity")
        plt.ylabel("$y$-velocity")
        nx = x.shape[0]
        ny = x.shape[1]
        pdf = np.zeros((nx, ny))
        print ("Evaluating the function")
        for i in range(nx):
            for j in range(ny):
                pdf[i, j] = kde.pdf([x[i, j], y[i, j]])

        #cs = ax.contour(x, y, pdf, vmin=0.0, vmax=1.6, label="Simulation")
        cs = ax.contour(x, y, pdf, label="Simulation", cmap=plt.cm.Paired)
        cs.set_clim(0, 1.6)
        plt.clabel(cs, inline=1, fontsize=5, fmt="%1.1f")
        fig.savefig(
            "v-pdf{0:04d}.png".format(n),
            bbox_inches='tight', dpi=300)
        plt.close()


def time_averaged_pdf(
    N0=0, N=3,
    vmax=1, resolution=0.05,
    x=np.array([0, 0, 0])  # position
):
    """
    Return pdf p(v, x)
    """

    data = np.genfromtxt(
                "pdf/VX-{0:04d}.csv".format(N0),
                delimiter=' ')
    for n in np.arange(N0 + 1, N0 + N):
        data = np.concatenate(
            (
                data,
                np.genfromtxt("pdf/VX-{0:04d}.csv".format(n), delimiter=' ')
            ),
            axis=0)

    print "Number of particles {0}".format(data.shape[0])
    # filtering
    #data = data[np.abs(data[:, 0] - x[0]) < 0.25, :]
    #data = data[np.abs(data[:, 1] - x[1]) < 0.25, :]
    #data = data[np.abs(data[:, 2] - x[2]) < 0.25, :]
    #data = data[:, 3:5]
    #print data.shape

    kde = KDEMultivariate(
        data=data[:, np.array([0, 1, 2, 3, 4])],
        bw='normal_reference', var_type='ccccc')

    vx, vy = np.mgrid[-vmax:vmax:resolution, -vmax:vmax:resolution]
    dA = resolution**2
    nx = vx.shape[0]
    ny = vx.shape[1]
    pdf = np.zeros((nx, ny))
    area = 0
    for i in range(nx):
        for j in range(ny):
            v = np.array([vx[i, j], vy[i, j]])
            pdf[i, j] = kde.pdf(np.concatenate((x, v), axis=1))
            area += pdf[i, j] * dA
    fig = plt.figure()
    ax = fig.gca()
    cs = ax.contour(vx, vy, pdf/area, label="Simulation", cmap=plt.cm.Paired)
    cs.set_clim(0, 1.6)
    plt.clabel(cs, inline=1, fontsize=5, fmt="%1.2f")
    fig.savefig(
        "v-pdf.png",
        bbox_inches='tight', dpi=300)
    plt.close()


def get_particles_in_a_box(particles, x, a=1.0):
    ps = particles[np.abs(particles[:, 0] - x[0]) < a, :]
    ps = ps[np.abs(ps[:, 1] - x[1]) < a, :]
    ps = ps[np.abs(ps[:, 2] - x[2]) < a, :]
    return ps


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def data_to_pdf(data, coords):
    num_of_variables = 1
    if len(data.shape) > 1:
        num_of_variables = data.shape[1]
    kde = KDEMultivariate(
        data=data, bw='normal_reference', var_type='c' * num_of_variables)
    return kde.pdf(coords)


def time_averaged_jointpdf(
    N0=0, N=3,
    vmax=1, resolution=0.05, a=1.0,
    x1=np.array([0, 0, 0]),  # position
    x2=np.array([0.1, 0.1, 0.1])  # position
):
    """
    Return pdf p2(v, x)
    """

    pairs = np.zeros((2, 2))
    for n in np.arange(N0, N0 + N):
        data = np.genfromtxt("pdf/VX-{0:04d}.csv".format(n), delimiter=' ')
        data1 = get_particles_in_a_box(data, x=x1, a=a)
        data1 = data1[:, 3]
        data2 = get_particles_in_a_box(data, x=x2, a=a)
        data2 = data2[:, 3]
        tmppairs = cartesian((data1, data2))
        pairs = np.concatenate((pairs, tmppairs), axis=0)

    pairs = pairs[2:, :]

    print "Number of pairs {0}".format(pairs.shape[0])

    # pairs = np.array([[0, 0], [1, 1], [2, 2]])
    kde = KDEMultivariate(
        data=pairs, bw='normal_reference', var_type='cc')

    vx1, vx2 = np.mgrid[-vmax:vmax:resolution, -vmax:vmax:resolution]

    coords = np.vstack([item.ravel() for item in [vx1, vx2]])
    pdf = data_to_pdf(pairs, coords)
    save_contour_plot(
        vx1, vx2, pdf.reshape(vx1.shape), filename="v-pdf2.png",
        title="$f^{(2)}(v^{(1)}_x,v^{(1)}_x)$")


def time_averaged_pdf_product(
    N0=0, N=3,
    vmax=1, resolution=0.05, a=1,
    x1=np.array([0, 0, 0]),  # position
    x2=np.array([0.4, 0.4, 0.4])  # position
):
    """
    Return pdf p2(v, x)
    """

    data = np.genfromtxt(
                "pdf/VX-{0:04d}.csv".format(N0),
                delimiter=' ')
    for n in np.arange(N0 + 1, N0 + N):
        data = np.concatenate(
            (
                data,
                np.genfromtxt("pdf/VX-{0:04d}.csv".format(n), delimiter=' ')
            ),
            axis=0)
    # filtering
    data1 = get_particles_in_a_box(data, x=x1, a=a)
    data1 = data1[:, 3]
    data2 = get_particles_in_a_box(data, x=x2, a=a)
    data2 = data2[:, 3]
    print "Number of particles in region 1 {0}".format(data1.shape[0])
    print "Number of particles in region 2 {0}".format(data2.shape[0])

    vx1, vx2 = np.mgrid[-vmax:vmax:resolution, -vmax:vmax:resolution]
    coords = vx1.flatten()
    pdf1 = data_to_pdf(data1, coords)
    coords = vx2.flatten()
    pdf2 = data_to_pdf(data1, coords)

    save_contour_plot(
        vx1, vx2, (pdf1*pdf2).reshape(vx1.shape),
        filename="v-pdfprod.png", title="$f^{(1)}(v^{(1)})f^{(1)}(v^{(2)}_x)$")

set_plt_params()

# speed_graphs()
# velocity_graphs(N0=0, N=5000, resolution=0.05)

# time_averaged_pdf(N0=5, N=10)
#time_averaged_jointpdf(
    #N0=4000, N=10, resolution=0.02, a=3,
    #x1=np.array([0, 0, 0]),
    #x2=np.array([6, 0, 0]))
#time_averaged_pdf_product(
    #N0=4000, N=10, resolution=0.01, a=3,
    #x1=np.array([0, 0, 0]),
    #x2=np.array([6, 0, 0]))
