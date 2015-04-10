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


def save_contour_plot(x, y, z, filename="pdf.png", title=""):
    fig = plt.figure()
    ax = fig.gca()
    cs = ax.contour(
        x, y, z, cmap=plt.cm.Paired)
    ## cs.set_clim(0, 1.6)
    plt.clabel(cs, inline=1, fontsize=5, fmt="%1.5f")
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


def time_averaged_pdf2(
    N0=0, N=3,
    vmax=1, resolution=0.05,
    x1=np.array([0, 0, 0]),  # position
    x2=np.array([0.1, 0.1, 0.1])  # position
):
    """
    Return pdf p2(v, x)
    """

    data = np.genfromtxt(
                "pdf/VX-{0:04d}.csv".format(N0),
                delimiter=' ')
    #for n in np.arange(N0 + 1, N0 + N):
        #data = np.concatenate(
            #(
                #data,
                #np.genfromtxt("pdf/VX-{0:04d}.csv".format(n), delimiter=' ')
            #),
            #axis=0)

    number_of_particles = data.shape[0]
    print "Number of particles {0}".format(number_of_particles)

    # These are ordered pairs so we have to take everything into account
    pairs = np.zeros(((number_of_particles - 1) * number_of_particles, 12))
    k = 0
    #def pairs(lst):
        #i = iter(lst)
        #first = prev = item = i.next()
        #for item in i:
            #yield prev, item
            #prev = item
        #yield item, first
    for n in range(number_of_particles):
        # Iterate over all non-n indices
        for m in range(number_of_particles):
            if n != m:
                pairs[k, :] = np.concatenate((data[n, :], data[m, :]), axis=1)
                k += 1

    print "Number of pairs {0}".format(pairs.shape[0])
    # filtering
    #data = data[np.abs(data[:, 0]) < 0.25, :]
    #data = data[np.abs(data[:, 1]) < 0.25, :]
    #data = data[np.abs(data[:, 2]) < 0.25, :]
    #data = data[:, 3:5]
    #print data.shape

    kde = KDEMultivariate(
        data=pairs[:, np.array([0, 1, 2, 3, 6, 7, 8, 9])],
        bw='normal_reference', var_type='cccccccc')

    vx1, vx2 = np.mgrid[-vmax:vmax:resolution, -vmax:vmax:resolution]
    dA = resolution**2
    n1 = vx1.shape[0]
    n2 = vx1.shape[1]
    pdf = np.zeros((n1, n2))
    area = 0
    for i in range(n1):
        for j in range(n2):
            v1 = np.array([vx1[i, j]])
            v2 = np.array([vx2[i, j]])
            pdf[i, j] = kde.pdf(np.concatenate((x1, v1, x2, v2), axis=1))
            area += pdf[i, j] * dA
    save_contour_plot(vx1, vx2, pdf/area, filename="v-pdf2.png")


def time_averaged_pdf_product(
    N0=0, N=3,
    vmax=1, resolution=0.05,
    x1=np.array([0, 0, 0]),  # position
    x2=np.array([0.1, 0.1, 0.1])  # position
):
    """
    Return pdf p2(v, x)
    """

    data = np.genfromtxt(
                "pdf/VX-{0:04d}.csv".format(N0),
                delimiter=' ')
    #for n in np.arange(N0 + 1, N0 + N):
        #data = np.concatenate(
            #(
                #data,
                #np.genfromtxt("pdf/VX-{0:04d}.csv".format(n), delimiter=' ')
            #),
            #axis=0)

    number_of_particles = data.shape[0]
    print "Number of particles {0}".format(number_of_particles)

    # filtering
    #data = data[np.abs(data[:, 0]) < 0.25, :]
    #data = data[np.abs(data[:, 1]) < 0.25, :]
    #data = data[np.abs(data[:, 2]) < 0.25, :]
    #data = data[:, 3:5]
    #print data.shape

    kde = KDEMultivariate(
        data=data[:, np.array([0, 1, 2, 3])],
        bw='normal_reference', var_type='cccc')

    vx1, vx2 = np.mgrid[-vmax:vmax:resolution, -vmax:vmax:resolution]
    dl = resolution
    n1 = vx1.shape[0]
    n2 = vx1.shape[1]
    pdf1 = np.zeros((n1, n2))
    pdf2 = np.zeros((n1, n2))
    area1 = 0
    area2 = 0
    for i in range(n1):
        for j in range(n2):
            v1 = np.array([vx1[i, j]])
            v2 = np.array([vx2[i, j]])
            pdf1[i, j] = kde.pdf(np.concatenate((x1, v1), axis=1))
            pdf2[i, j] = kde.pdf(np.concatenate((x2, v2), axis=1))
            area1 += pdf1[i, j] * dl
            area2 += pdf2[i, j] * dl
    f1v1 = pdf1 / area1
    f1v2 = pdf2 / area2
    f2 = f1v1 * f1v2
    area1 = 0
    area2 = 0
    area3 = 0
    for i in range(n1):
        for j in range(n2):
            area1 += f1v1[i, j] * dl
            area2 += f1v2[i, j] * dl
            area3 += f2[i, j] * dl**2

    print area1, area2, area3
    save_contour_plot(
        vx1, vx2, f2,
        filename="v-pdfprod.png")

set_plt_params()

# speed_graphs()
# velocity_graphs(N0=0, N=5000, resolution=0.05)

# time_averaged_pdf(N0=5, N=10)
#time_averaged_pdf2(N0=0, N=10)
time_averaged_pdf_product(N0=0, N=10)
