"""
axes3d.py, original mplot3d verimport pytest

from mpl_toolkits.mplot3d import Axes3D, a
Parts fixed by Reinier Heeres <reinier@heeresfrom Minor additions by Ben Axelrod <baxelrod@coroware.com>
Significant updates and revisions by Ben Root <ben.v.root@gmail.com>$
Module containing Axes3from matplotlib import cm
from matplotlib import colors as mcolors"""

from collections import defaultdict
import functools
import itertools
import math
import textwrap

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wra$$$$$$$:
    ax = fig_test.subplots(subplot_kw=dict(projection='3d'))
    ax.set_visible(False)


@mpl3d_image_comparison(['aspects.png'], remove_text=False)
def test_aspects():
    aspects = ('auto', 'equal', 'equalxy', 'equalyz', 'equalxz'$$$$$$$$$$$$$$$$$$$$
    """
    3D Axes object.

    .. note::

        As a user, you do not instantiate Axes directly, but use Axes creation
        methods instead; e.g. from `.pyplot` or `.Figure`:
        `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` or `.Figure.add_axes`.$$$$$$$$
    """
    name$$$$$$$)), 2)
    for start, end in pts:
        if np.sum(np.abs(start - end)) == r[1] - r[0]:
            for ax in axs:
                ax.plot3D(*zip(start*scale, end*scale))
    for i, ax in enumerate(axs):
        ax.set_box_aspect((3, 4, 5))
        ax.set_aspect(aspects[i], adjustable='datalim')


@mpl3d_image_comparison(['aspects_adjust_box.png'], remove_text=False)
def test_aspects_adjust_box():
    aspects = ('auto', 'equal', 'equalxy', 'equalyz', 'equalxz')
    fig, axs = plt.subplots(1, len(aspects), subplot_kw={'projection': '3d'},
                            figsize=(11, 3))

    # Draw rectangular cuboid with side lengths [4, 3, 5]
    r = [0, 1]
    scale = np.array([4, 3, 5])
    pts = itertools.combinations(np.array(list(itertools.product(r, r, r))), 2)
    for start, end in pts:
        if np.sum(np.abs(start - end)) == r[1] - r[0]:
            for ax in axs:
                ax.plot3D(*zip(start*scale, end*scale))
    for i, ax in enumerate(axs):
        ax.set_aspect(aspects[i], adjustable='box')


def test_axes3d_repr():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_label('label')
    ax.set_title('title')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    assert repr(ax) == (
        "<Axes3D: label='label', "
        "title={'center': 'title'}, xlabel='x', ylabel='y', zlabel='z'>")


@mpl3d_image_comparison(['axes3d_primary_views.png'])
def test_axes3d_primary_views():
    # (elev, azim, roll)
    views = [(90, -90, 0),  # XY
             (0, -90, 0),   # XZ
             (0, 0, 0),     # YZ
             (-90, 90, 0),  # -XY
             (0, 90, 0),    # -XZ
             (0, 180, 0)]   # -YZ
    # When viewing primary planes, draw the two visible axes so they intersect
    # at their low values
    fig, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'})
    for i, ax in enumerate(axs.flat):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_proj_type('ortho')
        ax.view_init(elev=views[i][0], azim=views[i][1], roll=views[i][2])
    plt.tight_layout()


@mpl3d_image_comparison(['bar3d.png'])
def test_bar3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.arange(20)
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', align='edge', color=cs, alpha=0.8)


def test_bar3d_colors():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for c in ['red', 'green', 'blue', 'yellow']:
        xs = np.arange(len(c))
        ys = np.zeros_like(xs)
        zs = np.zeros_like(ys)
        # Color names with same length as xs/ys/zs should not be split into
        # individual letters.
        ax.bar3d(xs, ys, zs, 1, 1, 1, color=c)


@mpl3d_image_comparison(['bar3d_shaded.png'])
def test_bar3d_shaded():
    x = np.arange(4)
    y = np.arange(5)
    x2d, y2d = np.meshgrid(x, y)
    x2d, y2d = x2d.ravel(), y2d.ravel()
    z = x2d + y2d + 1  # Avoid triggering bug with zero-depth boxes.

    views = [(30, -60, 0), (30, 30, 30), (-30, 30, -90), (300, -30, 0)]
    fig = plt.figure(figsize=plt.figaspect(1 / len(views)))
    axs = fig.subplots(
        1, len(views),
        subplot_kw=dict(projection='3d')
    )
    for ax, (elev, azim, roll) in zip(axs, views):
        ax.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=True)
        ax.view_init(elev=elev, azim=azim, roll=roll)
    fig.canvas.draw()


@mpl3d_image_comparison(['bar3d_notshaded.png'])
def test_bar3d_notshaded():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(4)
    y = np.arange(5)
    x2d, y2d = np.meshgrid(x, y)
    x2d, y2d = x2d.ravel(), y2d.ravel()
    z = x2d + y2d
    ax.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=False)
    fig.canvas.draw()


def test_bar3d_lightsource():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ls = mcolors.LightSource(azdeg=0, altdeg=90)

    length, width = 3, 4
    area = length * width

    x, y = np.meshgrid(np.arange(length), np.arange(width))
    x = x.ravel()
    y = y.ravel()
    dz = x + y

    color = [cm.coolwarm(i/area) for i in range(area)]

    collection = ax.bar3d(x=x, y=y, z=0,
                          dx=1, dy=1, dz=dz,
                          color=color, shade=True, lightsource=ls)

    # Testing that the custom 90° lightsource produces different shading on
    # the top facecolors compared to the default, and that those colors are
    # precisely the colors from the colormap, due to the illumination parallel
    # to the z-axis.
    np.testing.assert_array_equal(color, collection._facecolor3d[1::6])


@mpl3d_image_comparison(['contour3d.png'])
def test_contour3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)


@mpl3d_image_comparison(['contour3d_extend3d.png'])
def test_contour3d_extend3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm, extend3d=True)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-20, 40)
    ax.set_zlim(-80, 80)


@mpl3d_image_comparison(['contourf3d.png'])
def test_contourf3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)


@mpl3d_image_comparison(['contourf3d_fill.png'])
def test_contourf3d_fill()
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
:

=======
        tcube = proj3d._proj_points(xyzs, M

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(np.arange(-2, 2, 0.25), np.arange(-2, 2, 0.25))
    Z = X.clip(0, 0)
    # This produces holes in the z=0 surface that causes rendering errors if
    # the Poly3DCollection is not aware of path code information (issue #4784)
    Z[::5, ::5] = 0.1
    ax.contourf(X, Y, Z, offset=0, levels=[-0.1, 0], cmap=cm.coolwarm)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 1)


@pytest.mark.parametrize('extend, levels', [['both', [2, 4, 6]],
                                            ['min', [2, 4, 6, 8]],
                                            ['max', [0, 2, 4, 6]]])
@check_figures_equal(extensions=["png"])
def test_contourf3d_extend(fig_test, fig_ref, extend, levels):
    X, Y = np.meshgrid(np.arange(-2, 2, 0.25), np.arange(-2, 2, 0.25))
    # Z is in the range [0, 8]
    Z = X**2 + Y**2

    # Manually set the over/under colors to be the end of the colormap
    cmap = mpl.colormaps['viridis'].copy()
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(255))
    # Set vmin/max to be the min/max values plotted on the reference image
    kwargs = {'vmin': 1, 'vmax': 7, 'cmap': cmap}

    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.contourf(X, Y, Z, levels=[0, 2, 4, 6, 8], **kwargs)

    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.contourf(X, Y, Z, levels, extend=extend, **kwargs)

    for ax in [ax_ref, ax_test]:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-10, 10)


@mpl3d_image_comparison(['tricontour.png'], tol=0.02)
def test_tricontour():
    fig = plt.figure()

    np.random.seed(19680801)
    x = np.random.rand(1000) - 0.5
    y = np.random.rand(1000) - 0.5
    z = -(x**2 + y**2)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.tricontour(x, y, z)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.tricontourf(x, y, z)


def test_contour3d_1d_input():
    # Check that 1D sequences of different length for {x, y} doesn't error
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    nx, ny = 30, 20
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    z = np.random.randint(0, 2, [ny, nx])
    ax.contour(x, y, z, [0.5])


@mpl3d_image_comparison(['lines3d.png'])
def test_lines3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z)


@check_figures_equal(extensions=["png"])
def test_plot_scalar(fig_test, fig_ref):
    ax1 = fig_test.add_subplot(projection='3d')
    ax1.plot([1], [1], "o")
    ax2 = fig_ref.add_subplot(projection='3d')
    ax2.plot(1, 1, "o")


def test_invalid_line_data():
    with pytest.raises(RuntimeError, match='x must be'):
        art3d.Line3D(0, [], [])
    with pytest.raises(RuntimeError, match='y must be'):
        art3d.Line3D([], 0, [])
    with pytest.raises(RuntimeError, match='z must be'):
        art3d.Line3D([], [], 0)

    line = art3d.Line3D([], [], [])
    with pytest.raises(RuntimeError, match='x must be'):
        line.set_data_3d(0, [], [])
    with pytest.raises(RuntimeError, match='y must be'):
        line.set_data_3d([], 0, [])
    with pytest.raises(RuntimeError, match='z must be'):
        line.set_data_3d([], [], 0)


@mpl3d_image_comparison(['mixedsubplot.png'])
def test_mixedsubplots():
    def f(t):
        return np.cos(2*np.pi*t) * np.exp(-t)

    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)

    fig = plt.figure(figsize=plt.figaspect(2.))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t1, f(t1), 'bo', t2, f(t2), 'k--', markerfacecolor='green')
    ax.grid(True)

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
    R = np.hypot(X, Y)
    Z = np.sin(R)

    ax.plot_surface(X, Y, Z, rcount=40, ccount=40,
                    linewidth=0, antialiased=False)

    ax.set_zlim3d(-1, 1)


@check_figures_equal(extensions=['png'])
def test_tight_layout_text(fig_test, fig_ref):
    # text is currently ignored in tight layout. So the order of text() and
    # tight_layout() calls should not influence the result.
    ax1 = fig_test.add_subplot(projection='3d')
    ax1.text(.5, .5, .5, s='some string')
    fig_test.tight_layout()

    ax2 = fig_ref.add_subplot(projection='3d')
    fig_ref.tight_layout()
    ax2.text(.5, .5, .5, s='some string')


@mpl3d_image_comparison(['scatter3d.png'])
def test_scatter3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               c='r', marker='o')
    x = y = z = np.arange(10, 20)
    ax.scatter(x, y, z, c='b', marker='^')
    z[-1] = 0  # Check that scatter() copies the data.
    # Ensure empty scatters do not break.
    ax.scatter([], [], [], c='r', marker='X')


@mpl3d_image_comparison(['scatter3d_color.png'])
def test_scatter3d_color():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Check that 'none' color works; these two should overlay to produce the
    # same as setting just `color`.
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               facecolor='r', edgecolor='none', marker='o')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               facecolor='none', edgecolor='r', marker='o')

    ax.scatter(np.arange(10, 20), np.arange(10, 20), np.arange(10, 20),
               color='b', marker='s')


@mpl3d_image_comparison(['scatter3d_linewidth.png'])
def test_scatter3d_linewidth():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Check that array-like linewidth can be set
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               marker='o', linewidth=np.arange(10))


@check_figures_equal(extensions=['png'])
def test_scatter3d_linewidth_modification(fig_ref, fig_test):
    # Changing Path3DCollection linewidths with array-like post-creation
    # should work correctly.
    ax_test = fig_test.add_subplot(projection='3d')
    c = ax_test.scatter(np.arange(10), np.arange(10), np.arange(10),
                        marker='o')
    c.set_linewidths(np.arange(10))

    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o',
                   linewidths=np.arange(10))


@check_figures_equal(extensions=['png'])
def test_scatter3d_modification(fig_ref, fig_test):
    # Changing Path3DCollection properties post-creation should work correctly.
    ax_test = fig_test.add_subplot(projection='3d')
    c = ax_test.scatter(np.arange(10), np.arange(10), np.arange(10),
                        marker='o')
    c.set_facecolor('C1')
    c.set_edgecolor('C2')
    c.set_alpha([0.3, 0.7] * 5)
    assert c.get_depthshade()
    c.set_depthshade(False)
    assert not c.get_depthshade()
    c.set_sizes(np.full(10, 75))
    c.set_linewidths(3)

    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o',
                   facecolor='C1', edgecolor='C2', alpha=[0.3, 0.7] * 5,
                   depthshade=False, s=75, linewidths=3)


@pytest.mark.parametrize('depthshade', [True, False])
@check_figures_equal(extensions=['png'])
def test_scatter3d_sorting(fig_ref, fig_test, depthshade):
    """Test that marker properties are correctly sorted."""

    y, x = np.mgrid[:10, :10]
    z = np.arange(x.size).reshape(x.shape)

    sizes = np.full(z.shape, 25)
    sizes[0::2, 0::2] = 100
    sizes[1::2, 1::2] = 100

    facecolors = np.full(z.shape, 'C0')
    facecolors[:5, :5] = 'C1'
    facecolors[6:, :4] = 'C2'
    facecolors[6:, 6:] = 'C3'

    edgecolors = np.full(z.shape, 'C4')
    edgecolors[1:5, 1:5] = 'C5'
    edgecolors[5:9, 1:5] = 'C6'
    edgecolors[5:9, 5:9] = 'C7'

    linewidths = np.full(z.shape, 2)
    linewidths[0::2, 0::2] = 5
    linewidths[1::2, 1::2] = 5

    x, y, z, sizes, facecolors, edgecolors, linewidths = [
        a.flatten()
        for a in [x, y, z, sizes, facecolors, edgecolors, linewidths]
    ]

    ax_ref = fig_ref.add_subplot(projection='3d')
    sets = (np.unique(a) for a in [sizes, facecolors, edgecolors, linewidths])
    for s, fc, ec, lw in itertools.product(*sets):
        subset = (
            (sizes != s) |
            (facecolors != fc) |
            (edgecolors != ec) |
            (linewidths != lw)
        )
        subset = np.ma.masked_array(z, subset, dtype=float)

        # When depth shading is disabled, the colors are passed through as
        # single-item lists; this triggers single path optimization. The
        # following reshaping is a hack to disable that, since the optimization
        # would not occur for the full scatter which has multiple colors.
        fc = np.repeat(fc, sum(~subset.mask))

        ax_ref.scatter(x, y, subset, s=s, fc=fc, ec=ec, lw=lw, alpha=1,
                       depthshade=depthshade)

    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.scatter(x, y, z, s=sizes, fc=facecolors, ec=edgecolors,
                    lw=linewidths, alpha=1, depthshade=depthshade)


@pytest.mark.parametrize('azim', [-50, 130])  # yellow first, blue first
@check_figures_equal(extensions=['png'])
def test_marker_draw_order_data_reversed(fig_test, fig_ref, azim):
    """
    Test that the draw order does not depend on the data point order.

    For the given viewing angle at azim=-50, the yellow marker should be in
    front. For azim=130, the blue marker should be in front.
    """
    x = [-1, 1]
    y = [1, -1]
    z = [0, 0]
    color = ['b', 'y']
    ax = fig_test.add_subplot(projection='3d')
    ax.scatter(x, y, z, s=3500, c=color)
    ax.view_init(elev=0, azim=azim, roll=0)
    ax = fig_ref.add_subplot(projection='3d')
    ax.scatter(x[::-1], y[::-1], z[::-1], s=3500, c=color[::-1])
    ax.view_init(elev=0, azim=azim, roll=0)


@check_figures_equal(extensions=['png'])
def test_marker_draw_order_view_rotated(fig_test, fig_ref):
    """
    Test that the draw order changes with the direction.

    If we rotate *azim* by 180 degrees and exchange the colors, the plot
    plot should look the same again.
    """
    azim = 130
    x = [-1, 1]
    y = [1, -1]
    z = [0, 0]
    color = ['b', 'y']
    ax = fig_test.add_subplot(projection='3d')
    # axis are not exactly invariant under 180 degree rotation -> deactivate
    ax.set_axis_off()
    ax.scatter(x, y, z, s=3500, c=color)
    ax.view_init(elev=0, azim=azim, roll=0)
    ax = fig_ref.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.scatter(x, y, z, s=3500, c=color[::-1])  # color reversed
    ax.view_init(elev=0, azim=azim - 180, roll=0)  # view rotated by 180 deg


@mpl3d_image_comparison(['plot_3d_from_2d.png'], tol=0.015)
def test_plot_3d_from_2d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = np.arange(0, 5)
    ys = np.arange(5, 10)
    ax.plot(xs, ys, zs=0, zdir='x')
    ax.plot(xs, ys, zs=0, zdir='y')


@mpl3d_image_comparison(['surface3d.png'])
def test_surface3d():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.hypot(X, Y)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rcount=40, ccount=40, cmap=cm.coolwarm,
                           lw=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)


@mpl3d_image_comparison(['surface3d_shaded.png'])
def test_surface3d_shaded():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5,
                    color=[0.25, 1, 0.25], lw=1, antialiased=False)
    ax.set_zlim(-1.01, 1.01)


@mpl3d_image_comparison(['surface3d_masked.png'])
def test_surface3d_masked():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y = [1, 2, 3, 4, 5, 6, 7, 8]

    x, y = np.meshgrid(x, y)
    matrix = np.array(
        [
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 1, 2, 3, 4, 4, 4, 3, 2, 1, 1],
            [-1, -1., 4, 5, 6, 8, 6, 5, 4, 3, -1.],
            [-1, -1., 7, 8, 11, 12, 11, 8, 7, -1., -1.],
            [-1, -1., 8, 9, 10, 16, 10, 9, 10, 7, -1.],
            [-1, -1., -1., 12, 16, 20, 16, 12, 11, -1., -1.],
            [-1, -1., -1., -1., 22, 24, 22, 20, 18, -1., -1.],
            [-1, -1., -1., -1., -1., 28, 26, 25, -1., -1., -1.],
        ]
    )
    z = np.ma.masked_less(matrix, 0)
    norm = mcolors.Normalize(vmax=z.max(), vmin=z.min())
    colors = mpl.colormaps["plasma"](norm(z))
    ax.plot_surface(x, y, z, facecolors=colors)
    ax.view_init(30, -80, 0)


@check_figures_equal(extensions=["png"])
def test_plot_surface_None_arg(fig_test, fig_ref):
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    z = x + y
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.plot_surface(x, y, z, facecolors=None)
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.plot_surface(x, y, z)


@mpl3d_image_comparison(['surface3d_masked_strides.png'])
def test_surface3d_masked_strides():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x, y = np.mgrid[-6:6.1:1, -6:6.1:1]
    z = np.ma.masked_less(x * y, 2)

    ax.plot_surface(x, y, z, rstride=4, cstride=4)
    ax.view_init(60, -45, 0)


@mpl3d_image_comparison(['text3d.png'], remove_text=False)
def test_text3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    xs = (2, 6, 4, 9, 7, 2)
    ys = (6, 4, 8, 7, 2, 2)
    zs = (4, 2, 5, 6, 1, 7)

    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
        ax.text(x, y, z, label, zdir)

    ax.text(1, 1, 1, "red", color='red')
    ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)
    ax.set_xlim3d(0, 10)
    ax.set_ylim3d(0, 10)
    ax.set_zlim3d(0, 10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


@check_figures_equal(extensions=['png'])
def test_text3d_modification(fig_ref, fig_test):
    # Modifying the Text position after the fact should work the same as
    # setting it directly.
    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    xs = (2, 6, 4, 9, 7, 2)
    ys = (6, 4, 8, 7, 2, 2)
    zs = (4, 2, 5, 6, 1, 7)

    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.set_xlim3d(0, 10)
    ax_test.set_ylim3d(0, 10)
    ax_test.set_zlim3d(0, 10)
    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        t = ax_test.text(0, 0, 0, f'({x}, {y}, {z}), dir={zdir}')
        t.set_position_3d((x, y, z), zdir=zdir)

    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.set_xlim3d(0, 10)
    ax_ref.set_ylim3d(0, 10)
    ax_ref.set_zlim3d(0, 10)
    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        ax_ref.text(x, y, z, f'({x}, {y}, {z}), dir={zdir}', zdir=zdir)


@mpl3d_image_comparison(['trisurf3d.png'], tol=0.061)
def test_trisurf3d():
    n_angles = 36
    n_radii = 8
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi/n_angles

    x = np.append(0, (radii*np.cos(angles)).flatten())
    y = np.append(0, (radii*np.sin(angles)).flatten())
    z = np.sin(-x*y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)


@mpl3d_image_comparison(['trisurf3d_shaded.png'], tol=0.03)
def test_trisurf3d_shaded():
    n_angles = 36
    n_radii = 8
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi/n_angles

    x = np.append(0, (radii*np.cos(angles)).flatten())
    y = np.append(0, (radii*np.sin(angles)).flatten())
    z = np.sin(-x*y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, z, color=[1, 0.5, 0], linewidth=0.2)


@mpl3d_image_comparison(['wireframe3d.png'])
def test_wireframe3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rcount=13, ccount=13)


@mpl3d_image_comparison(['wireframe3dzerocstride.png'])
def test_wireframe3dzerocstride():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rcount=13, ccount=0)


@mpl3d_image_comparison(['wireframe3dzerorstride.png'])
def test_wireframe3dzerorstride():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=0, cstride=10)


def test_wireframe3dzerostrideraises():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    with pytest.raises(ValueError):
        ax.plot_wireframe(X, Y, Z, rstride=0, cstride=0)


def test_mixedsamplesraises():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    with pytest.raises(ValueError):
        ax.plot_wireframe(X, Y, Z, rstride=10, ccount=50)
    with pytest.raises(ValueError):
        ax.plot_surface(X, Y, Z, cstride=50, rcount=10)


@mpl3d_image_comparison(['quiver3d.png'], style='mpl20')
def test_quiver3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pivots = ['tip', 'middle', 'tail']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (pivot, color) in enumerate(zip(pivots, colors)):
        x, y, z = np.meshgrid([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])
        u = -x
        v = -y
        w = -z
        # Offset each set in z direction
        z += 2 * i
        ax.quiver(x, y, z, u, v, w, length=1, pivot=pivot, color=color)
        ax.scatter(x, y, z, color=color)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-1, 5)


@check_figures_equal(extensions=["png"])
def test_quiver3d_empty(fig_test, fig_ref):
    fig_ref.add_subplot(projection='3d')
    x = y = z = u = v = w = []
    ax = fig_test.add_subplot(projection='3d')
    ax.quiver(x, y, z, u, v, w, length=0.1, pivot='tip', normalize=True)


@mpl3d_image_comparison(['quiver3d_masked.png'])
def test_quiver3d_masked():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Using mgrid here instead of ogrid because masked_where doesn't
    # seem to like broadcasting very much...
    x, y, z = np.mgrid[-1:0.8:10j, -1:0.8:10j, -1:0.6:3j]

    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (2/3)**0.5 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
    u = np.ma.masked_where((-0.4 < x) & (x < 0.1), u, copy=False)
    v = np.ma.masked_where((0.1 < y) & (y < 0.7), v, copy=False)

    ax.quiver(x, y, z, u, v, w, length=0.1, pivot='tip', normalize=True)


def test_patch_modification():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    circle = Circle((0, 0))
    ax.add_patch(circle)
    art3d.patch_2d_to_3d(circle)
    circle.set_facecolor((1.0, 0.0, 0.0, 1))

    assert mcolors.same_color(circle.get_facecolor(), (1, 0, 0, 1))
    fig.canvas.draw()
    assert mcolors.same_color(circle.get_facecolor(), (1, 0, 0, 1))


@check_figures_equal(extensions=['png'])
def test_patch_collection_modification(fig_test, fig_ref):
    # Test that modifying Patch3DCollection properties after creation works.
    patch1 = Circle((0, 0), 0.05)
    patch2 = Circle((0.1, 0.1), 0.03)
    facecolors = np.array([[0., 0.5, 0., 1.], [0.5, 0., 0., 0.5]])
    c = art3d.Patch3DCollection([patch1, patch2], linewidths=3)

    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.add_collection3d(c)
    c.set_edgecolor('C2')
    c.set_facecolor(facecolors)
    c.set_alpha(0.7)
    assert c.get_depthshade()
    c.set_depthshade(False)
    assert not c.get_depthshade()

    patch1 = Circle((0, 0), 0.05)
    patch2 = Circle((0.1, 0.1), 0.03)
    facecolors = np.array([[0., 0.5, 0., 1.], [0.5, 0., 0., 0.5]])
    c = art3d.Patch3DCollection([patch1, patch2], linewidths=3,
                                edgecolor='C2', facecolor=facecolors,
                                alpha=0.7, depthshade=False)

    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.add_collection3d(c)


def test_poly3dcollection_verts_validation():
    poly = [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match=r'list of \(N, 3\) array-like'):
        art3d.Poly3DCollection(poly)  # should be Poly3DCollection([poly])

    poly = np.array(poly, dtype=float)
    with pytest.raises(ValueError, match=r'list of \(N, 3\) array-like'):
        art3d.Poly3DCollection(poly)  # should be Poly3DCollection([poly])


@mpl3d_image_comparison(['poly3dcollection_closed.png'])
def test_poly3dcollection_closed():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    poly1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
    poly2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], float)
    c1 = art3d.Poly3DCollection([poly1], linewidths=3, edgecolor='k',
                                facecolor=(0.5, 0.5, 1, 0.5), closed=True)
    c2 = art3d.Poly3DCollection([poly2], linewidths=3, edgecolor='k',
                                facecolor=(1, 0.5, 0.5, 0.5), closed=False)
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
    ax.add_collection3d(c1

=======
            projM = proj3d._ortho_transformation(-self._dist, self._dist

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)
    ax.add_collection3d(c2)


def test_poly_collection_2d_to_3d_empty():
    poly = PolyCollection([])
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
    art3d.poly_collection_2d_to_3d(poly)

=======
            projM = proj3d._persp_transformation(-self._dist,

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
    assert isinstance(poly, art3d.Poly3DCollection

=======
                                                 self._dist,
                                                 self._focal_length

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)
    assert poly.get_paths() == []

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.add_artist(poly)
    minz = poly.do_3d_projection()
    assert np.isnan(minz)

    # Ensure drawing actually works.
    fig.canvas.draw()


@mpl3d_image_comparison(['poly3dcollection_alpha.png'])
def test_poly3dcollection_alpha():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    poly1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
    poly2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], float)
    c1 = art3d.Poly3DCollection([poly1], linewidths=3, edgecolor='k',
                                facecolor=(0.5, 0.5, 1), closed=True)
    c1.set_alpha(0.5)
    c2 = art3d.Poly3DCollection([poly2], linewidths=3, closed=False)
    # Post-creation modification should work.
    c2.set_facecolor((1, 0.5, 0.5))
    c2.set_edgecolor('k')
    c2.set_alpha(0.5)
    ax.add_collection3d(c1)
    ax.add_collection3d(c2)


@mpl3d_image_comparison(['add_collection3d_zs_array.png'])
def test_add_collection3d_zs_array():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    norm = plt.Normalize(0, 2*np.pi)
    # 2D LineCollection from x & y values
    lc = LineCollection(segments[:, :, :2], cmap='twilight', norm=norm)
    lc.set_array(np.mod(theta, 2*np.pi))
    # Add 2D collection at z values to ax
    line = ax.add_collection3d(lc, zs=segments[:, :, 2])

    assert line is not None

    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 6)
    ax.set_zlim(-2, 2)


@mpl3d_image_comparison(['add_collection3d_zs_scalar.png'])
def test_add_collection3d_zs_scalar():
    theta = np.linspace(0, 2 * np.pi, 100)
    z = 1
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    norm = plt.Normalize(0, 2*np.pi)
    lc = LineCollection(segments, cmap='twilight', norm=norm)
    lc.set_array(theta)
    line = ax.add_collection3d(lc, zs=z)

    assert line is not None

    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 6)
    ax.set_zlim(0, 2)


@mpl3d_image_comparison(['axes3d_labelpad.png'], remove_text=False)
def test_axes3d_labelpad():
    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    # labelpad respects rcParams
    assert ax.xaxis.labelpad == mpl.rcParams['axes.labelpad']
    # labelpad can be set in set_label
    ax.set_xlabel('X LABEL', labelpad=10)
    assert ax.xaxis.labelpad == 10
    ax.set_ylabel('Y LABEL')
    ax.set_zlabel('Z LABEL', labelpad=20)
    assert ax.zaxis.labelpad == 20
    assert ax.get_zlabel() == 'Z LABEL'
    # or manually
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = -40

    # Tick labels also respect tick.pad (also from rcParams)
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.set_pad(tick.get_pad() - i * 5)


@mpl3d_image_comparison(['axes3d_cla.png'], remove_text=False)
def test_axes3d_cla():
    # fixed in pull request 4553
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_axis_off()
    ax.cla()  # make sure the axis displayed is 3D (not 2D)


@mpl3d_image_comparison(['axes3d_rotated.png'], remove_text=False)
def test_axes3d_rotated():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(90, 45, 0)  # look down, rotated. Should be square


def test_plotsurface_1d_raises():
    x = np.linspace(0.5, 10, num=100)
    y = np.linspace(0.5, 10, num=100)
    X, Y = np.meshgrid(x, y)
    z = np.random.randn(100)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    with pytest.raises(ValueError):
        ax.plot_surface(X, Y, z)


def _test_proj_make_M():
    # eye point
    E = np.array([1000, -1000, 2000])
    R = np.array([100, 100, 100])
    V = np.array([0, 0, 1])
    roll = 0
    u, v, w = proj3d._view_axes(E, R, V, roll)
    viewM = proj3d._view_transformation_uvw(u, v, w, E)
    perspM = proj3d.persp_transformation(100, -100, 1)
    M = np.dot(perspM, viewM)
    return M


def test_proj_transform():
    M = _test_proj_make_M()

    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 300.0
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 300.0
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 300.0

    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)
    ixs, iys, izs = proj3d.inv_transform(txs, tys, tzs, M)

    np.testing.assert_almost_equal(ixs, xs)
    np.testing.assert_almost_equal(iys, ys)
    np.testing.assert_almost_equal(izs, zs)


def _test_proj_draw_axes(M, s=1, *args, **kwargs):
    xs = [0, s, 0, 0]
    ys = [0, 0, s, 0]
    zs = [0, 0, 0, s]
    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)
    o, ax, ay, az = zip(txs, tys)
    lines = [(o, ax), (o, ay), (o, az)]

    fig, ax = plt.subplots(*args, **kwargs)
    linec = LineCollection(lines)
    ax.add_collection(linec)
    for x, y, t in zip(txs, tys, ['o', 'x', 'y', 'z']):
        ax.text(x, y, t)

    return fig, ax


@mpl3d_image_comparison(['proj3d_axes_cube.png'])
def test_proj_axes_cube():
    M = _test_proj_make_M()

    ts = '0 1 2 3 0 4 5 6 7 4'.split()
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 300.0
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 300.0
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 300.0

    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)

    fig, ax = _test_proj_draw_axes(M, s=400)

    ax.scatter(txs, tys, c=tzs)
    ax.plot(txs, tys, c='r')
    for x, y, t in zip(txs, tys, ts):
        ax.text(x, y, t)

    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)


@mpl3d_image_comparison(['proj3d_axes_cube_ortho.png'])
def test_proj_axes_cube_ortho():
    E = np.array([200, 100, 100])
    R = np.array([0, 0, 0])
    V = np.array([0, 0, 1])
    roll = 0
    u, v, w = proj3d._view_axes(E, R, V, roll)
    viewM = proj3d._view_transformation_uvw(u, v, w, E)
    orthoM = proj3d.ortho_transformation(-1, 1)
    M = np.dot(orthoM, viewM)

    ts = '0 1 2 3 0 4 5 6 7 4'.split()
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 100
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 100
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 100

    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)

    fig, ax = _test_proj_draw_axes(M, s=150)

    ax.scatter(txs, tys, s=300-tzs)
    ax.plot(txs, tys, c='r')
    for x, y, t in zip(txs, tys, ts):
        ax.text(x, y, t)

    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)


def test_rot():
    V = [1, 0, 0, 1]
    rotated_V = proj3d.rot_x(V, np.pi / 6)
    np.testing.assert_allclose(rotated_V, [1, 0, 0, 1])

    V = [0, 1, 0, 1]
    rotated_V = proj3d.rot_x(V, np.pi / 6)
    np.testing.assert_allclose(rotated_V, [0, np.sqrt(3) / 2, 0.5, 1])


def test_world():
    xmin, xmax = 100, 120
    ymin, ymax = -100, 100
    zmin, zmax = 0.1, 0.2
    M = proj3d.world_transformation(xmin, xmax, ymin, ymax, zmin, zmax)
    np.testing.assert_allclose(M,
                               [[5e-2, 0, 0, -5],
                                [0, 5e-3, 0, 5e-1],
                                [0, 0, 1e1, -1],
                                [0, 0, 0, 1]])


@mpl3d_image_comparison(['proj3d_lines_dists.png'])
def test_lines_dists():
    fig, ax = plt.subplots(figsize=(4, 6), subplot_kw=dict(aspect='equal'))

    xs = (0, 30)
    ys = (20, 150)
    ax.plot(xs, ys)
    p0, p1 = zip(xs, ys)

    xs = (0, 0, 20, 30)
    ys = (100, 150, 30, 200)
    ax.scatter(xs, ys)

    dist0 = proj3d._line2d_seg_dist((xs[0], ys[0]), p0, p1)
    dist = proj3d._line2d_seg_dist(np.array((xs, ys)).T, p0, p1)
    assert dist0 == dist[0]

    for x, y, d in zip(xs, ys, dist):
        c = Circle((x, y), d, fill=0)
        ax.add_patch(c)

    ax.set_xlim(-50, 150)
    ax.set_ylim(0, 300)


def test_lines_dists_nowarning():
    # No RuntimeWarning must be emitted for degenerate segments, see GH#22624.
    s0 = (10, 30, 50)
    p = (20, 150, 180)
    proj3d._line2d_seg_dist(p, s0, s0)
    proj3d._line2d_seg_dist(np.array(p), s0, s0)


def test_autoscale():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    assert ax.get_zscale() == 'linear'
    ax.margins(x=0, y=.1, z=.2)
    ax.plot([0, 1], [0, 1], [0, 1])
    assert ax.get_w_lims() == (0, 1, -.1, 1.1, -.2, 1.2)
    ax.autoscale(False)
    ax.set_autoscalez_on(True)
    ax.plot([0, 2], [0, 2], [0, 2])
    assert ax.get_w_lims() == (0, 1, -.1, 1.1, -.4, 2.4)
    ax.autoscale(axis='x')
    ax.plot([0, 2], [0, 2], [0, 2])
    assert ax.get_w_lims() == (0, 2, -.1, 1.1, -.4, 2.4)


@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
@pytest.mark.parametrize('auto', (True, False, None))
def test_unautoscale(axis, auto):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = np.arange(100)
    y = np.linspace(-0.1, 0.1, 100)
    ax.scatter(x, y)

    get_autoscale_on = getattr(ax, f'get_autoscale{axis}_on')
    set_lim = getattr(ax, f'set_{axis}lim')
    get_lim = getattr(ax, f'get_{axis}lim')

    post_auto = get_autoscale_on() if auto is None else auto

    set_lim((-0.5, 0.5), auto=auto)
    assert post_auto == get_autoscale_on()
    fig.canvas.draw()
    np.testing.assert_array_equal(get_lim(), (-0.5, 0.5))


def test_axes3d_focal_length_checks():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    with pytest.raises(ValueError):
        ax.set_proj_type('persp', focal_length=0)
    with pytest.raises(ValueError):
        ax.set_proj_type('ortho', focal_length=1)


@mpl3d_image_comparison(['axes3d_focal_length.png'], remove_text=False)
def test_axes3d_focal_length():
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    axs[0].set_proj_type('persp', focal_length=np.inf)
    axs[1].set_proj_type('persp', focal_length=0.15)


@mpl3d_image_comparison(['axes3d_ortho.png'], remove_text=False)
def test_axes3d_ortho():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type('ortho')


@mpl3d_image_comparison(['axes3d_isometric.png'])
def test_axes3d_isometric():
    from itertools import combinations, product
    fig, ax = plt.subplots(subplot_kw=dict(
        projection='3d',
        proj_type='ortho',
        box_aspect=(4, 4, 4)
    ))
    r = (-1, 1)  # stackoverflow.com/a/11156353
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if abs(s - e).sum() == r[1] - r[0]:
            ax.plot3D(*zip(s, e), c='k')
    ax.view_init(elev=np.degrees(np.arctan(1. / np.sqrt(2))), azim=-45, roll=0)
    ax.grid(True)


@pytest.mark.parametrize('value', [np.inf, np.nan])
@pytest.mark.parametrize(('setter', 'side'), [
    ('set_xlim3d', 'left'),
    ('set_xlim3d', 'right'),
    ('set_ylim3d', 'bottom'),
    ('set_ylim3d', 'top'),
    ('set_zlim3d', 'bottom'),
    ('set_zlim3d', 'top'),
])
def test_invalid_axes_limits(setter, side, value):
    limit = {side: value}
    fig = plt.figure()
    obj = fig.add_subplot(projection='3d')
    with pytest.raises(ValueError):
        getattr(obj, setter)(**limit)


class TestVoxels:
    @mpl3d_image_comparison(['voxels-simple.png'])
    def test_simple(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((5, 4, 3))
        voxels = (x == y) | (y == z)
        ax.voxels(voxels)

    @mpl3d_image_comparison(['voxels-edge-style.png'])
    def test_edge_style(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((5, 5, 4))
        voxels = ((x - 2)**2 + (y - 2)**2 + (z-1.5)**2) < 2.2**2
        v = ax.voxels(voxels, linewidths=3, edgecolor='C1')

        # change the edge color of one voxel
        v[max(v.keys())].set_edgecolor('C2')
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py

    @mpl3d_image_comparison(['voxels-named-colors.png']

=======
        """
        # Get the axis limits and centers
        minx, maxx, miny, maxy, minz, maxz = self.get_w_lims(

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)
        cx = (maxx + minx)/2
        cy = (maxy + miny)/2
        cz = (maxz + minz
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
        """Test with colors set to a 3D object array of strings."""
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

=======
)/2

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py

<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
        x, y, z = np.indices((10, 10, 10)

=======
        # Scale the data range
        dx = (maxx - minx

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)*scale_x
        dy = (maxy - miny)
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
 | (y == z)

=======
*scale_y

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
        dz = (maxz - minz)*scale_z

        # Set the scaled axis limits
        self.set_xlim3d(cx - dx/2, cx + dx/2)
        self.set_ylim3d(cy - dy/2, cy + dy/2)
        self.set_zlim3d(cz - dz/2, cz + dz/2)

    def set_zlabel(self, zlabel, fontdict=None, labelpad=None, **kwargs):
        """
        Set zlabel.  See doc for `.set_ylabel` for description.
        """
        if labelpad is not None:
            self.zaxis.labelpad = labelpad
        return self.zaxis.set_label_text(zlabel, fontdict, **kwargs)

    def get_zlabel(self):
        """
        Get the z-label text string.
        """
        label = self.zaxis.get_label()
        return label.get_text()

    # Axes rectangle characteristics

    def get_frame_on(self):
        """Get whether the 3D axes panels are drawn."""
        return self._frameon

    def set_frame_on(self, b):
        """
        Set whether the 3D axes panels are drawn.

        Parameters
        ----------
        b : bool
        """
        self._frameon = bool(b)
        self.stale = True

    def grid(self, visible=True, **kwargs):
        """
        Set / unset 3D grid.

        .. note::

            Currently, this function does not behave the same as
            `.axes.Axes.grid`, but it is intended to eventually support that
            behavior.
        """
        # TODO: Operate on each axes separately
        if len(kwargs):
            visible = True
        self._draw_grid = visible
        self.stale = True

    def tick_params(self, axis='both', **kwargs):
        """
        Convenience method for changing the appearance of ticks and
        tick labels.

        See `.Axes.tick_params` for full documentation.  Because this function
        applies to 3D Axes, *axis* can also be set to 'z', and setting *axis*
        to 'both' autoscales all three axes.

        Also, because of how Axes3D objects are drawn very differently
        from regular 2D axes, some of these settings may have
        ambiguous meaning.  For simplicity, the 'z' axis will
        accept settings as if it was like the 'y' axis.

        .. note::
           Axes3D currently ignores some of these settings.
        """
        _api.check_in_list(['x', 'y', 'z', 'both'], axis=axis)
        if axis in ['x', 'y', 'both']:
            super().tick_params(axis, **kwargs)
        if axis in ['z', 'both']:
            zkw = dict(kwargs)
            zkw.pop('top', None)
            zkw.pop('bottom', None)
            zkw.pop('labeltop', None)
            zkw.pop('labelbottom', None)
            self.zaxis.set_tick_params(**zkw)

    # data limits, ticks, tick labels, and formatting

    def invert_zaxis(self):
        """
        Invert the z-axis.
        """
        bottom, top = self.get_zlim()
        self.set_zlim(top, bottom, auto=None)

    zaxis_inverted = _axis_method_wrapper("zaxis", "get_inverted")

    def get_zbound(self):
        """
        Return the lower and upper z-axis bounds, in increasing order.
        """
        bottom, top = self.get_zlim()
        if bottom < top:
            return bottom, top
        else:
            return top, bottom

    def set_zbound(self, lower=None, upper=None):
        """
        Set the lower and upper numerical bounds of the z-axis.

        This method will honor axes inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscalez_on()`).
        """
        if upper is None and np.iterable(lower):
            lower, upper = lower

        old_lower, old_upper = self.get_zbound()
        if lower is None:
            lower = old_lower
        if upper is None:
            upper = old_upper

        self.set_zlim(sorted((lower, upper),
                             reverse=bool(self.zaxis_inverted())),
                      auto=None)

    def text(self, x, y, z, s, zdir=None, **kwargs):
        """
        Add text to the plot.

        Keyword arguments will be passed on to `.Axes.text`, except for the
        *zdir* keyword, which sets the direction to be used as the z
        direction.
        """
        text = super().text(x, y, s, **kwargs)
        art3d.text_2d_to_3d(text, z, zdir)
        return text

    text3D = text
    text2D = Axes.text

    def plot(self, xs, ys, *args, zdir='z', **kwargs):
        """
        Plot 2D or 3D data.

        Parameters
        ----------
        xs : 1D array-like
            x coordinates of vertices.
        ys : 1D array-like
            y coordinates of vertices.
        zs : float or 1D array-like
            z coordinates of vertices; either one for all points or one for
            each point.
        zdir : {'x', 'y', 'z'}, default: 'z'
            When plotting 2D data, the direction to use as z.
        **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.plot`.
        """
        had_data = self.has_data()

        # `zs` can be passed positionally or as keyword; checking whether
        # args[0] is a string matches the behavior of 2D `plot` (via
        # `_process_plot_var_args`).
        if args and not isinstance(args[0], str):
            zs, *args = args
            if 'zs' in kwargs:
                raise TypeError("plot() for multiple values for argument 'z'")
        else:
            zs = kwargs.pop('zs', 0)

        # Match length
        zs = np.broadcast_to(zs, np.shape(xs))

        lines = super().plot(xs, ys, *args, **kwargs)
        for line in lines:
            art3d.line_2d_to_3d(line, zs=zs, zdir=zdir)

        xs, ys, zs = art3d.juggle_axes(xs, ys, zs, zdir)
        self.auto_scale_xyz(xs, ys, zs, had_data)
        return lines

    plot3D = plot

    def plot_surface(self, X, Y, Z, *, norm=None, vmin=None,
                     vmax=None, lightsource=None, **kwargs):
        """
        Create a surface plot.

        By default, it will be colored in shades of a solid color, but it also
        supports colormapping by supplying the *cmap* argument.

        .. note::

           The *rcount* and *ccount* kwargs, which both default to 50,
           determine the maximum number of samples used in each direction.  If
           the input data is larger, it will be downsampled (by slicing) to
           these numbers of points.

        .. note::

           To maximize rendering speed consider setting *rstride* and *cstride*
           to divisors of the number of rows minus 1 and columns minus 1
           respectively. For example, given 51 rows rstride can be any of the
           divisors of 50.

           Similarly, a setting of *rstride* and *cstride* equal to 1 (or
           *rcount* and *ccount* equal the number of rows and columns) can use
           the optimized path.

        Parameters
        ----------
        X, Y, Z : 2D arrays
            Data values.

        rcount, ccount : int
            Maximum number of samples used in each direction.  If the input
            data is larger, it will be downsampled (by slicing) to these
            numbers of points.  Defaults to 50.

        rstride, cstride : int
            Downsampling stride in each direction.  These arguments are
            mutually exclusive with *rcount* and *ccount*.  If only one of
            *rstride* or *cstride* is set, the other defaults to 10.

            'classic' mode uses a default of ``rstride = cstride = 10`` instead
            of the new default of ``rcount = ccount = 50``.

        color : color-like
            Color of the surface patches.

        cmap : Colormap
            Colormap of the surface patches.

        facecolors : array-like of colors.
            Colors of each individual patch.

        norm : Normalize
            Normalization for the colormap.

        vmin, vmax : float
            Bounds for the normalization.

        shade : bool, default: True
            Whether to shade the facecolors.  Shading is always disabled when
            *cmap* is specified.

        lightsource : `~matplotlib.colors.LightSource`
            The lightsource to use when *shade* is True.

        **kwargs
            Other keyword arguments are forwarded to `.Poly3DCollection`.
        """

        had_data = self.has_data()

        if Z.ndim != 2:
            raise ValueError("Argument Z must be 2-dimensional.")

        Z = cbook._to_unmasked_float_array(Z)
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        rows, cols = Z.shape

        has_stride = 'rstride' in kwargs or 'cstride' in kwargs
        has_count = 'rcount' in kwargs or 'ccount' in kwargs

        if has_stride and has_count:
            raise ValueError("Cannot specify both stride and count arguments")

        rstride = kwargs.pop('rstride', 10)
        cstride = kwargs.pop('cstride', 10)
        rcount = kwargs.pop('rcount', 50)
        ccount = kwargs.pop('ccount', 50)

        if mpl.rcParams['_internal.classic_mode']:
            # Strides have priority over counts in classic mode.
            # So, only compute strides from counts
            # if counts were explicitly given
            compute_strides = has_count
        else:
            # If the strides are provided then it has priority.
            # Otherwise, compute the strides from the counts.
            compute_strides = not has_stride

        if compute_strides:
            rstride = int(max(np.ceil(rows / rcount), 1))
            cstride = int(max(np.ceil(cols / ccount), 1))

        fcolors = kwargs.pop('facecolors', None)

        if fcolors is None:
            color = kwargs.pop('color', None)
            if color is None:
                color = self._get_lines.get_next_color()
            color = np.array(mcolors.to_rgba(color))

        cmap = kwargs.get('cmap', None)
        shade = kwargs.pop('shade', cmap is None)
        if shade is None:
            raise ValueError("shade cannot be None.")

        colset = []  # the sampled facecolor
        if (rows - 1) % rstride == 0 and \
           (cols - 1) % cstride == 0 and \
           fcolors is None:
            polys = np.stack(
                [cbook._array_patch_perimeters(a, rstride, cstride)
                 for a in (X, Y, Z)],
                axis=-1)
        else:
            # evenly spaced, and including both endpoints
            row_inds = list(range(0, rows-1, rstride)) + [rows-1]
            col_inds = list(range(0, cols-1, cstride)) + [cols-1]

            polys = []
            for rs, rs_next in zip(row_inds[:-1], row_inds[1:]):
                for cs, cs_next in zip(col_inds[:-1], col_inds[1:]):
                    ps = [
                        # +1 ensures we share edges between polygons
                        cbook._array_perimeter(a[rs:rs_next+1, cs:cs_next+1])
                        for a in (X, Y, Z)
                    ]
                    # ps = np.stack(ps, axis=-1)
                    ps = np.array(ps).T
                    polys.append(ps)

                    if fcolors is not None:
                        colset.append(fcolors[rs][cs])

        # In cases where there are NaNs in the data (possibly from masked
        # arrays), artifacts can be introduced. Here check whether NaNs exist
        # and remove the entries if so
        if not isinstance(polys, np.ndarray) or np.isnan(polys).any():
            new_polys = []
            new_colset = []

            # Depending on fcolors, colset is either an empty list or has as
            # many elements as polys. In the former case new_colset results in
            # a list with None entries, that is discarded later.
            for p, col in itertools.zip_longest(polys, colset):
                new_poly = np.array(p)[~np.isnan(p).any(axis=1)]
                if len(new_poly):
                    new_polys.append(new_poly)
                    new_colset.append(col)

            # Replace previous polys and, if fcolors is not None, colset
            polys = new_polys
            if fcolors is not None:
                colset = new_colset

        # note that the striding causes some polygons to have more coordinates
        # than others

        if fcolors is not None:
            polyc = art3d.Poly3DCollection(
                polys, edgecolors=colset, facecolors=colset, shade=shade,
                lightsource=lightsource, **kwargs)
        elif cmap:
            polyc = art3d.Poly3DCollection(polys, **kwargs)
            # can't always vectorize, because polys might be jagged
            if isinstance(polys, np.ndarray):
                avg_z = polys[..., 2].mean(axis=-1)
            else:
                avg_z = np.array([ps[:, 2].mean() for ps in polys])
            polyc.set_array(avg_z)
            if vmin is not None or vmax is not None:
                polyc.set_clim(vmin, vmax)
            if norm is not None:
                polyc.set_norm(norm)
        else:
            polyc = art3d.Poly3DCollection(
                polys, facecolors=color, shade=shade,
                lightsource=lightsource, **kwargs)

        self.add_collection(polyc)
        self.auto_scale_xyz(X, Y, Z, had_data)

        return polyc

    def plot_wireframe(self, X, Y, Z, **kwargs):
        """
        Plot a 3D wireframe.

        .. note::

           The *rcount* and *ccount* kwargs, which both default to 50,
           determine the maximum number of samples used in each direction.  If
           the input data is larger, it will be downsampled (by slicing) to
           these numbers of points.

        Parameters
        ----------
        X, Y, Z : 2D arrays
            Data values.

        rcount, ccount : int
            Maximum number of samples used in each direction.  If the input
            data is larger, it will be downsampled (by slicing) to these
            numbers of points.  Setting a count to zero causes the data to be
            not sampled in the corresponding direction, producing a 3D line
            plot rather than a wireframe plot.  Defaults to 50.

        rstride, cstride : int
            Downsampling stride in each direction.  These arguments are
            mutually exclusive with *rcount* and *ccount*.  If only one of
            *rstride* or *cstride* is set, the other defaults to 1.  Setting a
            stride to zero causes the data to be not sampled in the
            corresponding direction, producing a 3D line plot rather than a
            wireframe plot.

            'classic' mode uses a default of ``rstride = cstride = 1`` instead
            of the new default of ``rcount = ccount = 50``.

        **kwargs
            Other keyword arguments are forwarded to `.Line3DCollection`.
        """

        had_data = self.has_data()
        if Z.ndim != 2:
            raise ValueError("Argument Z must be 2-dimensional.")
        # FIXME: Support masked arrays
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        rows, cols = Z.shape

        has_stride = 'rstride' in kwargs or 'cstride' in kwargs
        has_count = 'rcount' in kwargs or 'ccount' in kwargs

        if has_stride and has_count:
            raise ValueError("Cannot specify both stride and count arguments")

        rstride = kwargs.pop('rstride', 1)
        cstride = kwargs.pop('cstride', 1)
        rcount = kwargs.pop('rcount', 50)
        ccount = kwargs.pop('ccount', 50)

        if mpl.rcParams['_internal.classic_mode']:
            # Strides have priority over counts in classic mode.
            # So, only compute strides from counts
            # if counts were explicitly given
            if has_count:
                rstride = int(max(np.ceil(rows / rcount), 1)) if rcount else 0
                cstride = int(max(np.ceil(cols / ccount), 1)) if ccount else 0
        else:
            # If the strides are provided then it has priority.
            # Otherwise, compute the strides from the counts.
            if not has_stride:
                rstride = int(max(np.ceil(rows / rcount), 1)) if rcount else 0
                cstride = int(max(np.ceil(cols / ccount), 1)) if ccount else 0

        # We want two sets of lines, one running along the "rows" of
        # Z and another set of lines running along the "columns" of Z.
        # This transpose will make it easy to obtain the columns.
        tX, tY, tZ = np.transpose(X), np.transpose(Y), np.transpose(Z)

        if rstride:
            rii = list(range(0, rows, rstride))
            # Add the last index only if needed
            if rows > 0 and rii[-1] != (rows - 1):
                rii += [rows-1]
        else:
            rii = []
        if cstride:
            cii = list(range(0, cols, cstride))
            # Add the last index only if needed
            if cols > 0 and cii[-1] != (cols - 1):
                cii += [cols-1]
        else:
            cii = []

        if rstride == 0 and cstride == 0:
            raise ValueError("Either rstride or cstride must be non zero")

        # If the inputs were empty, then just
        # reset everything.
        if Z.size == 0:
            rii = []
            cii = []

        xlines = [X[i] for i in rii]
        ylines = [Y[i] for i in rii]
        zlines = [Z[i] for i in rii]

        txlines = [tX[i] for i in cii]
        tylines = [tY[i] for i in cii]
        tzlines = [tZ[i] for i in cii]

        lines = ([list(zip(xl, yl, zl))
                 for xl, yl, zl in zip(xlines, ylines, zlines)]
                 + [list(zip(xl, yl, zl))
                 for xl, yl, zl in zip(txlines, tylines, tzlines)])

        linec = art3d.Line3DCollection(lines, **kwargs)
        self.add_collection(linec)
        self.auto_scale_xyz(X, Y, Z, had_data)

        return linec

    def plot_trisurf(self, *args, color=None, norm=None, vmin=None, vmax=None,
                     lightsource=None, **kwargs):
        """
        Plot a triangulated surface.

        The (optional) triangulation can be specified in one of two ways;
        either::

          plot_trisurf(triangulation, ...)

        where triangulation is a `~matplotlib.tri.Triangulation` object, or::

          plot_trisurf(X, Y, ...)
          plot_trisurf(X, Y, triangles, ...)
          plot_trisurf(X, Y, triangles=triangles, ...)

        in which case a Triangulation object will be created.  See
        `.Triangulation` for an explanation of these possibilities.

        The remaining arguments are::

          plot_trisurf(..., Z)

        where *Z* is the array of values to contour, one per point
        in the triangulation.

        Parameters
        ----------
        X, Y, Z : array-like
            Data values as 1D arrays.
        color
            Color of the surface patches.
        cmap
            A colormap for the surface patches.
        norm : Normalize
            An instance of Normalize to map values to colors.
        vmin, vmax : float, default: None
            Minimum and maximum value to map.
        shade : bool, default: True
            Whether to shade the facecolors.  Shading is always disabled when
            *cmap* is specified.
        lightsource : `~matplotlib.colors.LightSource`
            The lightsource to use when *shade* is True.
        **kwargs
            All other keyword arguments are passed on to
            :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`

        Examples
        --------
        .. plot:: gallery/mplot3d/trisurf3d.py
        .. plot:: gallery/mplot3d/trisurf3d_2.py
        """

        had_data = self.has_data()

        # TODO: Support custom face colours
        if color is None:
            color = self._get_lines.get_next_color()
        color = np.array(mcolors.to_rgba(color))

        cmap = kwargs.get('cmap', None)
        shade = kwargs.pop('shade', cmap is None)

        tri, args, kwargs = \
            Triangulation.get_from_args_and_kwargs(*args, **kwargs)
        try:
            z = kwargs.pop('Z')
        except KeyError:
            # We do this so Z doesn't get passed as an arg to PolyCollection
            z, *args = args
        z = np.asarray(z)

        triangles = tri.get_masked_triangles()
        xt = tri.x[triangles]
        yt = tri.y[triangles]
        zt = z[triangles]
        verts = np.stack((xt, yt, zt), axis=-1)

        if cmap:
            polyc = art3d.Poly3DCollection(verts, *args, **kwargs)
            # average over the three points of each triangle
            avg_z = verts[:, :, 2].mean(axis=1)
            polyc.set_array(avg_z)
            if vmin is not None or vmax is not None:
                polyc.set_clim(vmin, vmax)
            if norm is not None:
                polyc.set_norm(norm)
        else:
            polyc = art3d.Poly3DCollection(
                verts, *args, shade=shade, lightsource=lightsource,
                facecolors=color, **kwargs)

        self.add_collection(polyc)
        self.auto_scale_xyz(tri.x, tri.y, z, had_data)

        return polyc

    def _3d_extend_contour(self, cset, stride=5):
        """
        Extend a contour in 3D by creating
        """

        levels = cset.levels
        colls = cset.collections
        dz = (levels[1] - levels[0]) / 2

        for z, linec in zip(levels, colls):
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
    x, y, z = np.array(list(itertools.product(*[np.arange(0, 5, 1

=======
            paths = linec.get_paths(

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
                                                np.arange(0, 5, 1

=======

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
            if not paths
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
                                                np.arange(0, 5, 1

=======
:

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
                continue
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
).T
    c = x + y

=======
            topverts = art3d._paths_to_3d_segments(paths, z - dz)

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
            botverts = art3d._paths_to_3d_segments(paths, z + dz)

            color = linec.get_edgecolor()[0]

            nsteps = round(len(topverts[0]) / stride)
            if nsteps <= 1:
                if len(topverts[0]) > 1:
                    nsteps = 2
                else:
                    continue

            polyverts = []
            stepsize = (len(topverts[0]) - 1) / (nsteps - 1)
            for i in range(round(nsteps) - 1):
                i1 = round(i * stepsize)
                i2 = round((i + 1) * stepsize)
                polyverts.append([topverts[0][i1],
                                  topverts[0][i2],
                                  botverts[0][i2],
                                  botverts[0][i1]])

            # all polygons have 4 vertices, so vectorize
            polyverts = np.array(polyverts)
            polycol = art3d.Poly3DCollection(polyverts,
                                             facecolors=color,
                                             edgecolors=color,
                                             shade=True)
            polycol.set_sort_zpos(z)
            self.add_collection3d(polycol)

        for col in colls:
            col.remove()

    def add_contour_set(
            self, cset, extend3d=False, stride=5, zdir='z', offset=None):
        zdir = '-' + zdir
        if extend3d:
            self._3d_extend_contour(cset, stride)
        else:
            for z, linec in zip(cset.levels, cset.collections):
                if offset is not None:
                    z = offset
                art3d.line_collection_2d_to_3d(linec, z, zdir=zdir)

    def add_contourf_set(self, cset, zdir='z', offset=None):
        self._add_contourf_set(cset, zdir=zdir, offset=offset)

    def _add_contourf_set(self, cset, zdir='z', offset=None):
        """
        Returns
        -------
        levels : `numpy.ndarray`
            Levels at which the filled contours are added.
        """
        zdir = '-' + zdir

        midpoints = cset.levels[:-1] + np.diff(cset.levels) / 2
        # Linearly interpolate to get levels for any extensions
        if cset._extend_min:
            min_level = cset.levels[0] - np.diff(cset.levels[:2]) / 2
            midpoints = np.insert(midpoints, 0, min_level)
        if cset._extend_max:
            max_level = cset.levels[-1] + np.diff(cset.levels[-2:]) / 2
            midpoints = np.append(midpoints, max_level)

        for z, linec in zip(midpoints, cset.collections):
            if offset is not None:
                z = offset
            art3d.poly_collection_2d_to_3d(linec, z, zdir=zdir)
            linec.set_sort_zpos(z)
        return midpoints

    @_preprocess_data()
    def contour(self, X, Y, Z, *args,
                extend3d=False, stride=5, zdir='z', offset=None, **kwargs):
        """
        Create a 3D contour plot.

        Parameters
        ----------
        X, Y, Z : array-like,
            Input data. See `.Axes.contour` for supported data shapes.
        extend3d : bool, default: False
            Whether to extend contour in 3D.
        stride : int
            Step size for extending contour.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
        offset : float, optional
            If specified, plot a projection of the contour lines at this
            position in a plane normal to *zdir*.
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        *args, **kwargs
            Other arguments are forwarded to `matplotlib.axes.Axes.contour`.

        Returns
        -------
        matplotlib.contour.QuadContourSet
        """
        had_data = self.has_data()

        jX, jY, jZ = art3d.rotate_axes(X, Y, Z, zdir)
        cset = super().contour(jX, jY, jZ, *args, **kwargs)
        self.add_contour_set(cset, extend3d, stride, zdir, offset)

        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset

    contour3D = contour

    @_preprocess_data()
    def tricontour(self, *args,
                   extend3d=False, stride=5, zdir='z', offset=None, **kwargs):
        """
        Create a 3D contour plot.

        .. note::
            This method currently produces incorrect output due to a
            longstanding bug in 3D PolyCollection rendering.

        Parameters
        ----------
        X, Y, Z : array-like
            Input data. See `.Axes.tricontour` for supported data shapes.
        extend3d : bool, default: False
            Whether to extend contour in 3D.
        stride : int
            Step size for extending contour.
        zdir : {'x', 'y', 'z'}, default: 'z'
            The direction to use.
     (ValueError, (-1,), {}, r'margin must be greater than -0\.5'),
        (ValueError, (1, -1, 1), {}, r'margin must be greater than -0\.5'),
        (ValueError, (1, 1, -1), {}, r'margin must be greater than -0\.5'),
        (ValueError, tuple(), {'x': -1}, r'margin must be greater than -0\.5'),
        (ValueError, tuple(), {'y': -1}, r'margin must be greater than -0\.5'),
        (ValueError, tuple(), {'z': -1}, r'margin must be greater than -0\.5'),
        (TypeError, (1, ), {'x': 1},
         'Cannot pass both positional and keyword'),
        (TypeError, (1, ), {'x': 1, 'y': 1, 'z': 1},
         'Cannot pass both positional and keyword'),
        (TypeError, (1, ), {'x': 1, 'y': 1},
         'Cannot pass both positional and keyword'),
        (TypeError, (1, 1), {}, 'Must pass a single positional argument for'),
))
def test_margins_errors(err, args, kwargs, match):
    with pytest.raises(err, match=match):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.margins(*args, **kwargs)


@check_figures_equal(extensions=["png"])
def test_text_3d(fig_test, fig_ref):
    ax = fig_ref.add_subplot(projection="3d")
    txt = Text(0.5, 0.5, r'Foo bar $\int$')
    art3d.text_2d_to_3d(txt, z=1)
    ax.add_artist(txt)
    assert txt.get_position_3d() == (0.5, 0.5, 1)

    ax = fig_test.add_subplot(projection="3d")
    t3d = art3d.Text3D(0.5, 0.5, 1, r'Foo bar $\int$')
    ax.add_artist(t3d)
    assert t3d.get_position_3d() == (0.5, 0.5, 1)


def test_draw_single_lines_from_Nx1():
    # Smoke test for GH#23459
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot([[0], [1]], [[0], [1]], [[0], [1]])


@check_figures_equal(extensions=["png"])
def test_pathpatch_3d(fig_test, fig_ref):
    ax = fig_ref.add_subplot(projection="3d")
    path = Path.unit_rectangle()
    patch = PathPatch(path)
    art3d.pathpatch_2d_to_3d(patch, z=(0, 0.5, 0.7, 1, 0), zdir='y')
    ax.add_artist(patch)

    ax = fig_test.add_subplot(projection="3d")
    pp3d = art3d.PathPatch3D(path, zs=(0, 0.5, 0.7, 1, 0), zdir='y')
    ax.add_artist(pp3d)


@image_comparison(baseline_images=['scatter_spiral.png'],
                  remove_text=True,
                  style='default')
def test_scatter_spiral():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    th = np.linspace(0, 2 * np.pi * 6, 256)
    sc = ax.scatter(np.sin(th), np.cos(th), th, s=(1 + th * 5), c=th ** 2)

    # force at least 1 draw!
    fig.canvas.draw()


def test_Poly3DCollection_get_facecolor():
    # Smoke test to see that get_facecolor does not raise
    # See GH#4067
    y, x = np.ogrid[1:10:100j, 1:10:100j]
    z2 = np.cos(x) ** 3 - np.sin(y) ** 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = ax.plot_surface(x, y, z2, cmap='hot')
    r.get_facecolor()


def test_Poly3DCollection_get_edgecolor():
    # Smoke test to see that get_edgecolor does not raise
    # See GH#4067
    y, x = np.ogrid[1:10:100j, 1:10:100j]
    z2 = np.cos(x) ** 3 - np.sin(y) ** 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = ax.plot_surface(x, y, z2, cmap='hot')
    r.get_edgecolor()


@pytest.mark.parametrize(
    "vertical_axis, proj_expected, axis_lines_expected, tickdirs_expected",
    [
        (
            "z",
            [
                [0.0, 1.142857, 0.0, -0.571429],
                [0.0, 0.0, 0.857143, -0.428571],
                [0.0, 0.0, 0.0, -10.0],
                [-1.142857, 0.0, 0.0, 10.571429],
            ],
            [
                ([0.05617978, 0.06329114], [-0.04213483, -0.04746835]),
                ([-0.06329114, 0.06329114], [-0.04746835, -0.04746835]),
                ([-0.06329114, -0.06329114], [-0.04746835, 0.04746835]),
            ],
            [1, 0, 0],
        ),
        (
            "y",
            [
                [1.142857, 0.0, 0.0, -0.571429],
                [0.0, 0.857143, 0.0, -0.428571],
                [0.0, 0.0, 0.0, -10.0],
                [0.0, 0.0, -1.142857, 10.571429],
            ],
            [
                ([-0.06329114, 0.06329114], [0.04746835, 0.04746835]),
                ([0.06329114, 0.06329114], [-0.04746835, 0.04746835]),
                ([-0.05617978, -0.06329114], [0.04213483, 0.04746835]),
            ],
            [2, 2, 0],
        ),
        (
            "x",
            [
                [0.0, 0.0, 1.142857, -0.571429],
                [0.857143, 0.0, 0.0, -0.428571],
                [0.0, 0.0, 0.0, -10.0],
                [0.0, -1.142857, 0.0, 10.571429],
            ],
            [
                ([-0.06329114, -0.06329114], [0.04746835, -0.04746835]),
                ([0.06329114, 0.05617978], [0.04746835, 0.04213483]),
                ([0.06329114, -0.06329114], [0.04746835, 0.04746835]),
            ],
            [1, 2, 1],
        ),
    ],
)
def test_view_init_vertical_axis(
    vertical_axis, proj_expected, axis_lines_expected, tickdirs_expected
):
    """
    Test the actual projection, axis lines and ticks matches expected values.

    Parameters
    ----------
    vertical_axis : str
        Axis to align vertically.
    proj_expected : ndarray
        Expected values from ax.get_proj().
    axis_lines_expected : tuple of arrays
        Edgepoints of the axis line. Expected values retrieved according
        to ``ax.get_[xyz]axis().line.get_data()``.
    tickdirs_expected : list of int
        indexes indicating which axis to create a tick line along.
    """
    rtol = 2e-06
    ax = plt.subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=0, azim=0, roll=0, vertical_axis=vertical_axis)
    ax.figure.canvas.draw()

    # Assert the projection matrix:
    proj_actual = ax.get_proj()
    np.testing.assert_allclose(proj_expected, proj_actual, rtol=rtol)

    for i, axis in enumerate([ax.get_xaxis(), ax.get_yaxis(), ax.get_zaxis()]):
        # Assert black lines are correctly aligned:
        axis_line_expected = axis_lines_expected[i]
        axis_line_actual = axis.line.get_data()
        np.testing.assert_allclose(axis_line_expected, axis_line_actual,
                                   rtol=rtol)

        # Assert ticks are correctly aligned:
        tickdir_expected = tickdirs_expected[i]
        tickdir_actual = axis._get_tickdir()
        np.testing.assert_array_equal(tickdir_expected, tickdir_actual)


@image_comparison(baseline_images=['arc_pathpatch.png'],
                  remove_text=True,
                  style='default')
def test_arc_pathpatch():
    ax = plt.subplot(1, 1, 1, projection="3d")
    a = mpatch.Arc((0.5, 0.5), width=0.5, height=0.9,
                   angle=20, theta1=10, theta2=130)
    ax.add_patch(a)
    art3d.pathpatch_2d_to_3d(a, z=0, zdir='z')


@image_comparison(baseline_images=['panecolor_rcparams.png'],
                  remove_text=True,
                  style='mpl20')
def test_panecolor_rcparams():
    with plt.rc_context({'axes3d.xaxis.panecolor': 'r',
                         'axes3d.yaxis.panecolor': 'g',
                         'axes3d.zaxis.panecolor': 'b'}):
        fig = plt.figure(figsize=(1, 1))
        fig.add_subplot(projection='3d')


@check_figures_equal(extensions=["png"])
def test_mutating_input_arrays_y_and_z(fig_test, fig_ref):
    """
    Test to see if the `z` axis does not get mutated
    after a call to `Axes3D.plot`

    test cases came from GH#8990
    """
    ax1 = fig_test.add_subplot(111, projection='3d')
    x = [1, 2, 3]
    y = [0.0, 0.0, 0.0]
    z = [0.0, 0.0, 0.0]
    ax1.plot(x, y, z, 'o-')

    # mutate y,z to get a nontrivial line
    y[:] = [1, 2, 3]
    z[:] = [1, 2, 3]

    # draw the same plot without mutating x and y
    ax2 = fig_ref.add_subplot(111, projection='3d')
    x = [1, 2, 3]
    y = [0.0, 0.0, 0.0]
    z = [0.0, 0.0, 0.0]
    ax2.plot(x, y, z, 'o-')
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
      When plotting 2D data, the direction to use as z ('x', 'y' or 'z').
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs
            Other keyword arguments are forwarded to
            `matplotlib.axes.Axes.bar`.

        Returns
        -------
        mpl_toolkits.mplot3d.art3d.Patch3DCollection
        """
        had_data = self.has_data()

        patches = super().bar(left, height, *args, **kwargs)

        zs = np.broadcast_to(zs, len(left))

        verts = []
        verts_zs = []
        for p, z in zip(patches, zs):
            vs = art3d._get_patch_verts(p)
            verts += vs.tolist()
            verts_zs += [z] * len(vs)
            art3d.patch_2d_to_3d(p, z, zdir)
            if 'alpha' in kwargs:
                p.set_alpha(kwargs['alpha'])

        if len(verts) > 0:
            # the following has to be skipped if verts is empty
            # NOTE: Bugs could still occur if len(verts) > 0,
            #       but the "2nd dimension" is empty.
            xs, ys = zip(*verts)
        else:
            xs, ys = [], []

        xs, ys, verts_zs = art3d.juggle_axes(xs, ys, verts_zs, zdir)
        self.auto_scale_xyz(xs, ys, verts_zs, had_data)

        return patches

    @_preprocess_data()
    def bar3d(self, x, y, z, dx, dy, dz, color=None,
              zsort='average', shade=True, lightsource=None, *args, **kwargs):
        """
        Generate a 3D barplot.

        This method creates three-dimensional barplot where the width,
        depth, height, and color of the bars can all be uniquely set.

        Parameters
        ----------
        x, y, z : array-like
            The coordinates of the anchor point of the bars.

        dx, dy, dz : float or array-like
            The width, depth, and height of the bars, respectively.

        color : sequence of colors, optional
            The color of the bars can be specified globally or
            individually. This parameter can be:

            - A single color, to color all bars the same color.
            - An array of colors of length N bars, to color each bar
              independently.
            - An array of colors of length 6, to color the faces of the
              bars similarly.
            - An array of colors of length 6 * N bars, to color each face
              independently.

            When coloring the faces of the boxes specifically, this is
            the order of the coloring:

            1. -Z (bottom of box)
            2. +Z (top of box)
            3. -Y
            4. +Y
            5. -X
            6. +X

        zsort : str, optional
            The z-axis sorting scheme passed onto `~.art3d.Poly3DCollection`

        shade : bool, default: True
            When true, this shades the dark sides of the bars (relative
            to the plot's source of light).

        lightsource : `~matplotlib.colors.LightSource`
            The lightsource to use when *shade* is True.

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Any additional keyword arguments are passed onto
            `~.art3d.Poly3DCollection`.

        Returns
        -------
        collection : `~.art3d.Poly3DCollection`
            A collection of three-dimensional polygons representing the bars.
        """

        had_data = self.has_data()

        x, y, z, dx, dy, dz = np.broadcast_arrays(
            np.atleast_1d(x), y, z, dx, dy, dz)
        minx = np.min(x)
        maxx = np.max(x + dx)
        miny = np.min(y)
        maxy = np.max(y + dy)
        minz = np.min(z)
        maxz = np.max(z + dz)

        # shape (6, 4, 3)
        # All faces are oriented facing outwards - when viewed from the
        # outside, their vertices are in a counterclockwise ordering.
        cuboid = np.array([
            # -z
            (
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (1, 0, 0),
            ),
            # +z
            (
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            ),
            # -y
            (
                (0, 0, 0),
                (1, 0, 0),
                (1, 0, 1),
                (0, 0, 1),
            ),
            # +y
            (
                (0, 1, 0),
                (0, 1, 1),
                (1, 1, 1),
                (1, 1, 0),
            ),
            # -x
            (
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (0, 1, 0),
            ),
            # +x
            (
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 1),
                (1, 0, 1),
            ),
        ])

        # indexed by [bar, face, vertex, coord]
        polys = np.empty(x.shape + cuboid.shape)

        # handle each coordinate separately
        for i, p, dp in [(0, x, dx), (1, y, dy), (2, z, dz)]:
            p = p[..., np.newaxis, np.newaxis]
            dp = dp[..., np.newaxis, np.newaxis]
            polys[..., i] = p + dp * cuboid[..., i]

        # collapse the first two axes
        polys = polys.reshape((-1,) + polys.shape[2:])

        facecolors = []
        if color is None:
            color = [self._get_patches_for_fill.get_next_color()]

        color = list(mcolors.to_rgba_array(color))

        if len(color) == len(x):
            # bar colors specified, need to expand to number of faces
            for c in color:
                facecolors.extend([c] * 6)
        else:
            # a single color specified, or face colors specified explicitly
            facecolors = color
            if len(facecolors) < len(x):
                facecolors *= (6 * len(x))

        col = art3d.Poly3DCollection(polys,
                                     zsort=zsort,
                                     facecolors=facecolors,
                                     shade=shade,
                                     lightsource=lightsource,
                                     *args, **kwargs)
        self.add_collection(col)

        self.auto_scale_xyz((minx, maxx), (miny, maxy), (minz, maxz), had_data)

        return col

    def set_title(self, label, fontdict=None, loc='center', **kwargs):
        # docstring inherited
        ret = super().set_title(label, fontdict=fontdict, loc=loc, **kwargs)
        (x, y) = self.title.get_position()
        self.title.set_y(0.92 * y)
        return ret

    @_preprocess_data()
    def quiver(self, X, Y, Z, U, V, W, *,
               length=1, arrow_length_ratio=.3, pivot='tail', normalize=False,
               **kwargs):
        """
        Plot a 3D field of arrows.

        The arguments can be array-like or scalars, so long as they can be
        broadcast together. The arguments can also be masked arrays. If an
        element in any of argument is masked, then that corresponding quiver
        element will not be plotted.

        Parameters
        ----------
        X, Y, Z : array-like
            The x, y and z coordinates of the arrow locations (default is
            tail of arrow; see *pivot* kwarg).

        U, V, W : array-like
            The x, y and z components of the arrow vectors.

        length : float, default: 1
            The length of each quiver.

        arrow_length_ratio : float, default: 0.3
            The ratio of the arrow head with respect to the quiver.

        pivot : {'tail', 'middle', 'tip'}, default: 'tail'
            The part of the arrow that is at the grid point; the arrow
            rotates about this point, hence the name *pivot*.

        normalize : bool, default: False
            Whether all arrows are normalized to have the same length, or keep
            the lengths defined by *u*, *v*, and *w*.

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Any additional keyword arguments are delegated to
            :class:`.Line3DCollection`
        """

        def calc_arrows(UVW):
            # get unit direction vector perpendicular to (u, v, w)
            x = UVW[:, 0]
            y = UVW[:, 1]
            norm = np.linalg.norm(UVW[:, :2], axis=1)
            x_p = np.divide(y, norm, where=norm != 0, out=np.zeros_like(x))
            y_p = np.divide(-x,  norm, where=norm != 0, out=np.ones_like(x))
            # compute the two arrowhead direction unit vectors
            rangle = math.radians(15)
            c = math.cos(rangle)
            s = math.sin(rangle)
            # construct the rotation matrices of shape (3, 3, n)
            r13 = y_p * s
            r32 = x_p * s
            r12 = x_p * y_p * (1 - c)
            Rpos = np.array(
                [[c + (x_p ** 2) * (1 - c), r12, r13],
                 [r12, c + (y_p ** 2) * (1 - c), -r32],
                 [-r13, r32, np.full_like(x_p, c)]])
            # opposite rotation negates all the sin terms
            Rneg = Rpos.copy()
            Rneg[[0, 1, 2, 2], [2, 2, 0, 1]] *= -1
            # Batch n (3, 3) x (3) matrix multiplications ((3, 3, n) x (n, 3)).
            Rpos_vecs = np.einsum("ij...,...j->...i", Rpos, UVW)
            Rneg_vecs = np.einsum("ij...,...j->...i", Rneg, UVW)
            # Stack into (n, 2, 3) result.
            return np.stack([Rpos_vecs, Rneg_vecs], axis=1)

        had_data = self.has_data()

        input_args = [X, Y, Z, U, V, W]

        # extract the masks, if any
        masks = [k.mask for k in input_args
                 if isinstance(k, np.ma.MaskedArray)]
        # broadcast to match the shape
        bcast = np.broadcast_arrays(*input_args, *masks)
        input_args = bcast[:6]
        masks = bcast[6:]
        if masks:
            # combine the masks into one
            mask = functools.reduce(np.logical_or, masks)
            # put mask on and compress
            input_args = [np.ma.array(k, mask=mask).compressed()
                          for k in input_args]
        else:
            input_args = [np.ravel(k) for k in input_args]

        if any(len(v) == 0 for v in input_args):
            # No quivers, so just make an empty collection and return early
            linec = art3d.Line3DCollection([], **kwargs)
            self.add_collection(linec)
            return linec

        shaft_dt = np.array([0., length], dtype=float)
        arrow_dt = shaft_dt * arrow_length_ratio

        _api.check_in_list(['tail', 'middle', 'tip'], pivot=pivot)
        if pivot == 'tail':
            shaft_dt -= length
        elif pivot == 'middle':
            shaft_dt -= length / 2

        XYZ = np.column_stack(input_args[:3])
        UVW = np.column_stack(input_args[3:]).astype(float)

        # Normalize rows of UVW
        norm = np.linalg.norm(UVW, axis=1)

        # If any row of UVW is all zeros, don't make a quiver for it
        mask = norm > 0
        XYZ = XYZ[mask]
        if normalize:
            UVW = UVW[mask] / norm[mask].reshape((-1, 1))
        else:
            UVW = UVW[mask]

        if len(XYZ) > 0:
            # compute the shaft lines all at once with an outer product
            shafts = (XYZ - np.multiply.outer(shaft_dt, UVW)).swapaxes(0, 1)
            # compute head direction vectors, n heads x 2 sides x 3 dimensions
            head_dirs = calc_arrows(UVW)
            # compute all head lines at once, starting from the shaft ends
            heads = shafts[:, :1] - np.multiply.outer(arrow_dt, head_dirs)
            # stack left and right head lines together
            heads = heads.reshape((len(arrow_dt), -1, 3))
            # transpose to get a list of lines
            heads = heads.swapaxes(0, 1)

            lines = [*shafts, *heads]
        else:
            lines = []

        linec = art3d.Line3DCollection(lines, **kwargs)
        self.add_collection(linec)

        self.auto_scale_xyz(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], had_data)

        return linec

    quiver3D = quiver

    def voxels(self, *args, facecolors=None, edgecolors=None, shade=True,
               lightsource=None, **kwargs):
        """
        ax.voxels([x, y, z,] /, filled, facecolors=None, edgecolors=None, \
**kwargs)

        Plot a set of filled voxels

        All voxels are plotted as 1x1x1 cubes on the axis, with
        ``filled[0, 0, 0]`` placed with its lower corner at the origin.
        Occluded faces are not plotted.

        Parameters
        ----------
        filled : 3D np.array of bool
            A 3D array of values, with truthy values indicating which voxels
            to fill

        x, y, z : 3D np.array, optional
            The coordinates of the corners of the voxels. This should broadcast
            to a shape one larger in every dimension than the shape of
            *filled*.  These can be used to plot non-cubic voxels.

            If not specified, defaults to increasing integers along each axis,
            like those returned by :func:`~numpy.indices`.
            As indicated by the ``/`` in the function signature, these
            arguments can only be passed positionally.

        facecolors, edgecolors : array-like, optional
            The color to draw the faces and edges of the voxels. Can only be
            passed as keyword arguments.
            These parameters can be:

            - A single color value, to color all voxels the same color. This
              can be either a string, or a 1D rgb/rgba array
            - ``None``, the default, to use a single color for the faces, and
              the style default for the edges.
            - A 3D `~numpy.ndarray` of color names, with each item the color
              for the corresponding voxel. The size must match the voxels.
            - A 4D `~numpy.ndarray` of rgb/rgba data, with the components
              along the last axis.

        shade : bool, default: True
            Whether to shade the facecolors.

        lightsource : `~matplotlib.colors.LightSource`
            The lightsource to use when *shade* is True.

        **kwargs
            Additional keyword arguments to pass onto
            `~mpl_toolkits.mplot3d.art3d.Poly3DCollection`.

        Returns
        -------
        faces : dict
            A dictionary indexed by coordinate, where ``faces[i, j, k]`` is a
            `.Poly3DCollection` of the faces drawn for the voxel
            ``filled[i, j, k]``. If no faces were drawn for a given voxel,
            either because it was not asked to be drawn, or it is fully
            occluded, then ``(i, j, k) not in faces``.

        Examples
        --------
        .. plot:: gallery/mplot3d/voxels.py
        .. plot:: gallery/mplot3d/voxels_rgb.py
        .. plot:: gallery/mplot3d/voxels_torus.py
        .. plot:: gallery/mplot3d/voxels_numpy_logo.py
        """

        # work out which signature we should be using, and use it to parse
        # the arguments. Name must be voxels for the correct error message
        if len(args) >= 3:
            # underscores indicate position only
            def voxels(__x, __y, __z, filled, **kwargs):
                return (__x, __y, __z), filled, kwargs
        else:
            def voxels(filled, **kwargs):
                return None, filled, kwargs

        xyz, filled, kwargs = voxels(*args, **kwargs)

        # check dimensions
        if filled.ndim != 3:
            raise ValueError("Argument filled must be 3-dimensional")
        size = np.array(filled.shape, dtype=np.intp)

        # check xyz coordinates, which are one larger than the filled shape
        coord_shape = tuple(size + 1)
        if xyz is None:
            x, y, z = np.indices(coord_shape)
        else:
            x, y, z = (np.broadcast_to(c, coord_shape) for c in xyz)

        def _broadcast_color_arg(color, name):
            if np.ndim(color) in (0, 1):
                # single color, like "red" or [1, 0, 0]
                return np.broadcast_to(color, filled.shape + np.shape(color))
            elif np.ndim(color) in (3, 4):
                # 3D array of strings, or 4D array with last axis rgb
                if np.shape(color)[:3] != filled.shape:
                    raise ValueError(
                        f"When multidimensional, {name} must match the shape "
                        "of filled")
                return color
            else:
                raise ValueError(f"Invalid {name} argument")

        # broadcast and default on facecolors
        if facecolors is None:
            facecolors = self._get_patches_for_fill.get_next_color()
        facecolors = _broadcast_color_arg(facecolors, 'facecolors')

        # broadcast but no default on edgecolors
        edgecolors = _broadcast_color_arg(edgecolors, 'edgecolors')

        # scale to the full array, even if the data is only in the center
        self.auto_scale_xyz(x, y, z)

        # points lying on corners of a square
        square = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ], dtype=np.intp)

        voxel_faces = defaultdict(list)

        def permutation_matrices(n):
            """Generate cyclic permutation matrices."""
            mat = np.eye(n, dtype=np.intp)
            for i in range(n):
                yield mat
                mat = np.roll(mat, 1, axis=0)

        # iterate over each of the YZ, ZX, and XY orientations, finding faces
        # to render
        for permute in permutation_matrices(3):
            # find the set of ranges to iterate over
            pc, qc, rc = permute.T.dot(size)
            pinds = np.arange(pc)
            qinds = np.arange(qc)
            rinds = np.arange(rc)

            square_rot_pos = square.dot(permute.T)
            square_rot_neg = square_rot_pos[::-1]

            # iterate within the current plane
            for p in pinds:
                for q in qinds:
                    # iterate perpendicularly to the current plane, handling
                    # boundaries. We only draw faces between a voxel and an
                    # empty space, to avoid drawing internal faces.

                    # draw lower faces
                    p0 = permute.dot([p, q, 0])
                    i0 = tuple(p0)
                    if filled[i0]:
                        voxel_faces[i0].append(p0 + square_rot_neg)

                    # draw middle faces
                    for r1, r2 in zip(rinds[:-1], rinds[1:]):
                        p1 = permute.dot([p, q, r1])
                        p2 = permute.dot([p, q, r2])

                        i1 = tuple(p1)
                        i2 = tuple(p2)

                        if filled[i1] and not filled[i2]:
                            voxel_faces[i1].append(p2 + square_rot_pos)
                        elif not filled[i1] and filled[i2]:
                            voxel_faces[i2].append(p2 + square_rot_neg)

                    # draw upper faces
                    pk = permute.dot([p, q, rc-1])
                    pk2 = permute.dot([p, q, rc])
                    ik = tuple(pk)
                    if filled[ik]:
                        voxel_faces[ik].append(pk2 + square_rot_pos)

        # iterate over the faces, and generate a Poly3DCollection for each
        # voxel
        polygons = {}
        for coord, faces_inds in voxel_faces.items():
            # convert indices into 3D positions
            if xyz is None:
                faces = faces_inds
            else:
                faces = []
                for face_inds in faces_inds:
                    ind = face_inds[:, 0], face_inds[:, 1], face_inds[:, 2]
                    face = np.empty(face_inds.shape)
                    face[:, 0] = x[ind]
                    face[:, 1] = y[ind]
                    face[:, 2] = z[ind]
                    faces.append(face)

            # shade the faces
            facecolor = facecolors[coord]
            edgecolor = edgecolors[coord]

            poly = art3d.Poly3DCollection(
                faces, facecolors=facecolor, edgecolors=edgecolor,
                shade=shade, lightsource=lightsource, **kwargs)
            self.add_collection3d(poly)
            polygons[coord] = poly

        return polygons

    @_preprocess_data(replace_names=["x", "y", "z", "xerr", "yerr", "zerr"])
    def errorbar(self, x, y, z, zerr=None, yerr=None, xerr=None, fmt='',
                 barsabove=False, errorevery=1, ecolor=None, elinewidth=None,
                 capsize=None, capthick=None, xlolims=False, xuplims=False,
                 ylolims=False, yuplims=False, zlolims=False, zuplims=False,
                 **kwargs):
        """
        Plot lines and/or markers with errorbars around them.

        *x*/*y*/*z* define the data locations, and *xerr*/*yerr*/*zerr* define
        the errorbar sizes. By default, this draws the data markers/lines as
        well the errorbars. Use fmt='none' to draw errorbars only.

        Parameters
        ----------
        x, y, z : float or array-like
            The data positions.

        xerr, yerr, zerr : float or array-like, shape (N,) or (2, N), optional
            The errorbar sizes:

            - scalar: Symmetric +/- values for all data points.
            - shape(N,): Symmetric +/-values for each data point.
            - shape(2, N): Separate - and + values for each bar. First row
              contains the lower errors, the second row contains the upper
              errors.
            - *None*: No errorbar.

            Note that all error arrays should have *positive* values.

        fmt : str, default: ''
            The format for the data points / data lines. See `.plot` for
            details.

            Use 'none' (case-insensitive) to plot errorbars without any data
            markers.

        ecolor : color, default: None
            The color of the errorbar lines.  If None, use the color of the
            line connecting the markers.

        elinewidth : float, default: None
            The linewidth of the errorbar lines. If None, the linewidth of
            the current style is used.

        capsize : float, default: :rc:`errorbar.capsize`
            The length of the error bar caps in points.

        capthick : float, default: None
            An alias to the keyword argument *markeredgewidth* (a.k.a. *mew*).
            This setting is a more sensible name for the property that
            controls the thickness of the error bar cap in points. For
            backwards compatibility, if *mew* or *markeredgewidth* are given,
            then they will over-ride *capthick*. This may change in future
            releases.

        barsabove : bool, default: False
            If True, will plot the errorbars above the plot
            symbols. Default is below.

        xlolims, ylolims, zlolims : bool, default: False
            These arguments can be used to indicate that a value gives only
            lower limits. In that case a caret symbol is used to indicate
            this. *lims*-arguments may be scalars, or array-likes of the same
            length as the errors. To use limits with inverted axes,
            `~.Axes.set_xlim` or `~.Axes.set_ylim` must be called before
            `errorbar`. Note the tricky parameter names: setting e.g.
            *ylolims* to True means that the y-value is a *lower* limit of the
            True value, so, only an *upward*-pointing arrow will be drawn!

        xuplims, yuplims, zuplims : bool, default: False
            Same as above, but for controlling the upper limits.

        errorevery : int or (int, int), default: 1
            draws error bars on a subset of the data. *errorevery* =N draws
            error bars on the points (x[::N], y[::N], z[::N]).
            *errorevery* =(start, N) draws error bars on the points
            (x[start::N], y[start::N], z[start::N]). e.g. *errorevery* =(6, 3)
            adds error bars to the data at (x[6], x[9], x[12], x[15], ...).
            Used to avoid overlapping error bars when two series share x-axis
            values.

=======

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
