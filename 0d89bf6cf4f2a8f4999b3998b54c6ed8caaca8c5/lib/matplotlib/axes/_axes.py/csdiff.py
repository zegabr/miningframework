from collections import namedtuple
import datetime
from decimal import Decimal
import io
from itertools import product
import platform
from types import SimpleNamespace

import dateutil.tz

import numpy as np
from numpy import ma
from cycler import cycler
import pytest

import matplotlib
import matplotlib as mpl
from matplotlib.testing.decorators import (
    image_comparison, check_figures_equal, remove_ticks_and_titles)
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import matplotlib.font_manager as mfont_manager
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from numpy.testing import (
    assert_allclose, assert_array_equal, assert_array_almost_equal)
from matplotlib import rc_context
from matplotlib.cbook import MatplotlibDeprecationWarning

# Note: Some test cases are run twice: once normally and once with labeled data
#       These two must be defined in the same test function or need to have
#       different baseline images to prevent race conditions when pytest runs
#       the tests with multiple threads.


def test_get_labels():
    fig, ax = plt.subplots()
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    assert ax.get_xlabel() == 'x label'
    assert ax.get_ylabel() == 'y label'


@check_figures_equal()
def test_label_loc_vertical(fig_test, fig_ref):
    ax = fig_test.subplots()
    sc = ax.scatter([1, 2], [1, 2], c=[1, 2], label='scatter')
    ax.legend()
    ax.set_ylabel('Y Label', loc='top')
    ax.set_xlabel('X Label', loc='right')
    cbar = fig_test.colorbar(sc)
    cbar.set_label("Z Label", loc='top')

    ax = fig_ref.subplots()
    sc = ax.scatter([1, 2], [1, 2], c=[1, 2], label='scatter')
    ax.legend()
    ax.set_ylabel('Y Label', y=1, ha='right')
    ax.set_xlabel('X Label', x=1, ha='right')
    cbar = fig_ref.colorbar(sc)
    cbar.set_label("Z Label", y=1, ha='right')


@check_figures_equal()
def test_label_loc_horizontal(fig_test, fig_ref):
    ax = fig_test.subplots()
    sc = ax.scatter([1, 2], [1, 2], c=[1, 2], label='scatter')
    ax.legend()
    ax.set_ylabel('Y Label', loc='bottom')
    ax.set_xlabel('X Label', loc='left')
    cbar = fig_test.colorbar(sc, orientation='horizontal')
    cbar.set_label("Z Label", loc='left')

    ax = fig_ref.subplots()
    sc = ax.scatter([1, 2], [1, 2], c=[1, 2], label='scatter')
    ax.legend()
    ax.set_ylabel('Y Label', y=0, ha='left')
    ax.set_xlabel('X Label', x=0, ha='left')
    cbar = fig_ref.colorbar(sc, orientation='horizontal')
    cbar.set_label("Z Label", x=0, ha='left')


@check_figures_equal()
def test_label_loc_rc(fig_test, fig_ref):
    with matplotlib.rc_context({"xaxis.labellocation": "right",
                                "yaxis.labellocation": "top"}):
        ax = fig_test.subplots()
        sc = ax.scatter([1, 2], [1, 2], c=[1, 2], label='scatter')
        ax.legend()
        ax.set_ylabel('Y Label')
        ax.set_xlabel('X Label')
        cbar = fig_test.colorbar(sc, orientation='horizontal')
        cbar.set_label("Z Label")

    ax = fig_ref.subplots()
    sc = ax.scatter([1, 2], [1, 2], c=[1, 2], label='scatter')
    ax.legend()
    ax.set_ylabel('Y Label', y=1, ha='right')
    ax.set_xlabel('X Label', x=1, ha='right')
    cbar = fig_ref.colorbar(sc, orientation='horizontal')
    cbar.set_label("Z Label", x=1, ha='right')


@check_figures_equal(extensions=["png"])
def test_acorr(fig_test, fig_ref):
    np.random.seed(19680801)
    Nx = 512
    x = np.random.normal(0, 1, Nx).cumsum()
    maxlags = Nx-1

    ax_test = fig_test.subplots()
    ax_test.acorr(x, maxlags=maxlags)

    ax_ref = fig_ref.subplots()
    # Normalized autocorrelation
    norm_auto_corr = np.correlate(x, x, mode="full")/np.dot(x, x)
    lags = np.arange(-maxlags, maxlags+1)
    norm_auto_corr = norm_auto_corr[Nx-1-maxlags:Nx+maxlags]
    ax_ref.vlines(lags, [0], norm_auto_corr)
    ax_ref.axhline(y=0, xmin=0, xmax=1)


@check_figures_equal(extensions=["png"])
def test_spy(fig_test, fig_ref):
    np.random.seed(19680801)
    a = np.ones(32 * 32)
    a[:16 * 32] = 0
    np.random.shuffle(a)
    a = a.reshape((32, 32))

    axs_test = fig_test.subplots(2)
    axs_test[0].spy(a)
    axs_test[1].spy(a, marker=".", origin="lower")

    axs_ref = fig_ref.subplots(2)
    axs_ref[0].imshow(a, cmap="gray_r", interpolation="nearest")
    axs_ref[0].xaxis.tick_top()
    axs_ref[1].plot(*np.nonzero(a)[::-1], ".", markersize=10)
    axs_ref[1].set(
        aspect=1, xlim=axs_ref[0].get_xlim(), ylim=axs_ref[0].get_ylim()[::-1])
    for ax in axs_ref:
        ax.xaxis.set_ticks_position("both")


def test_spy_invalid_kwargs():
    fig, ax = plt.subplots()
    for unsupported_kw in [{'interpolation': 'nearest'},
                           {'marker': 'o', 'linestyle': 'solid'}]:
        with pytest.raises(TypeError):
            ax.spy(np.eye(3, 3), **unsupported_kw)


@check_figures_equal(extensions=["png"])
def test_matshow(fig_test, fig_ref):
    mpl.style.use("mpl20")
    a = np.random.rand(32, 32)
    fig_test.add_subplot().matshow(a)
    ax_ref = fig_ref.add_subplot()
    ax_ref.imshow(a)
    ax_ref.xaxis.tick_top()
    ax_ref.xaxis.set_ticks_position('both')


@image_comparison(['formatter_ticker_001',
                   'formatter_ticker_002',
                   'formatter_ticker_003',
                   'formatter_ticker_004',
                   'formatter_ticker_005',
                   ])
def test_formatter_ticker():
    import matplotlib.testing.jpl_units as units
    units.register()

    # This should affect the tick size.  (Tests issue #543)
    matplotlib.rcParams['lines.markeredgewidth'] = 30

    # This essentially test to see if user specified labels get overwritten
    # by the auto labeler functionality of the axes.
    xdata = [x*units.sec for x in range(10)]
    ydata1 = [(1.5*y - 0.5)*units.km for y in range(10)]
    ydata2 = [(1.75*y - 1.0)*units.km for y in range(10)]

    ax = plt.figure().subplots()
    ax.set_xlabel("x-label 001")

    ax = plt.figure().subplots()
    ax.set_xlabel("x-label 001")
    ax.plot(xdata, ydata1, color='blue', xunits="sec")

    ax = plt.figure().subplots()
    ax.set_xlabel("x-label 001")
    ax.plot(xdata, ydata1, color='blue', xunits="sec")
    ax.set_xlabel("x-label 003")

    ax = plt.figure().subplots()
    ax.plot(xdata, ydata1, color='blue', xunits="sec")
    ax.plot(xdata, ydata2, color='green', xunits="hour")
    ax.set_xlabel("x-label 004")

    # See SF bug 2846058
    # https://sourceforge.net/tracker/?func=detail&aid=2846058&group_id=80706&atid=560720
    ax = plt.figure().subplots()
    ax.plot(xdata, ydata1, color='blue', xunits="sec")
    ax.plot(xdata, ydata2, color='green', xunits="hour")
    ax.set_xlabel("x-label 005")
    ax.autoscale_view()


def test_funcformatter_auto_formatter():
    def _formfunc(x, pos):
        return ''

    ax = plt.figure().subplots()

    assert ax.xaxis.isDefault_majfmt
    assert ax.xaxis.isDefault_minfmt
    assert ax.yaxis.isDefault_majfmt
    assert ax.yaxis.isDefault_minfmt

    ax.xaxis.set_major_formatter(_formfunc)

    assert not ax.xaxis.isDefault_majfmt
    assert ax.xaxis.isDefault_minfmt
    assert ax.yaxis.isDefault_majfmt
    assert ax.yaxis.isDefault_minfmt

    targ_funcformatter = mticker.FuncFormatter(_formfunc)

    assert isinstance(ax.xaxis.get_major_formatter(),
                      mticker.FuncFormatter)

    assert ax.xaxis.get_major_formatter().func == targ_funcformatter.func


def test_strmethodformatter_auto_formatter():
    formstr = '{x}_{pos}'

    ax = plt.figure().subplots()

    assert ax.xaxis.isDefault_majfmt
    assert ax.xaxis.isDefault_minfmt
    assert ax.yaxis.isDefault_majfmt
    assert ax.yaxis.isDefault_minfmt

    ax.yaxis.set_minor_formatter(formstr)

    assert ax.xaxis.isDefault_majfmt
    assert ax.xaxis.isDefault_minfmt
    assert ax.yaxis.isDefault_majfmt
    assert not ax.yaxis.isDefault_minfmt

    targ_strformatter = mticker.StrMethodFormatter(formstr)

    assert isinstance(ax.yaxis.get_minor_formatter(),
                      mticker.StrMethodFormatter)

    assert ax.yaxis.get_minor_formatter().fmt == targ_strformatter.fmt


@image_comparison(["twin_axis_locators_formatters"])
def test_twin_axis_locators_formatters():
    vals = np.linspace(0, 1, num=5, endpoint=True)
    locs = np.sin(np.pi * vals / 2.0)

    majl = plt.FixedLocator(locs)
    minl = plt.FixedLocator([0.1, 0.2, 0.3])

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot([0.1, 100], [0, 1])
    ax1.yaxis.set_major_locator(majl)
    ax1.yaxis.set_minor_locator(minl)
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%08.2lf'))
    ax1.yaxis.set_minor_formatter(plt.FixedFormatter(['tricks', 'mind',
                                                      'jedi']))

    ax1.xaxis.set_major_locator(plt.LinearLocator())
    ax1.xaxis.set_minor_locator(plt.FixedLocator([15, 35, 55, 75]))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%05.2lf'))
    ax1.xaxis.set_minor_formatter(plt.FixedFormatter(['c', '3', 'p', 'o']))
    ax1.twiny()
    ax1.twinx()


def test_twinx_cla():
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax3 = ax2.twiny()
    plt.draw()
    assert not ax2.xaxis.get_visible()
    assert not ax2.patch.get_visible()
    ax2.cla()
    ax3.cla()

    assert not ax2.xaxis.get_visible()
    assert not ax2.patch.get_visible()
    assert ax2.yaxis.get_visible()

    assert ax3.xaxis.get_visible()
    assert not ax3.patch.get_visible()
    assert not ax3.yaxis.get_visible()

    assert ax.xaxis.get_visible()
    assert ax.patch.get_visible()
    assert ax.yaxis.get_visible()


@pytest.mark.parametrize('twin', ('x', 'y'))
@check_figures_equal(extensions=['png'], tol=0.19)
def test_twin_logscale(fig_test, fig_ref, twin):
    twin_func = f'twin{twin}'  # test twinx or twiny
    set_scale = f'set_{twin}scale'
    x = np.arange(1, 100)

    # Change scale after twinning.
    ax_test = fig_test.add_subplot(2, 1, 1)
    ax_twin = getattr(ax_test, twin_func)()
    getattr(ax_test, set_scale)('log')
    ax_twin.plot(x, x)

    # Twin after changing scale.
    ax_test = fig_test.add_subplot(2, 1, 2)
    getattr(ax_test, set_scale)('log')
    ax_twin = getattr(ax_test, twin_func)()
    ax_twin.plot(x, x)

    for i in [1, 2]:
        ax_ref = fig_ref.add_subplot(2, 1, i)
        getattr(ax_ref, set_scale)('log')
        ax_ref.plot(x, x)

        # This is a hack because twinned Axes double-draw the frame.
        # Remove this when that is fixed.
        Path = matplotlib.path.Path
        fig_ref.add_artist(
            matplotlib.patches.PathPatch(
                Path([[0, 0], [0, 1],
                      [0, 1], [1, 1],
                      [1, 1], [1, 0],
                      [1, 0], [0, 0]],
                     [Path.MOVETO, Path.LINETO] * 4),
                transform=ax_ref.transAxes,
                facecolor='none',
                edgecolor=mpl.rcParams['axes.edgecolor'],
                linewidth=mpl.rcParams['axes.linewidth'],
                capstyle='projecting'))

    remove_ticks_and_titles(fig_test)
    remove_ticks_and_titles(fig_ref)


@image_comparison(['twin_autoscale.png'])
def test_twinx_axis_scales():
    x = np.array([0, 0.5, 1])
    y = 0.5 * x
    x2 = np.array([0, 1, 2])
    y2 = 2 * x2

    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1), autoscalex_on=False, autoscaley_on=False)
    ax.plot(x, y, color='blue', lw=10)

    ax2 = plt.twinx(ax)
    ax2.plot(x2, y2, 'r--', lw=5)

    ax.margins(0, 0)
    ax2.margins(0, 0)


def test_twin_inherit_autoscale_setting():
    fig, ax = plt.subplots()
    ax_x_on = ax.twinx()
    ax.set_autoscalex_on(False)
    ax_x_off = ax.twinx()

    assert ax_x_on.get_autoscalex_on()
    assert not ax_x_off.get_autoscalex_on()

    ax_y_on = ax.twiny()
    ax.set_autoscaley_on(False)
    ax_y_off = ax.twiny()

    assert ax_y_on.get_autoscaley_on()
    assert not ax_y_off.get_autoscaley_on()


def test_inverted_cla():
    # GitHub PR #5450. Setting autoscale should reset
    # axes to be non-inverted.
    # plotting an image, then 1d graph, axis is now down
    fig = plt.figure(0)
    ax = fig.gca()
    # 1. test that a new axis is not inverted per default
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()
    img = np.random.random((100, 100))
    ax.imshow(img)
    # 2. test that a image axis is inverted
    assert not ax.xaxis_inverted()
    assert ax.yaxis_inverted()
    # 3. test that clearing and plotting a line, axes are
    # not inverted
    ax.cla()
    x = np.linspace(0, 2*np.pi, 100)
    ax.plot(x, np.cos(x))
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()

    # 4. autoscaling should not bring back axes to normal
    ax.cla()
    ax.imshow(img)
    plt.autoscale()
    assert not ax.xaxis_inverted()
    assert ax.yaxis_inverted()

    # 5. two shared axes. Inverting the master axis should invert the shared
    # axes; clearing the master axis should bring axes in shared
    # axes back to normal.
    ax0 = plt.subplot(211)
    ax1 = plt.subplot(212, sharey=ax0)
    ax0.yaxis.set_inverted(True)
    assert ax1.yaxis_inverted()
    ax1.plot(x, np.cos(x))
    ax0.cla()
    assert not ax1.yaxis_inverted()
    ax1.cla()
    # 6. clearing the nonmaster should not touch limits
    ax0.imshow(img)
    ax1.plot(x, np.cos(x))
    ax1.cla()
    assert ax.yaxis_inverted()

    # clean up
    plt.close(fig)


@check_figures_equal(extensions=["png"])
def test_minorticks_on_rcParams_both(fig_test, fig_ref):
    with matplotlib.rc_context({"xtick.minor.visible": True,
                                "ytick.minor.visible": True}):
        ax_test = fig_test.subplots()
        ax_test.plot([0, 1], [0, 1])
    ax_ref = fig_ref.subplots()
    ax_ref.plot([0, 1], [0, 1])
    ax_ref.minorticks_on()


@image_comparison(["autoscale_tiny_range"], remove_text=True)
def test_autoscale_tiny_range():
    # github pull #904
    fig, axs = plt.subplots(2, 2)
    for i, ax in enumerate(axs.flat):
        y1 = 10**(-11 - i)
        ax.plot([0, 1], [1, 1 + y1])


@mpl.style.context('default')
def test_autoscale_tight():
    fig, ax = plt.subplots(1, 1)
    ax.plot([1, 2, 3, 4])
    ax.autoscale(enable=True, axis='x', tight=False)
    ax.autoscale(enable=True, axis='y', tight=True)
    assert_allclose(ax.get_xlim(), (-0.15, 3.15))
    assert_allclose(ax.get_ylim(), (1.0, 4.0))


@mpl.style.context('default')
def test_autoscale_log_shared():
    # related to github #7587
    # array starts at zero to trigger _minpos handling
    x = np.arange(100, dtype=float)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.loglog(x, x)
    ax2.semilogx(x, x)
    ax1.autoscale(tight=True)
    ax2.autoscale(tight=True)
    plt.draw()
    lims = (x[1], x[-1])
    assert_allclose(ax1.get_xlim(), lims)
    assert_allclose(ax1.get_ylim(), lims)
    assert_allclose(ax2.get_xlim(), lims)
    assert_allclose(ax2.get_ylim(), (x[0], x[-1]))


@mpl.style.context('default')
def test_use_sticky_edges():
    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [2, 3]], origin='lower')
    assert_allclose(ax.get_xlim(), (-0.5, 1.5))
    assert_allclose(ax.get_ylim(), (-0.5, 1.5))
    ax.use_sticky_edges = False
    ax.autoscale()
    xlim = (-0.5 - 2 * ax._xmargin, 1.5 + 2 * ax._xmargin)
    ylim = (-0.5 - 2 * ax._ymargin, 1.5 + 2 * ax._ymargin)
    assert_allclose(ax.get_xlim(), xlim)
    assert_allclose(ax.get_ylim(), ylim)
    # Make sure it is reversible:
    ax.use_sticky_edges = True
    ax.autoscale()
    assert_allclose(ax.get_xlim(), (-0.5, 1.5))
    assert_allclose(ax.get_ylim(), (-0.5, 1.5))


@check_figures_equal(extensions=["png"])
def test_sticky_shared_axes(fig_test, fig_ref):
    # Check that sticky edges work whether they are set in an axes that is a
    # "master" in a share, or an axes that is a "follower".
    Z = np.arange(15).reshape(3, 5)

    ax0 = fig_test.add_subplot(211)
    ax1 = fig_test.add_subplot(212, sharex=ax0)
    ax1.pcolormesh(Z)

    ax0 = fig_ref.add_subplot(212)
    ax1 = fig_ref.add_subplot(211, sharex=ax0)
    ax0.pcolormesh(Z)


@image_comparison(['offset_points'], remove_text=True)
def test_basic_annotate():
    # Setup some data
    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2.0*np.pi * t)

    # Offset Points

    fig = plt.figure()
    ax = fig.add_subplot(autoscale_on=False, xlim=(-1, 5), ylim=(-3, 5))
    line, = ax.plot(t, s, lw=3, color='purple')

    ax.annotate('local max', xy=(3, 1), xycoords='data',
                xytext=(3, 3), textcoords='offset points')


@image_comparison(['arrow_simple.png'], remove_text=True)
def test_arrow_simple():
    # Simple image test for ax.arrow
    # kwargs that take discrete values
    length_includes_head = (True, False)
    shape = ('full', 'left', 'right')
    head_starts_at_zero = (True, False)
    # Create outer product of values
    kwargs = product(length_includes_head, shape, head_starts_at_zero)

    fig, axs = plt.subplots(3, 4)
    for i, (ax, kwarg) in enumerate(zip(axs.flat, kwargs)):
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        # Unpack kwargs
        (length_includes_head, shape, head_starts_at_zero) = kwarg
        theta = 2 * np.pi * i / 12
        # Draw arrow
        ax.arrow(0, 0, np.sin(theta), np.cos(theta),
                 width=theta/100,
                 length_includes_head=length_includes_head,
                 shape=shape,
                 head_starts_at_zero=head_starts_at_zero,
                 head_width=theta / 10,
                 head_length=theta / 10)


def test_arrow_empty():
    _, ax = plt.subplots()
    # Create an empty FancyArrow
    ax.arrow(0, 0, 0, 0, head_length=0)


def test_arrow_in_view():
    _, ax = plt.subplots()
    ax.arrow(1, 1, 1, 1)
    assert ax.get_xlim() == (0.8, 2.2)
    assert ax.get_ylim() == (0.8, 2.2)


def test_annotate_default_arrow():
    # Check that we can make an annotation arrow with only default properties.
    fig, ax = plt.subplots()
    ann = ax.annotate("foo", (0, 1), xytext=(2, 3))
    assert ann.arrow_patch is None
    ann = ax.annotate("foo", (0, 1), xytext=(2, 3), arrowprops={})
    assert ann.arrow_patch is not None


@image_comparison(['fill_units.png'], savefig_kwarg={'dpi': 60})
def test_fill_units():
    import matplotlib.testing.jpl_units as units
    units.register()

    # generate some data
    t = units.Epoch("ET", dt=datetime.datetime(2009, 4, 27))
    value = 10.0 * units.deg
    day = units.Duration("ET", 24.0 * 60.0 * 60.0)
    dt = np.arange('2009-04-27', '2009-04-29', dtype='datetime64[D]')
    dtn = mdates.date2num(dt)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.plot([t], [value], yunits='deg', color='red')
    ind = [0, 0, 1, 1]
    ax1.fill(dtn[ind], [0.0, 0.0, 90.0, 0.0], 'b')

    ax2.plot([t], [value], yunits='deg', color='red')
    ax2.fill([t, t, t + day, t + day],
             [0.0, 0.0, 90.0, 0.0], 'b')

    ax3.plot([t], [value], yunits='deg', color='red')
    ax3.fill(dtn[ind],
             [0 * units.deg, 0 * units.deg, 90 * units.deg, 0 * units.deg],
             'b')

    ax4.plot([t], [value], yunits='deg', color='red')
    ax4.fill([t, t, t + day, t + day],
             [0 * units.deg, 0 * units.deg, 90 * units.deg, 0 * units.deg],
             facecolor="blue")
    fig.autofmt_xdate()


def test_plot_format_kwarg_redundant():
    with pytest.warns(UserWarning, match="marker .* redundantly defined"):
        plt.plot([0], [0], 'o', marker='x')
    with pytest.warns(UserWarning, match="linestyle .* redundantly defined"):
        plt.plot([0], [0], '-', linestyle='--')
    with pytest.warns(UserWarning, match="color .* redundantly defined"):
        plt.plot([0], [0], 'r', color='blue')
    # smoke-test: should not warn
    plt.errorbar([0], [0], fmt='none', color='blue')


@image_comparison(['single_point', 'single_point'])
def test_single_point():
    # Issue #1796: don't let lines.marker affect the grid
    matplotlib.rcParams['lines.marker'] = 'o'
    matplotlib.rcParams['axes.grid'] = True

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot([0], [0], 'o')
    ax2.plot([1], [1], 'o')

    # Reuse testcase from above for a labeled data test
    data = {'a': [0], 'b': [1]}

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot('a', 'a', 'o', data=data)
    ax2.plot('b', 'b', 'o', data=data)


@image_comparison(['single_date.png'], style='mpl20')
def test_single_date():

    # use former defaults to match existing baseline image
    plt.rcParams['axes.formatter.limits'] = -7, 7
    dt = mdates.date2num(np.datetime64('0000-12-31'))

    time1 = [721964.0]
    data1 = [-65.54]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot_date(time1 + dt, data1, 'o', color='r')
    ax[1].plot(time1, data1, 'o', color='r')


@check_figures_equal(extensions=["png"])
def test_shaped_data(fig_test, fig_ref):
    row = np.arange(10).reshape((1, -1))
    col = np.arange(0, 100, 10).reshape((-1, 1))

    axs = fig_test.subplots(2)
    axs[0].plot(row)  # Actually plots nothing (columns are single points).
    axs[1].plot(col)  # Same as plotting 1d.

    axs = fig_ref.subplots(2)
    # xlim from the implicit "x=0", ylim from the row datalim.
    axs[0].set(xlim=(-.06, .06), ylim=(0, 9))
    axs[1].plot(col.ravel())


def test_structured_data():
    # support for structured data
    pts = np.array([(1, 1), (2, 2)], dtype=[("ones", float), ("twos", float)])

    # this should not read second name as a format and raise ValueError
    axs = plt.figure().subplots(2)
    axs[0].plot("ones", "twos", data=pts)
    axs[1].plot("ones", "twos", "r", data=pts)


@image_comparison(['aitoff_proj'], extensions=["png"],
                  remove_text=True, style='mpl20')
def test_aitoff_proj():
    """
    Test aitoff projection ref.:
    https://github.com/matplotlib/matplotlib/pull/14451
    """
    x = np.linspace(-np.pi, np.pi, 20)
    y = np.linspace(-np.pi / 2, np.pi / 2, 20)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 4.2),
                           subplot_kw=dict(projection="aitoff"))
    ax.grid()
    ax.plot(X.flat, Y.flat, 'o', markersize=4)


@image_comparison(['axvspan_epoch'])
def test_axvspan_epoch():
    import matplotlib.testing.jpl_units as units
    units.register()

    # generate some data
    t0 = units.Epoch("ET", dt=datetime.datetime(2009, 1, 20))
    tf = units.Epoch("ET", dt=datetime.datetime(2009, 1, 21))
    dt = units.Duration("ET", units.day.convert("sec"))

    ax = plt.gca()
    ax.axvspan(t0, tf, facecolor="blue", alpha=0.25)
    ax.set_xlim(t0 - 5.0*dt, tf + 5.0*dt)


@image_comparison(['axhspan_epoch'], tol=0.02)
def test_axhspan_epoch():
    import matplotlib.testing.jpl_units as units
    units.register()

    # generate some data
    t0 = units.Epoch("ET", dt=datetime.datetime(2009, 1, 20))
    tf = units.Epoch("ET", dt=datetime.datetime(2009, 1, 21))
    dt = units.Duration("ET", units.day.convert("sec"))

    ax = plt.gca()
    ax.axhspan(t0, tf, facecolor="blue", alpha=0.25)
    ax.set_ylim(t0 - 5.0*dt, tf + 5.0*dt)


@image_comparison(['hexbin_extent.png', 'hexbin_extent.png'], remove_text=True)
def test_hexbin_extent():
    # this test exposes sf bug 2856228
    fig, ax = plt.subplots()
    data = (np.arange(2000) / 2000).reshape((2, 1000))
    x, y = data

    ax.hexbin(x, y, extent=[.1, .3, .6, .7])

    # Reuse testcase from above for a labeled data test
    data = {"x": x, "y": y}

    fig, ax = plt.subplots()
    ax.hexbin("x", "y", extent=[.1, .3, .6, .7], data=data)


@image_comparison(['hexbin_empty.png'], remove_text=True)
def test_hexbin_empty():
    # From #3886: creating hexbin from empty dataset raises ValueError
    ax = plt.gca()
    ax.hexbin([], [])


def test_hexbin_pickable():
    # From #1973: Test that picking a hexbin collection works
    fig, ax = plt.subplots()
    data = (np.arange(200) / 200).reshape((2, 100))
    x, y = data
    hb = ax.hexbin(x, y, extent=[.1, .3, .6, .7], picker=-1)
    mouse_event = SimpleNamespace(x=400, y=300)
    assert hb.contains(mouse_event)[0]


@image_comparison(['hexbin_log.png'], style='mpl20')
def test_hexbin_log():
    # Issue #1636 (and also test log scaled colorbar)

    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    np.random.seed(19680801)
    n = 100000
    x = np.random.standard_normal(n)
    y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
    y = np.power(2, y * 0.5)

    fig, ax = plt.subplots()
    h = ax.hexbin(x, y, yscale='log', bins='log',
                  marginals=True, reduce_C_function=np.sum)
    plt.colorbar(h)


def test_hexbin_log_clim():
    x, y = np.arange(200).reshape((2, 100))
    fig, ax = plt.subplots()
    h = ax.hexbin(x, y, bins='log', vmin=2, vmax=100)
    assert h.get_clim() == (2, 100)


def test_inverted_limits():
    # Test gh:1553
    # Calling invert_xaxis prior to plotting should not disable autoscaling
    # while still maintaining the inverted direction
    fig, ax = plt.subplots()
    ax.invert_xaxis()
    ax.plot([-5, -3, 2, 4], [1, 2, -3, 5])

    assert ax.get_xlim() == (4, -5)
    assert ax.get_ylim() == (-3, 5)
    plt.close()

    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.plot([-5, -3, 2, 4], [1, 2, -3, 5])

    assert ax.get_xlim() == (-5, 4)
    assert ax.get_ylim() == (5, -3)

    # Test inverting nonlinear axes.
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylim(10, 1)
    assert ax.get_ylim() == (10, 1)


@image_comparison(['nonfinite_limits'])
def test_nonfinite_limits():
    x = np.arange(0., np.e, 0.01)
    # silence divide by zero warning from log(0)
    with np.errstate(divide='ignore'):
        y = np.log(x)
    x[len(x)//2] = np.nan
    fig, ax = plt.subplots()
    ax.plot(x, y)


@mpl.style.context('default')
@pytest.mark.parametrize('plot_fun',
                         ['scatter', 'plot', 'fill_between'])
@check_figures_equal(extensions=["png"])
def test_limits_empty_data(plot_fun, fig_test, fig_ref):
    # Check that plotting empty data doesn't change autoscaling of dates
    x = np.arange("2010-01-01", "2011-01-01", dtype="datetime64[D]")

    ax_test = fig_test.subplots()
    ax_ref = fig_ref.subplots()

    getattr(ax_test, plot_fun)([], [])

    for ax in [ax_test, ax_ref]:
        getattr(ax, plot_fun)(x, range(len(x)), color='C0')


@image_comparison(['imshow', 'imshow'], remove_text=True, style='mpl20')
def test_imshow():
    # use former defaults to match existing baseline image
    matplotlib.rcParams['image.interpolation'] = 'nearest'
    # Create a NxN image
    N = 100
    (x, y) = np.indices((N, N))
    x -= N//2
    y -= N//2
    r = np.sqrt(x**2+y**2-x*y)

    # Create a contour plot at N/4 and extract both the clip path and transform
    fig, ax = plt.subplots()
    ax.imshow(r)

    # Reuse testcase from above for a labeled data test
    data = {"r": r}
    fig, ax = plt.subplots()
    ax.imshow("r", data=data)


@image_comparison(['imshow_clip'], style='mpl20')
def test_imshow_clip():
    # As originally reported by Gellule Xg <gellule.xg@free.fr>
    # use former defaults to match existing baseline image
    matplotlib.rcParams['image.interpolation'] = 'nearest'

    # Create a NxN image
    N = 100
    (x, y) = np.indices((N, N))
    x -= N//2
    y -= N//2
    r = np.sqrt(x**2+y**2-x*y)

    # Create a contour plot at N/4 and extract both the clip path and transform
    fig, ax = plt.subplots()

    c = ax.contour(r, [N/4])
    x = c.collections[0]
    clip_path = x.get_paths()[0]
    clip_transform = x.get_transform()

    clip_path = mtransforms.TransformedPath(clip_path, clip_transform)

    # Plot the image clipped by the contour
    ax.imshow(r, clip_path=clip_path)


def test_imshow_norm_vminvmax():
    """Parameters vmin, vmax should error if norm is given."""
    a = [[1, 2], [3, 4]]
    ax = plt.axes()
    with pytest.raises(ValueError,
                       match="Passing parameters norm and vmin/vmax "
                             "simultaneously is not supported."):
        ax.imshow(a, norm=mcolors.Normalize(-10, 10), vmin=0, vmax=5)


@image_comparison(['polycollection_joinstyle'], remove_text=True)
def test_polycollection_joinstyle():
    # Bug #2890979 reported by Matthew West
    fig, ax = plt.subplots()
    verts = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
    c = mpl.collections.PolyCollection([verts], linewidths=40)
    ax.add_collection(c)
    ax.set_xbound(0, 3)
    ax.set_ybound(0, 3)


@pytest.mark.parametrize(
    'x, y1, y2', [
        (np.zeros((2, 2)), 3, 3),
        (np.arange(0.0, 2, 0.02), np.zeros((2, 2)), 3),
        (np.arange(0.0, 2, 0.02), 3, np.zeros((2, 2)))
    ], ids=[
        '2d_x_input',
        '2d_y1_input',
        '2d_y2_input'
    ]
)
def test_fill_between_input(x, y1, y2):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.fill_between(x, y1, y2)


@pytest.mark.parametrize(
    'y, x1, x2', [
        (np.zeros((2, 2)), 3, 3),
        (np.arange(0.0, 2, 0.02), np.zeros((2, 2)), 3),
        (np.arange(0.0, 2, 0.02), 3, np.zeros((2, 2)))
    ], ids=[
        '2d_y_input',
        '2d_x1_input',
        '2d_x2_input'
    ]
)
def test_fill_betweenx_input(y, x1, x2):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.fill_betweenx(y, x1, x2)


@image_comparison(['fill_between_interpolate'], remove_text=True)
def test_fill_between_interpolate():
    x = np.arange(0.0, 2, 0.02)
    y1 = np.sin(2*np.pi*x)
    y2 = 1.2*np.sin(4*np.pi*x)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(x, y1, x, y2, color='black')
    ax1.fill_between(x, y1, y2, where=y2 >= y1, facecolor='white', hatch='/',
                     interpolate=True)
    ax1.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red',
                     interpolate=True)

    # Test support for masked arrays.
    y2 = np.ma.masked_greater(y2, 1.0)
    # Test that plotting works for masked arrays with the first element masked
    y2[0] = np.ma.masked
    ax2.plot(x, y1, x, y2, color='black')
    ax2.fill_between(x, y1, y2, where=y2 >= y1, facecolor='green',
                     interpolate=True)
    ax2.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red',
                     interpolate=True)


@image_comparison(['fill_between_interpolate_decreasing'],
                  style='mpl20', remove_text=True)
def test_fill_between_interpolate_decreasing():
    p = np.array([724.3, 700, 655])
    t = np.array([9.4, 7, 2.2])
    prof = np.array([7.9, 6.6, 3.8])

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.plot(t, p, 'tab:red')
    ax.plot(prof, p, 'k')

    ax.fill_betweenx(p, t, prof, where=prof < t,
                     facecolor='blue', interpolate=True, alpha=0.4)
    ax.fill_betweenx(p, t, prof, where=prof > t,
                     facecolor='red', interpolate=True, alpha=0.4)

    ax.set_xlim(0, 30)
    ax.set_ylim(800, 600)


@image_comparison(['fill_between_interpolate_nan'], remove_text=True)
def test_fill_between_interpolate_nan():
    # Tests fix for issue #18986.
    x = np.arange(10)
    y1 = np.asarray([8, 18, np.nan, 18, 8, 18, 24, 18, 8, 18])
    y2 = np.asarray([18, 11, 8, 11, 18, 26, 32, 30, np.nan, np.nan])

    # NumPy <1.19 issues warning 'invalid value encountered in greater_equal'
    # for comparisons that include nan.
    with np.errstate(invalid='ignore'):
        greater2 = y2 >= y1
        greater1 = y1 >= y2

    fig, ax = plt.subplots()

    ax.plot(x, y1, c='k')
    ax.plot(x, y2, c='b')
    ax.fill_between(x, y1, y2, where=greater2, facecolor="green",
                    interpolate=True, alpha=0.5)
    ax.fill_between(x, y1, y2, where=greater1, facecolor="red",
                    interpolate=True, alpha=0.5)


# test_symlog and test_symlog2 used to have baseline images in all three
# formats, but the png and svg baselines got invalidated by the removal of
# minor tick overstriking.
@image_comparison(['symlog.pdf'])
def test_symlog():
    x = np.array([0, 1, 2, 4, 6, 9, 12, 24])
    y = np.array([1000000, 500000, 100000, 100, 5, 0, 0, 0])

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_yscale('symlog')
    ax.set_xscale('linear')
    ax.set_ylim(-1, 10000000)


@image_comparison(['symlog2.pdf'], remove_text=True)
def test_symlog2():
    # Numbers from -50 to 50, with 0.1 as step
    x = np.arange(-50, 50, 0.001)

    fig, axs = plt.subplots(5, 1)
    for ax, linthresh in zip(axs, [20., 2., 1., 0.1, 0.01]):
        ax.plot(x, x)
        ax.set_xscale('symlog', linthresh=linthresh)
        ax.grid(True)
    axs[-1].set_ylim(-0.1, 0.1)


def test_pcolorargs_5205():
    # Smoketest to catch issue found in gh:5205
    x = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    y = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0,
         0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    X, Y = np.meshgrid(x, y)
    Z = np.hypot(X, Y)

    plt.pcolor(Z)
    plt.pcolor(list(Z))
    plt.pcolor(x, y, Z[:-1, :-1])
    plt.pcolor(X, Y, list(Z[:-1, :-1]))


@image_comparison(['pcolormesh'], remove_text=True)
def test_pcolormesh():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    n = 12
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n*2)
    X, Y = np.meshgrid(x, y)
    Qx = np.cos(Y) - np.cos(X)
    Qz = np.sin(Y) + np.sin(X)
    Qx = (Qx + 1.1)
    Z = np.hypot(X, Y) / 5
    Z = (Z - Z.min()) / Z.ptp()

    # The color array can include masked values:
    Zm = ma.masked_where(np.abs(Qz) < 0.5 * np.max(Qz), Z)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.pcolormesh(Qx, Qz, Z[:-1, :-1], lw=0.5, edgecolors='k')
    ax2.pcolormesh(Qx, Qz, Z[:-1, :-1], lw=2, edgecolors=['b', 'w'])
    ax3.pcolormesh(Qx, Qz, Z, shading="gouraud")


@image_comparison(['pcolormesh_alpha'], extensions=["png", "pdf"],
                  remove_text=True)
def test_pcolormesh_alpha():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    n = 12
    X, Y = np.meshgrid(
        np.linspace(-1.5, 1.5, n),
        np.linspace(-1.5, 1.5, n*2)
    )
    Qx = X
    Qy = Y + np.sin(X)
    Z = np.hypot(X, Y) / 5
    Z = (Z - Z.min()) / Z.ptp()
    vir = plt.get_cmap("viridis", 16)
    # make another colormap with varying alpha
    colors = vir(np.arange(16))
    colors[:, 3] = 0.5 + 0.5*np.sin(np.arange(16))
    cmap = mcolors.ListedColormap(colors)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    for ax in ax1, ax2, ax3, ax4:
        ax.add_patch(mpatches.Rectangle(
            (0, -1.5), 1.5, 3, facecolor=[.7, .1, .1, .5], zorder=0
        ))
    # ax1, ax2: constant alpha
    ax1.pcolormesh(Qx, Qy, Z[:-1, :-1], cmap=vir, alpha=0.4,
                   shading='flat', zorder=1)
    ax2.pcolormesh(Qx, Qy, Z, cmap=vir, alpha=0.4, shading='gouraud', zorder=1)
    # ax3, ax4: alpha from colormap
    ax3.pcolormesh(Qx, Qy, Z[:-1, :-1], cmap=cmap, shading='flat', zorder=1)
    ax4.pcolormesh(Qx, Qy, Z, cmap=cmap, shading='gouraud', zorder=1)


@image_comparison(['pcolormesh_datetime_axis.png'],
                  remove_text=False, style='mpl20')
def test_pcolormesh_datetime_axis():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, top=0.98, bottom=.15)
    base = datetime.datetime(2013, 1, 1)
    x = np.array([base + datetime.timedelta(days=d) for d in range(21)])
    y = np.arange(21)
    z1, z2 = np.meshgrid(np.arange(20), np.arange(20))
    z = z1 * z2
    plt.subplot(221)
    plt.pcolormesh(x[:-1], y[:-1], z[:-1, :-1])
    plt.subplot(222)
    plt.pcolormesh(x, y, z)
    x = np.repeat(x[np.newaxis], 21, axis=0)
    y = np.repeat(y[:, np.newaxis], 21, axis=1)
    plt.subplot(223)
    plt.pcolormesh(x[:-1, :-1], y[:-1, :-1], z[:-1, :-1])
    plt.subplot(224)
    plt.pcolormesh(x, y, z)
    for ax in fig.get_axes():
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(30)


@image_comparison(['pcolor_datetime_axis.png'],
                  remove_text=False, style='mpl20')
def test_pcolor_datetime_axis():
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, top=0.98, bottom=.15)
    base = datetime.datetime(2013, 1, 1)
    x = np.array([base + datetime.timedelta(days=d) for d in range(21)])
    y = np.arange(21)
    z1, z2 = np.meshgrid(np.arange(20), np.arange(20))
    z = z1 * z2
    plt.subplot(221)
    plt.pcolor(x[:-1], y[:-1], z[:-1, :-1])
    plt.subplot(222)
    plt.pcolor(x, y, z)
    x = np.repeat(x[np.newaxis], 21, axis=0)
    y = np.repeat(y[:, np.newaxis], 21, axis=1)
    plt.subplot(223)
    plt.pcolor(x[:-1, :-1], y[:-1, :-1], z[:-1, :-1])
    plt.subplot(224)
    plt.pcolor(x, y, z)
    for ax in fig.get_axes():
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(30)


def test_pcolorargs():
    n = 12
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n*2)
    X, Y = np.meshgrid(x, y)
    Z = np.hypot(X, Y) / 5

    _, ax = plt.subplots()
    with pytest.raises(TypeError):
        ax.pcolormesh(y, x, Z)
    with pytest.raises(TypeError):
        ax.pcolormesh(X, Y, Z.T)
    with pytest.raises(TypeError):
        ax.pcolormesh(x, y, Z[:-1, :-1], shading="gouraud")
    with pytest.raises(TypeError):
        ax.pcolormesh(X, Y, Z[:-1, :-1], shading="gouraud")
    x[0] = np.NaN
    with pytest.raises(ValueError):
        ax.pcolormesh(x, y, Z[:-1, :-1])
    with np.errstate(invalid='ignore'):
        x = np.ma.array(x, mask=(x < 0))
    with pytest.raises(ValueError):
        ax.pcolormesh(x, y, Z[:-1, :-1])
    # Expect a warning with non-increasing coordinates
    x = [359, 0, 1]
    y = [-10, 10]
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    with pytest.warns(UserWarning,
                      match='are not monotonically increasing or decreasing'):
        ax.pcolormesh(X, Y, Z, shading='auto')


@check_figures_equal(extensions=["png"])
def test_pcolornearest(fig_test, fig_ref):
    ax = fig_test.subplots()
    x = np.arange(0, 10)
    y = np.arange(0, 3)
    np.random.seed(19680801)
    Z = np.random.randn(2, 9)
    ax.pcolormesh(x, y, Z, shading='flat')

    ax = fig_ref.subplots()
    # specify the centers
    x2 = x[:-1] + np.diff(x) / 2
    y2 = y[:-1] + np.diff(y) / 2
    ax.pcolormesh(x2, y2, Z, shading='nearest')


@check_figures_equal(extensions=["png"])
def test_pcolornearestunits(fig_test, fig_ref):
    ax = fig_test.subplots()
    x = [datetime.datetime.fromtimestamp(x * 3600) for x in range(10)]
    y = np.arange(0, 3)
    np.random.seed(19680801)
    Z = np.random.randn(2, 9)
    ax.pcolormesh(x, y, Z, shading='flat')

    ax = fig_ref.subplots()
    # specify the centers
    x2 = [datetime.datetime.fromtimestamp((x + 0.5) * 3600) for x in range(9)]
    y2 = y[:-1] + np.diff(y) / 2
    ax.pcolormesh(x2, y2, Z, shading='nearest')


def test_pcolorflaterror():
    fig, ax = plt.subplots()
    x = np.arange(0, 9)
    y = np.arange(0, 3)
    np.random.seed(19680801)
    Z = np.random.randn(3, 9)
    with pytest.raises(TypeError, match='Dimensions of C'):
        ax.pcolormesh(x, y, Z, shading='flat')


@check_figures_equal(extensions=["png"])
def test_pcolorauto(fig_test, fig_ref):
    ax = fig_test.subplots()
    x = np.arange(0, 10)
    y = np.arange(0, 4)
    np.random.seed(19680801)
    Z = np.random.randn(3, 9)
    # this is the same as flat; note that auto is default
    ax.pcolormesh(x, y, Z)

    ax = fig_ref.subplots()
    # specify the centers
    x2 = x[:-1] + np.diff(x) / 2
    y2 = y[:-1] + np.diff(y) / 2
    # this is same as nearest:
    ax.pcolormesh(x2, y2, Z)


@image_comparison(['canonical'])
def test_canonical():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])


@image_comparison(['arc_angles.png'], remove_text=True, style='default')
def test_arc_angles():
    # Ellipse parameters
    w = 2
    h = 1
    centre = (0.2, 0.5)
    scale = 2

    fig, axs = plt.subplots(3, 3)
    for i, ax in enumerate(axs.flat):
        theta2 = i * 360 / 9
        theta1 = theta2 - 45

        ax.add_patch(mpatches.Ellipse(centre, w, h, alpha=0.3))
        ax.add_patch(mpatches.Arc(centre, w, h, theta1=theta1, theta2=theta2))
        # Straight lines intersecting start and end of arc
        ax.plot([scale * np.cos(np.deg2rad(theta1)) + centre[0],
                 centre[0],
                 scale * np.cos(np.deg2rad(theta2)) + centre[0]],
                [scale * np.sin(np.deg2rad(theta1)) + centre[1],
                 centre[1],
                 scale * np.sin(np.deg2rad(theta2)) + centre[1]])

        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)

        # This looks the same, but it triggers a different code path when it
        # gets large enough.
        w *= 10
        h *= 10
        centre = (centre[0] * 10, centre[1] * 10)
        scale *= 10


@image_comparison(['arc_ellipse'], remove_text=True)
def test_arc_ellipse():
    xcenter, ycenter = 0.38, 0.52
    width, height = 1e-1, 3e-1
    angle = -30

    theta = np.deg2rad(np.arange(360))
    x = width / 2. * np.cos(theta)
    y = height / 2. * np.sin(theta)

    rtheta = np.deg2rad(angle)
    R = np.array([
        [np.cos(rtheta), -np.sin(rtheta)],
        [np.sin(rtheta), np.cos(rtheta)]])

    x, y = np.dot(R, np.array([x, y]))
    x += xcenter
    y += ycenter

    fig = plt.figure()
    ax = fig.add_subplot(211, aspect='auto')
    ax.fill(x, y, alpha=0.2, facecolor='yellow', edgecolor='yellow',
            linewidth=1, zorder=1)

    e1 = mpatches.Arc((xcenter, ycenter), width, height,
                      angle=angle, linewidth=2, fill=False, zorder=2)

    ax.add_patch(e1)

    ax = fig.add_subplot(212, aspect='equal')
    ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)
    e2 = mpatches.Arc((xcenter, ycenter), width, height,
                      angle=angle, linewidth=2, fill=False, zorder=2)

    ax.add_patch(e2)


def test_marker_as_markerstyle():
    fix, ax = plt.subplots()
    m = mmarkers.MarkerStyle('o')
    ax.plot([1, 2, 3], [3, 2, 1], marker=m)
    ax.scatter([1, 2, 3], [4, 3, 2], marker=m)
    ax.errorbar([1, 2, 3], [5, 4, 3], marker=m)


@image_comparison(['markevery'], remove_text=True)
def test_markevery():
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.sqrt(x/10 + 0.5)

    # check marker only plot
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', label='default')
    ax.plot(x, y, 'd', markevery=None, label='mark all')
    ax.plot(x, y, 's', markevery=10, label='mark every 10')
    ax.plot(x, y, '+', markevery=(5, 20), label='mark every 5 starting at 10')
    ax.legend()


@image_comparison(['markevery_line'], remove_text=True, tol=0.005)
def test_markevery_line():
    # TODO: a slight change in rendering between Inkscape versions may explain
    # why one had to introduce a small non-zero tolerance for the SVG test
    # to pass. One may try to remove this hack once Travis' Inkscape version
    # is modern enough. FWIW, no failure with 0.92.3 on my computer (#11358).
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.sqrt(x/10 + 0.5)

    # check line/marker combos
    fig, ax = plt.subplots()
    ax.plot(x, y, '-o', label='default')
    ax.plot(x, y, '-d', markevery=None, label='mark all')
    ax.plot(x, y, '-s', markevery=10, label='mark every 10')
    ax.plot(x, y, '-+', markevery=(5, 20), label='mark every 5 starting at 10')
    ax.legend()


@image_comparison(['markevery_linear_scales'], remove_text=True, tol=0.001)
def test_markevery_linear_scales():
    cases = [None,
             8,
             (30, 8),
             [16, 24, 30], [0, -1],
             slice(100, 200, 3),
             0.1, 0.3, 1.5,
             (0.0, 0.1), (0.45, 0.1)]

    cols = 3
    gs = matplotlib.gridspec.GridSpec(len(cases) // cols + 1, cols)

    delta = 0.11
    x = np.linspace(0, 10 - 2 * delta, 200) + delta
    y = np.sin(x) + 1.0 + delta

    for i, case in enumerate(cases):
        row = (i // cols)
        col = i % cols
        plt.subplot(gs[row, col])
        plt.title('markevery=%s' % str(case))
        plt.plot(x, y, 'o', ls='-', ms=4,  markevery=case)


@image_comparison(['markevery_linear_scales_zoomed'], remove_text=True)
def test_markevery_linear_scales_zoomed():
    cases = [None,
             8,
             (30, 8),
             [16, 24, 30], [0, -1],
             slice(100, 200, 3),
             0.1, 0.3, 1.5,
             (0.0, 0.1), (0.45, 0.1)]

    cols = 3
    gs = matplotlib.gridspec.GridSpec(len(cases) // cols + 1, cols)

    delta = 0.11
    x = np.linspace(0, 10 - 2 * delta, 200) + delta
    y = np.sin(x) + 1.0 + delta

    for i, case in enumerate(cases):
        row = (i // cols)
        col = i % cols
        plt.subplot(gs[row, col])
        plt.title('markevery=%s' % str(case))
        plt.plot(x, y, 'o', ls='-', ms=4,  markevery=case)
        plt.xlim((6, 6.7))
        plt.ylim((1.1, 1.7))


@image_comparison(['markevery_log_scales'], remove_text=True)
def test_markevery_log_scales():
    cases = [None,
             8,
             (30, 8),
             [16, 24, 30], [0, -1],
             slice(100, 200, 3),
             0.1, 0.3, 1.5,
             (0.0, 0.1), (0.45, 0.1)]

    cols = 3
    gs = matplotlib.gridspec.GridSpec(len(cases) // cols + 1, cols)

    delta = 0.11
    x = np.linspace(0, 10 - 2 * delta, 200) + delta
    y = np.sin(x) + 1.0 + delta

    for i, case in enumerate(cases):
        row = (i // cols)
        col = i % cols
        plt.subplot(gs[row, col])
        plt.title('markevery=%s' % str(case))
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(x, y, 'o', ls='-', ms=4,  markevery=case)


@image_comparison(['markevery_polar'], style='default', remove_text=True)
def test_markevery_polar():
    cases = [None,
             8,
             (30, 8),
             [16, 24, 30], [0, -1],
             slice(100, 200, 3),
             0.1, 0.3, 1.5,
             (0.0, 0.1), (0.45, 0.1)]

    cols = 3
    gs = matplotlib.gridspec.GridSpec(len(cases) // cols + 1, cols)

    r = np.linspace(0, 3.0, 200)
    theta = 2 * np.pi * r

    for i, case in enumerate(cases):
        row = (i // cols)
        col = i % cols
        plt.subplot(gs[row, col], polar=True)
        plt.title('markevery=%s' % str(case))
        plt.plot(theta, r, 'o', ls='-', ms=4,  markevery=case)


@image_comparison(['marker_edges'], remove_text=True)
def test_marker_edges():
    x = np.linspace(0, 1, 10)
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x), 'y.', ms=30.0, mew=0, mec='r')
    ax.plot(x+0.1, np.sin(x), 'y.', ms=30.0, mew=1, mec='r')
    ax.plot(x+0.2, np.sin(x), 'y.', ms=30.0, mew=2, mec='b')


@image_comparison(['bar_tick_label_single.png', 'bar_tick_label_single.png'])
def test_bar_tick_label_single():
    # From 2516: plot bar with array of string labels for x axis
    ax = plt.gca()
    ax.bar(0, 1, align='edge', tick_label='0')

    # Reuse testcase from above for a labeled data test
    data = {"a": 0, "b": 1}
    fig, ax = plt.subplots()
    ax = plt.gca()
    ax.bar("a", "b", align='edge', tick_label='0', data=data)


def test_nan_bar_values():
    fig, ax = plt.subplots()
    ax.bar([0, 1], [np.nan, 4])


def test_bar_ticklabel_fail():
    fig, ax = plt.subplots()
    ax.bar([], [])


@image_comparison(['bar_tick_label_multiple.png'])
def test_bar_tick_label_multiple():
    # From 2516: plot bar with array of string labels for x axis
    ax = plt.gca()
    ax.bar([1, 2.5], [1, 2], width=[0.2, 0.5], tick_label=['a', 'b'],
           align='center')


@image_comparison(['bar_tick_label_multiple_old_label_alignment.png'])
def test_bar_tick_label_multiple_old_alignment():
    # Test that the alignment for class is backward compatible
    matplotlib.rcParams["ytick.alignment"] = "center"
    ax = plt.gca()
    ax.bar([1, 2.5], [1, 2], width=[0.2, 0.5], tick_label=['a', 'b'],
           align='center')


@check_figures_equal(extensions=["png"])
def test_bar_decimal_center(fig_test, fig_ref):
    ax = fig_test.subplots()
    x0 = [1.5, 8.4, 5.3, 4.2]
    y0 = [1.1, 2.2, 3.3, 4.4]
    x = [Decimal(x) for x in x0]
    y = [Decimal(y) for y in y0]
    # Test image - vertical, align-center bar chart with Decimal() input
    ax.bar(x, y, align='center')
    # Reference image
    ax = fig_ref.subplots()
    ax.bar(x0, y0, align='center')


@check_figures_equal(extensions=["png"])
def test_barh_decimal_center(fig_test, fig_ref):
    ax = fig_test.subplots()
    x0 = [1.5, 8.4, 5.3, 4.2]
    y0 = [1.1, 2.2, 3.3, 4.4]
    x = [Decimal(x) for x in x0]
    y = [Decimal(y) for y in y0]
    # Test image - horizontal, align-center bar chart with Decimal() input
    ax.barh(x, y, height=[0.5, 0.5, 1, 1], align='center')
    # Reference image
    ax = fig_ref.subplots()
    ax.barh(x0, y0, height=[0.5, 0.5, 1, 1], align='center')


@check_figures_equal(extensions=["png"])
def test_bar_decimal_width(fig_test, fig_ref):
    x = [1.5, 8.4, 5.3, 4.2]
    y = [1.1, 2.2, 3.3, 4.4]
    w0 = [0.7, 1.45, 1, 2]
    w = [Decimal(i) for i in w0]
    # Test image - vertical bar chart with Decimal() width
    ax = fig_test.subplots()
    ax.bar(x, y, width=w, align='center')
    # Reference image
    ax = fig_ref.subplots()
    ax.bar(x, y, width=w0, align='center')


@check_figures_equal(extensions=["png"])
def test_barh_decimal_height(fig_test, fig_ref):
    x = [1.5, 8.4, 5.3, 4.2]
    y = [1.1, 2.2, 3.3, 4.4]
    h0 = [0.7, 1.45, 1, 2]
    h = [Decimal(i) for i in h0]
    # Test image - horizontal bar chart with Decimal() height
    ax = fig_test.subplots()
    ax.barh(x, y, height=h, align='center')
    # Reference image
    ax = fig_ref.subplots()
    ax.barh(x, y, height=h0, align='center')


def test_bar_color_none_alpha():
    ax = plt.gca()
    rects = ax.bar([1, 2], [2, 4], alpha=0.3, color='none', edgecolor='r')
    for rect in rects:
        assert rect.get_facecolor() == (0, 0, 0, 0)
        assert rect.get_edgecolor() == (1, 0, 0, 0.3)


def test_bar_edgecolor_none_alpha():
    ax = plt.gca()
    rects = ax.bar([1, 2], [2, 4], alpha=0.3, color='r', edgecolor='none')
    for rect in rects:
        assert rect.get_facecolor() == (1, 0, 0, 0.3)
        assert rect.get_edgecolor() == (0, 0, 0, 0)


@image_comparison(['barh_tick_label.png'])
def test_barh_tick_label():
    # From 2516: plot barh with array of string labels for y axis
    ax = plt.gca()
    ax.barh([1, 2.5], [1, 2], height=[0.2, 0.5], tick_label=['a', 'b'],
            align='center')


def test_bar_timedelta():
    """Smoketest that bar can handle width and height in delta units."""
    fig, ax = plt.subplots()
    ax.bar(datetime.datetime(2018, 1, 1), 1.,
           width=datetime.timedelta(hours=3))
    ax.bar(datetime.datetime(2018, 1, 1), 1.,
           xerr=datetime.timedelta(hours=2),
           width=datetime.timedelta(hours=3))
    fig, ax = plt.subplots()
    ax.barh(datetime.datetime(2018, 1, 1), 1,
            height=datetime.timedelta(hours=3))
    ax.barh(datetime.datetime(2018, 1, 1), 1,
            height=datetime.timedelta(hours=3),
            yerr=datetime.timedelta(hours=2))
    fig, ax = plt.subplots()
    ax.barh([datetime.datetime(2018, 1, 1), datetime.datetime(2018, 1, 1)],
            np.array([1, 1.5]),
            height=datetime.timedelta(hours=3))
    ax.barh([datetime.datetime(2018, 1, 1), datetime.datetime(2018, 1, 1)],
            np.array([1, 1.5]),
            height=[datetime.timedelta(hours=t) for t in [1, 2]])
    ax.broken_barh([(datetime.datetime(2018, 1, 1),
                     datetime.timedelta(hours=1))],
                   (10, 20))


def test_boxplot_dates_pandas(pd):
    # smoke test for boxplot and dates in pandas
    data = np.random.rand(5, 2)
    years = pd.date_range('1/1/2000',
                          periods=2, freq=pd.DateOffset(years=1)).year
    plt.figure()
    plt.boxplot(data, positions=years)


def test_pcolor_regression(pd):
    from pandas.plotting import (
        register_matplotlib_converters,
        deregister_matplotlib_converters,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    times = [datetime.datetime(2021, 1, 1)]
    while len(times) < 7:
        times.append(times[-1] + datetime.timedelta(seconds=120))

    y_vals = np.arange(5)

    time_axis, y_axis = np.meshgrid(times, y_vals)
    shape = (len(y_vals) - 1, len(times) - 1)
    z_data = np.arange(shape[0] * shape[1])

    z_data.shape = shape
    try:
        register_matplotlib_converters()

        im = ax.pcolormesh(time_axis, y_axis, z_data)
        # make sure this does not raise!
        fig.canvas.draw()
    finally:
        deregister_matplotlib_converters()


def test_bar_pandas(pd):
    # Smoke test for pandas
    df = pd.DataFrame(
        {'year': [2018, 2018, 2018],
         'month': [1, 1, 1],
         'day': [1, 2, 3],
         'value': [1, 2, 3]})
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    monthly = df[['date', 'value']].groupby(['date']).sum()
    dates = monthly.index
    forecast = monthly['value']
    baseline = monthly['value']

    fig, ax = plt.subplots()
    ax.bar(dates, forecast, width=10, align='center')
    ax.plot(dates, baseline, color='orange', lw=4)


def test_bar_pandas_indexed(pd):
    # Smoke test for indexed pandas
    df = pd.DataFrame({"x": [1., 2., 3.], "width": [.2, .4, .6]},
                      index=[1, 2, 3])
    fig, ax = plt.subplots()
    ax.bar(df.x, 1., width=df.width)


@mpl.style.context('default')
@check_figures_equal()
def test_bar_hatches(fig_test, fig_ref):
    ax_test = fig_test.subplots()
    ax_ref = fig_ref.subplots()

    x = [1, 2]
    y = [2, 3]
    hatches = ['x', 'o']
    for i in range(2):
        ax_ref.bar(x[i], y[i], color='C0', hatch=hatches[i])

    ax_test.bar(x, y, hatch=hatches)


def test_pandas_minimal_plot(pd):
    # smoke test that series and index objcets do not warn
    x = pd.Series([1, 2], dtype="float64")
    plt.plot(x, x)
    plt.plot(x.index, x)
    plt.plot(x)
    plt.plot(x.index)


@image_comparison(['hist_log'], remove_text=True)
def test_hist_log():
    data0 = np.linspace(0, 1, 200)**3
    data = np.concatenate([1 - data0, 1 + data0])
    fig, ax = plt.subplots()
    ax.hist(data, fill=False, log=True)


@check_figures_equal(extensions=["png"])
def test_hist_log_2(fig_test, fig_ref):
    axs_test = fig_test.subplots(2, 3)
    axs_ref = fig_ref.subplots(2, 3)
    for i, histtype in enumerate(["bar", "step", "stepfilled"]):
        # Set log scale, then call hist().
        axs_test[0, i].set_yscale("log")
        axs_test[0, i].hist(1, 1, histtype=histtype)
        # Call hist(), then set log scale.
        axs_test[1, i].hist(1, 1, histtype=histtype)
        axs_test[1, i].set_yscale("log")
        # Use hist(..., log=True).
        for ax in axs_ref[:, i]:
            ax.hist(1, 1, log=True, histtype=histtype)


def test_hist_log_barstacked():
    fig, axs = plt.subplots(2)
    axs[0].hist([[0], [0, 1]], 2, histtype="barstacked")
    axs[0].set_yscale("log")
    axs[1].hist([0, 0, 1], 2, histtype="barstacked")
    axs[1].set_yscale("log")
    fig.canvas.draw()
    assert axs[0].get_ylim() == axs[1].get_ylim()


@image_comparison(['hist_bar_empty.png'], remove_text=True)
def test_hist_bar_empty():
    # From #3886: creating hist from empty dataset raises ValueError
    ax = plt.gca()
    ax.hist([], histtype='bar')


@image_comparison(['hist_step_empty.png'], remove_text=True)
def test_hist_step_empty():
    # From #3886: creating hist from empty dataset raises ValueError
    ax = plt.gca()
    ax.hist([], histtype='step')


@image_comparison(['hist_step_filled.png'], remove_text=True)
def test_hist_step_filled():
    np.random.seed(0)
    x = np.random.randn(1000, 3)
    n_bins = 10

    kwargs = [{'fill': True}, {'fill': False}, {'fill': None}, {}]*2
    types = ['step']*4+['stepfilled']*4
    fig, axs = plt.subplots(nrows=2, ncols=4)

    for kg, _type, ax in zip(kwargs, types, axs.flat):
        ax.hist(x, n_bins, histtype=_type, stacked=True, **kg)
        ax.set_title('%s/%s' % (kg, _type))
        ax.set_ylim(bottom=-50)

    patches = axs[0, 0].patches
    assert all(p.get_facecolor() == p.get_edgecolor() for p in patches)


@image_comparison(['hist_density.png'])
def test_hist_density():
    np.random.seed(19680801)
    data = np.random.standard_normal(2000)
    fig, ax = plt.subplots()
    ax.hist(data, density=True)


def test_hist_unequal_bins_density():
    # Test correct behavior of normalized histogram with unequal bins
    # https://github.com/matplotlib/matplotlib/issues/9557
    rng = np.random.RandomState(57483)
    t = rng.randn(100)
    bins = [-3, -1, -0.5, 0, 1, 5]
    mpl_heights, _, _ = plt.hist(t, bins=bins, density=True)
    np_heights, _ = np.histogram(t, bins=bins, density=True)
    assert_allclose(mpl_heights, np_heights)


def test_hist_datetime_datasets():
    data = [[datetime.datetime(2017, 1, 1), datetime.datetime(2017, 1, 1)],
            [datetime.datetime(2017, 1, 1), datetime.datetime(2017, 1, 2)]]
    fig, ax = plt.subplots()
    ax.hist(data, stacked=True)
    ax.hist(data, stacked=False)


@pytest.mark.parametrize("bins_preprocess",
                         [mpl.dates.date2num,
                          lambda bins: bins,
                          lambda bins: np.asarray(bins).astype('datetime64')],
                         ids=['date2num', 'datetime.datetime',
                              'np.datetime64'])
def test_hist_datetime_datasets_bins(bins_preprocess):
    data = [[datetime.datetime(2019, 1, 5), datetime.datetime(2019, 1, 11),
             datetime.datetime(2019, 2, 1), datetime.datetime(2019, 3, 1)],
            [datetime.datetime(2019, 1, 11), datetime.datetime(2019, 2, 5),
             datetime.datetime(2019, 2, 18), datetime.datetime(2019, 3, 1)]]

    date_edges = [datetime.datetime(2019, 1, 1), datetime.datetime(2019, 2, 1),
                  datetime.datetime(2019, 3, 1)]

    fig, ax = plt.subplots()
    _, bins, _ = ax.hist(data, bins=bins_preprocess(date_edges), stacked=True)
    np.testing.assert_allclose(bins, mpl.dates.date2num(date_edges))

    _, bins, _ = ax.hist(data, bins=bins_preprocess(date_edges), stacked=False)
    np.testing.assert_allclose(bins, mpl.dates.date2num(date_edges))


@pytest.mark.parametrize('data, expected_number_of_hists',
                         [([], 1),
                          ([[]], 1),
                          ([[], []], 2)])
def test_hist_with_empty_input(data, expected_number_of_hists):
    hists, _, _ = plt.hist(data)
    hists = np.asarray(hists)

    if hists.ndim == 1:
        assert 1 == expected_number_of_hists
    else:
        assert hists.shape[0] == expected_number_of_hists


@pytest.mark.parametrize("histtype, zorder",
                         [("bar", mpl.patches.Patch.zorder),
                          ("step", mpl.lines.Line2D.zorder),
                          ("stepfilled", mpl.patches.Patch.zorder)])
def test_hist_zorder(histtype, zorder):
    ax = plt.figure().add_subplot()
    ax.hist([1, 2], histtype=histtype)
    assert ax.patches
    for patch in ax.patches:
        assert patch.get_zorder() == zorder


@check_figures_equal(extensions=['png'])
def test_stairs(fig_test, fig_ref):
    import matplotlib.lines as mlines
    y = np.array([6, 14, 32, 37, 48, 32, 21,  4])  # hist
    x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])  # bins

    test_axes = fig_test.subplots(3, 2).flatten()
    test_axes[0].stairs(y, x, baseline=None)
    test_axes[1].stairs(y, x, baseline=None, orientation='horizontal')
    test_axes[2].stairs(y, x)
    test_axes[3].stairs(y, x, orientation='horizontal')
    test_axes[4].stairs(y, x)
    test_axes[4].semilogy()
    test_axes[5].stairs(y, x, orientation='horizontal')
    test_axes[5].semilogx()

    # defaults of `PathPatch` to be used for all following Line2D
    style = {'solid_joinstyle': 'miter', 'solid_capstyle': 'butt'}

    ref_axes = fig_ref.subplots(3, 2).flatten()
    ref_axes[0].plot(x, np.append(y, y[-1]), drawstyle='steps-post', **style)
    ref_axes[1].plot(np.append(y[0], y), x, drawstyle='steps-post', **style)

    ref_axes[2].plot(x, np.append(y, y[-1]), drawstyle='steps-post', **style)
    ref_axes[2].add_line(mlines.Line2D([x[0], x[0]], [0, y[0]], **style))
    ref_axes[2].add_line(mlines.Line2D([x[-1], x[-1]], [0, y[-1]], **style))
    ref_axes[2].set_ylim(0, None)

    ref_axes[3].plot(np.append(y[0], y), x, drawstyle='steps-post', **style)
    ref_axes[3].add_line(mlines.Line2D([0, y[0]], [x[0], x[0]], **style))
    ref_axes[3].add_line(mlines.Line2D([0, y[-1]], [x[-1], x[-1]], **style))
    ref_axes[3].set_xlim(0, None)

    ref_axes[4].plot(x, np.append(y, y[-1]), drawstyle='steps-post', **style)
    ref_axes[4].add_line(mlines.Line2D([x[0], x[0]], [0, y[0]], **style))
    ref_axes[4].add_line(mlines.Line2D([x[-1], x[-1]], [0, y[-1]], **style))
    ref_axes[4].semilogy()

    ref_axes[5].plot(np.append(y[0], y), x, drawstyle='steps-post', **style)
    ref_axes[5].add_line(mlines.Line2D([0, y[0]], [x[0], x[0]], **style))
    ref_axes[5].add_line(mlines.Line2D([0, y[-1]], [x[-1], x[-1]], **style))
    ref_axes[5].semilogx()


@check_figures_equal(extensions=['png'])
def test_stairs_fill(fig_test, fig_ref):
    h, bins = [1, 2, 3, 4, 2], [0, 1, 2, 3, 4, 5]
    bs = -2
    # Test
    test_axes = fig_test.subplots(2, 2).flatten()
    test_axes[0].stairs(h, bins, fill=True)
    test_axes[1].stairs(h, bins, orientation='horizontal', fill=True)
    test_axes[2].stairs(h, bins, baseline=bs, fill=True)
    test_axes[3].stairs(h, bins, baseline=bs, orientation='horizontal',
                        fill=True)

    # # Ref
    ref_axes = fig_ref.subplots(2, 2).flatten()
    ref_axes[0].fill_between(bins, np.append(h, h[-1]), step='post', lw=0)
    ref_axes[0].set_ylim(0, None)
    ref_axes[1].fill_betweenx(bins, np.append(h, h[-1]), step='post', lw=0)
    ref_axes[1].set_xlim(0, None)
    ref_axes[2].fill_between(bins, np.append(h, h[-1]),
                             np.ones(len(h)+1)*bs, step='post', lw=0)
    ref_axes[2].set_ylim(bs, None)
    ref_axes[3].fill_betweenx(bins, np.append(h, h[-1]),
                              np.ones(len(h)+1)*bs, step='post', lw=0)
    ref_axes[3].set_xlim(bs, None)


@check_figures_equal(extensions=['png'])
def test_stairs_update(fig_test, fig_ref):
    # fixed ylim because stairs() does autoscale, but updating data does not
    ylim = -3, 4
    # Test
    test_ax = fig_test.add_subplot()
    h = test_ax.stairs([1, 2, 3])
    test_ax.set_ylim(ylim)
    h.set_data([3, 2, 1])
    h.set_data(edges=np.arange(4)+2)
    h.set_data([1, 2, 1], np.arange(4)/2)
    h.set_data([1, 2, 3])
    h.set_data(None, np.arange(4))
    assert np.allclose(h.get_data()[0], np.arange(1, 4))
    assert np.allclose(h.get_data()[1], np.arange(4))
    h.set_data(baseline=-2)
    assert h.get_data().baseline == -2

    # Ref
    ref_ax = fig_ref.add_subplot()
    h = ref_ax.stairs([1, 2, 3], baseline=-2)
    ref_ax.set_ylim(ylim)


@check_figures_equal(extensions=['png'])
def test_stairs_baseline_0(fig_test, fig_ref):
    # Test
    test_ax = fig_test.add_subplot()
    test_ax.stairs([5, 6, 7], baseline=None)

    # Ref
    ref_ax = fig_ref.add_subplot()
    style = {'solid_joinstyle': 'miter', 'solid_capstyle': 'butt'}
    ref_ax.plot(range(4), [5, 6, 7, 7], drawstyle='steps-post', **style)
    ref_ax.set_ylim(0, None)


def test_stairs_empty():
    ax = plt.figure().add_subplot()
    ax.stairs([], [42])
    assert ax.get_xlim() == (39, 45)
    assert ax.get_ylim() == (-0.06, 0.06)


def test_stairs_invalid_nan():
    with pytest.raises(ValueError, match='Nan values in "edges"'):
        plt.stairs([1, 2], [0, np.nan, 1])


def test_stairs_invalid_mismatch():
    with pytest.raises(ValueError, match='Size mismatch'):
        plt.stairs([1, 2], [0, 1])


def test_stairs_invalid_update():
    h = plt.stairs([1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match='Nan values in "edges"'):
        h.set_data(edges=[1, np.nan, 2])


def test_stairs_invalid_update2():
    h = plt.stairs([1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match='Size mismatch'):
        h.set_data(edges=np.arange(5))


@image_comparison(['test_stairs_options.png'], remove_text=True)
def test_stairs_options():
    x, y = np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4]).astype(float)
    yn = y.copy()
    yn[1] = np.nan

    fig, ax = plt.subplots()
    ax.stairs(y*3, x, color='green', fill=True, label="A")
    ax.stairs(y, x*3-3, color='red', fill=True,
              orientation='horizontal', label="B")
    ax.stairs(yn, x, color='orange', ls='--', lw=2, label="C")
    ax.stairs(yn/3, x*3-2, ls='--', lw=2, baseline=0.5,
              orientation='horizontal', label="D")
    ax.stairs(y[::-1]*3+13, x-1, color='red', ls='--', lw=2, baseline=None,
              label="E")
    ax.stairs(y[::-1]*3+14, x, baseline=26,
              color='purple', ls='--', lw=2, label="F")
    ax.stairs(yn[::-1]*3+15, x+1, baseline=np.linspace(27, 25, len(y)),
              color='blue', ls='--', lw=2, label="G", fill=True)
    ax.stairs(y[:-1][::-1]*2+11, x[:-1]+0.5, color='black', ls='--', lw=2,
              baseline=12, hatch='//', label="H")
    ax.legend(loc=0)


@image_comparison(['test_stairs_datetime.png'])
def test_stairs_datetime():
    f, ax = plt.subplots(constrained_layout=True)
    ax.stairs(np.arange(36),
              np.arange(np.datetime64('2001-12-27'),
                        np.datetime64('2002-02-02')))
    plt.xticks(rotation=30)


def contour_dat():
    x = np.linspace(-3, 5, 150)
    y = np.linspace(-3, 5, 120)
    z = np.cos(x) + np.sin(y[:, np.newaxis])
    return x, y, z


@image_comparison(['contour_hatching'], remove_text=True, style='mpl20')
def test_contour_hatching():
    x, y, z = contour_dat()
    fig, ax = plt.subplots()
    ax.contourf(x, y, z, 7, hatches=['/', '\\', '//', '-'],
                cmap=plt.get_cmap('gray'),
                extend='both', alpha=0.5)


@image_comparison(['contour_colorbar'], style='mpl20')
def test_contour_colorbar():
    x, y, z = contour_dat()

    fig, ax = plt.subplots()
    cs = ax.contourf(x, y, z, levels=np.arange(-1.8, 1.801, 0.2),
                     cmap=plt.get_cmap('RdBu'),
                     vmin=-0.6,
                     vmax=0.6,
                     extend='both')
    cs1 = ax.contour(x, y, z, levels=np.arange(-2.2, -0.599, 0.2),
                     colors=['y'],
                     linestyles='solid',
                     linewidths=2)
    cs2 = ax.contour(x, y, z, levels=np.arange(0.6, 2.2, 0.2),
                     colors=['c'],
                     linewidths=2)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.add_lines(cs1)
    cbar.add_lines(cs2, erase=False)


@image_comparison(['hist2d', 'hist2d'], remove_text=True, style='mpl20')
def test_hist2d():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    np.random.seed(0)
    # make it not symmetric in case we switch x and y axis
    x = np.random.randn(100)*2+5
    y = np.random.randn(100)-2
    fig, ax = plt.subplots()
    ax.hist2d(x, y, bins=10, rasterized=True)

    # Reuse testcase from above for a labeled data test
    data = {"x": x, "y": y}
    fig, ax = plt.subplots()
    ax.hist2d("x", "y", bins=10, data=data, rasterized=True)


@image_comparison(['hist2d_transpose'], remove_text=True, style='mpl20')
def test_hist2d_transpose():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    np.random.seed(0)
    # make sure the output from np.histogram is transposed before
    # passing to pcolorfast
    x = np.array([5]*100)
    y = np.random.randn(100)-2
    fig, ax = plt.subplots()
    ax.hist2d(x, y, bins=10, rasterized=True)


def test_hist2d_density():
    x, y = np.random.random((2, 100))
    ax = plt.figure().subplots()
    for obj in [ax, plt]:
        obj.hist2d(x, y, density=True)


class TestScatter:
    @image_comparison(['scatter'], style='mpl20', remove_text=True)
    def test_scatter_plot(self):
        data = {"x": np.array([3, 4, 2, 6]), "y": np.array([2, 5, 2, 3]),
                "c": ['r', 'y', 'b', 'lime'], "s": [24, 15, 19, 29],
                "c2": ['0.5', '0.6', '0.7', '0.8']}

        fig, ax = plt.subplots()
        ax.scatter(data["x"] - 1., data["y"] - 1., c=data["c"], s=data["s"])
        ax.scatter(data["x"] + 1., data["y"] + 1., c=data["c2"], s=data["s"])
        ax.scatter("x", "y", c="c", s="s", data=data)

    @image_comparison(['scatter_marker.png'], remove_text=True)
    def test_scatter_marker(self):
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)
        ax0.scatter([3, 4, 2, 6], [2, 5, 2, 3],
                    c=[(1, 0, 0), 'y', 'b', 'lime'],
                    s=[60, 50, 40, 30],
                    edgecolors=['k', 'r', 'g', 'b'],
                    marker='s')
        ax1.scatter([3, 4, 2, 6], [2, 5, 2, 3],
                    c=[(1, 0, 0), 'y', 'b', 'lime'],
                    s=[60, 50, 40, 30],
                    edgecolors=['k', 'r', 'g', 'b'],
                    marker=mmarkers.MarkerStyle('o', fillstyle='top'))
        # unit area ellipse
        rx, ry = 3, 1
        area = rx * ry * np.pi
        theta = np.linspace(0, 2 * np.pi, 21)
        verts = np.column_stack([np.cos(theta) * rx / area,
                                 np.sin(theta) * ry / area])
        ax2.scatter([3, 4, 2, 6], [2, 5, 2, 3],
                    c=[(1, 0, 0), 'y', 'b', 'lime'],
                    s=[60, 50, 40, 30],
                    edgecolors=['k', 'r', 'g', 'b'],
                    marker=verts)

    @image_comparison(['scatter_2D'], remove_text=True, extensions=['png'])
    def test_scatter_2D(self):
        x = np.arange(3)
        y = np.arange(2)
        x, y = np.meshgrid(x, y)
        z = x + y
        fig, ax = plt.subplots()
        ax.scatter(x, y, c=z, s=200, edgecolors='face')

    @check_figures_equal(extensions=["png"])
    def test_scatter_decimal(self, fig_test, fig_ref):
        x0 = np.array([1.5, 8.4, 5.3, 4.2])
        y0 = np.array([1.1, 2.2, 3.3, 4.4])
        x = np.array([Decimal(i) for i in x0])
        y = np.array([Decimal(i) for i in y0])
        c = ['r', 'y', 'b', 'lime']
        s = [24, 15, 19, 29]
        # Test image - scatter plot with Decimal() input
        ax = fig_test.subplots()
        ax.scatter(x, y, c=c, s=s)
        # Reference image
        ax = fig_ref.subplots()
        ax.scatter(x0, y0, c=c, s=s)

    def test_scatter_color(self):
        # Try to catch cases where 'c' kwarg should have been used.
        with pytest.raises(ValueError):
            plt.scatter([1, 2], [1, 2], color=[0.1, 0.2])
        with pytest.raises(ValueError):
            plt.scatter([1, 2, 3], [1, 2, 3], color=[1, 2, 3])

    def test_scatter_unfilled(self):
        coll = plt.scatter([0, 1, 2], [1, 3, 2], c=['0.1', '0.3', '0.5'],
                           marker=mmarkers.MarkerStyle('o', fillstyle='none'),
                           linewidths=[1.1, 1.2, 1.3])
        assert coll.get_facecolors().shape == (0, 4)  # no facecolors
        assert_array_equal(coll.get_edgecolors(), [[0.1, 0.1, 0.1, 1],
                                                   [0.3, 0.3, 0.3, 1],
                                                   [0.5, 0.5, 0.5, 1]])
        assert_array_equal(coll.get_linewidths(), [1.1, 1.2, 1.3])

    @mpl.style.context('default')
    def test_scatter_unfillable(self):
        coll = plt.scatter([0, 1, 2], [1, 3, 2], c=['0.1', '0.3', '0.5'],
                           marker='x',
                           linewidths=[1.1, 1.2, 1.3])
        assert_array_equal(coll.get_facecolors(), coll.get_edgecolors())
        assert_array_equal(coll.get_edgecolors(), [[0.1, 0.1, 0.1, 1],
                                                   [0.3, 0.3, 0.3, 1],
                                                   [0.5, 0.5, 0.5, 1]])
        assert_array_equal(coll.get_linewidths(), [1.1, 1.2, 1.3])

    def test_scatter_size_arg_size(self):
        x = np.arange(4)
        with pytest.raises(ValueError, match='same size as x and y'):
            plt.scatter(x, x, x[1:])
        with pytest.raises(ValueError, match='same size as x and y'):
            plt.scatter(x[1:], x[1:], x)
        with pytest.raises(ValueError, match='float array-like'):
            plt.scatter(x, x, 'foo')

    def test_scatter_edgecolor_RGB(self):
        # Github issue 19066
        coll = plt.scatter([1, 2, 3], [1, np.nan, np.nan],
                            edgecolor=(1, 0, 0))
        assert mcolors.same_color(coll.get_edgecolor(), (1, 0, 0))
        coll = plt.scatter([1, 2, 3, 4], [1, np.nan, np.nan, 1],
                            edgecolor=(1, 0, 0, 1))
        assert mcolors.same_color(coll.get_edgecolor(), (1, 0, 0, 1))

    @check_figures_equal(extensions=["png"])
    def test_scatter_invalid_color(self, fig_test, fig_ref):
        ax = fig_test.subplots()
        cmap = plt.get_cmap("viridis", 16)
        cmap.set_bad("k", 1)
        # Set a nonuniform size to prevent the last call to `scatter` (plotting
        # the invalid points separately in fig_ref) from using the marker
        # stamping fast path, which would result in slightly offset markers.
        ax.scatter(range(4), range(4),
                   c=[1, np.nan, 2, np.nan], s=[1, 2, 3, 4],
                   cmap=cmap, plotnonfinite=True)
        ax = fig_ref.subplots()
        cmap = plt.get_cmap("viridis", 16)
        ax.scatter([0, 2], [0, 2], c=[1, 2], s=[1, 3], cmap=cmap)
        ax.scatter([1, 3], [1, 3], s=[2, 4], color="k")

    @check_figures_equal(extensions=["png"])
    def test_scatter_no_invalid_color(self, fig_test, fig_ref):
        # With plotninfinite=False we plot only 2 points.
        ax = fig_test.subplots()
        cmap = plt.get_cmap("viridis", 16)
        cmap.set_bad("k", 1)
        ax.scatter(range(4), range(4),
                   c=[1, np.nan, 2, np.nan], s=[1, 2, 3, 4],
                   cmap=cmap, plotnonfinite=False)
        ax = fig_ref.subplots()
        ax.scatter([0, 2], [0, 2], c=[1, 2], s=[1, 3], cmap=cmap)

    def test_scatter_norm_vminvmax(self):
        """Parameters vmin, vmax should error if norm is given."""
        x = [1, 2, 3]
        ax = plt.axes()
        with pytest.raises(ValueError,
                           match="Passing parameters norm and vmin/vmax "
                                 "simultaneously is not supported."):
            ax.scatter(x, x, c=x, norm=mcolors.Normalize(-10, 10),
                       vmin=0, vmax=5)

    @check_figures_equal(extensions=["png"])
    def test_scatter_single_point(self, fig_test, fig_ref):
        ax = fig_test.subplots()
        ax.scatter(1, 1, c=1)
        ax = fig_ref.subplots()
        ax.scatter([1], [1], c=[1])

    @check_figures_equal(extensions=["png"])
    def test_scatter_different_shapes(self, fig_test, fig_ref):
        x = np.arange(10)
        ax = fig_test.subplots()
        ax.scatter(x, x.reshape(2, 5), c=x.reshape(5, 2))
        ax = fig_ref.subplots()
        ax.scatter(x.reshape(5, 2), x, c=x.reshape(2, 5))

    # Parameters for *test_scatter_c*. NB: assuming that the
    # scatter plot will have 4 elements. The tuple scheme is:
    # (*c* parameter case, exception regexp key or None if no exception)
    params_test_scatter_c = [
        # single string:
        ('0.5', None),
        # Single letter-sequences
        (["rgby"], "conversion"),
        # Special cases
        ("red", None),
        ("none", None),
        (None, None),
        (["r", "g", "b", "none"], None),
        # Non-valid color spec (FWIW, 'jaune' means yellow in French)
        ("jaune", "conversion"),
        (["jaune"], "conversion"),  # wrong type before wrong size
        (["jaune"]*4, "conversion"),
        # Value-mapping like
        ([0.5]*3, None),  # should emit a warning for user's eyes though
        ([0.5]*4, None),  # NB: no warning as matching size allows mapping
        ([0.5]*5, "shape"),
        # list of strings:
        (['0.5', '0.4', '0.6', '0.7'], None),
        (['0.5', 'red', '0.6', 'C5'], None),
        (['0.5', 0.5, '0.6', 'C5'], "conversion"),
        # RGB values
        ([[1, 0, 0]], None),
        ([[1, 0, 0]]*3, "shape"),
        ([[1, 0, 0]]*4, None),
        ([[1, 0, 0]]*5, "shape"),
        # RGBA values
        ([[1, 0, 0, 0.5]], None),
        ([[1, 0, 0, 0.5]]*3, "shape"),
        ([[1, 0, 0, 0.5]]*4, None),
        ([[1, 0, 0, 0.5]]*5, "shape"),
        # Mix of valid color specs
        ([[1, 0, 0, 0.5]]*3 + [[1, 0, 0]], None),
        ([[1, 0, 0, 0.5], "red", "0.0"], "shape"),
        ([[1, 0, 0, 0.5], "red", "0.0", "C5"], None),
        ([[1, 0, 0, 0.5], "red", "0.0", "C5", [0, 1, 0]], "shape"),
        # Mix of valid and non valid color specs
        ([[1, 0, 0, 0.5], "red", "jaune"], "conversion"),
        ([[1, 0, 0, 0.5], "red", "0.0", "jaune"], "conversion"),
        ([[1, 0, 0, 0.5], "red", "0.0", "C5", "jaune"], "conversion"),
    ]

    @pytest.mark.parametrize('c_case, re_key', params_test_scatter_c)
    def test_scatter_c(self, c_case, re_key):
        def get_next_color():
            return 'blue'  # currently unused

        xsize = 4
        # Additional checking of *c* (introduced in #11383).
        REGEXP = {
            "shape": "^'c' argument has [0-9]+ elements",  # shape mismatch
            "conversion": "^'c' argument must be a color",  # bad vals
            }

        if re_key is None:
            mpl.axes.Axes._parse_scatter_color_args(
                c=c_case, edgecolors="black", kwargs={}, xsize=xsize,
                get_next_color_func=get_next_color)
        else:
            with pytest.raises(ValueError, match=REGEXP[re_key]):
                mpl.axes.Axes._parse_scatter_color_args(
                    c=c_case, edgecolors="black", kwargs={}, xsize=xsize,
                    get_next_color_func=get_next_color)

    @mpl.style.context('default')
    @check_figures_equal(extensions=["png"])
    def test_scatter_single_color_c(self, fig_test, fig_ref):
        rgb = [[1, 0.5, 0.05]]
        rgba = [[1, 0.5, 0.05, .5]]

        # set via color kwarg
        ax_ref = fig_ref.subplots()
        ax_ref.scatter(np.ones(3), range(3), color=rgb)
        ax_ref.scatter(np.ones(4)*2, range(4), color=rgba)

        # set via broadcasting via c
        ax_test = fig_test.subplots()
        ax_test.scatter(np.ones(3), range(3), c=rgb)
        ax_test.scatter(np.ones(4)*2, range(4), c=rgba)

    def test_scatter_linewidths(self):
        x = np.arange(5)

        fig, ax = plt.subplots()
        for i in range(3):
            pc = ax.scatter(x, np.full(5, i), c=f'C{i}', marker='x', s=100,
                            linewidths=i + 1)
            assert pc.get_linewidths() == i + 1

        pc = ax.scatter(x, np.full(5, 3), c='C3', marker='x', s=100,
                        linewidths=[*range(1, 5), None])
        assert_array_equal(pc.get_linewidths(),
                           [*range(1, 5), mpl.rcParams['lines.linewidth']])


def _params(c=None, xsize=2, *, edgecolors=None, **kwargs):
    return (c, edgecolors, kwargs if kwargs is not None else {}, xsize)
_result = namedtuple('_result', 'c, colors')


@pytest.mark.parametrize(
    'params, expected_result',
    [(_params(),
      _result(c='b', colors=np.array([[0, 0, 1, 1]]))),
     (_params(c='r'),
      _result(c='r', colors=np.array([[1, 0, 0, 1]]))),
     (_params(c='r', colors='b'),
      _result(c='r', colors=np.array([[1, 0, 0, 1]]))),
     # color
     (_params(color='b'),
      _result(c='b', colors=np.array([[0, 0, 1, 1]]))),
     (_params(color=['b', 'g']),
      _result(c=['b', 'g'], colors=np.array([[0, 0, 1, 1], [0, .5, 0, 1]]))),
     ])
def test_parse_scatter_color_args(params, expected_result):
    def get_next_color():
        return 'blue'  # currently unused

    c, colors, _edgecolors = mpl.axes.Axes._parse_scatter_color_args(
        *params, get_next_color_func=get_next_color)
    assert c == expected_result.c
    assert_allclose(colors, expected_result.colors)

del _params
del _result


@pytest.mark.parametrize(
    'kwargs, expected_edgecolors',
    [(dict(), None),
     (dict(c='b'), None),
     (dict(edgecolors='r'), 'r'),
     (dict(edgecolors=['r', 'g']), ['r', 'g']),
     (dict(edgecolor='r'), 'r'),
     (dict(edgecolors='face'), 'face'),
     (dict(edgecolors='none'), 'none'),
     (dict(edgecolor='r', edgecolors='g'), 'r'),
     (dict(c='b', edgecolor='r', edgecolors='g'), 'r'),
     (dict(color='r'), 'r'),
     (dict(color='r', edgecolor='g'), 'g'),
     ])
def test_parse_scatter_color_args_edgecolors(kwargs, expected_edgecolors):
    def get_next_color():
        return 'blue'  # currently unused

    c = kwargs.pop('c', None)
    edgecolors = kwargs.pop('edgecolors', None)
    _, _, result_edgecolors = \
        mpl.axes.Axes._parse_scatter_color_args(
            c, edgecolors, kwargs, xsize=2, get_next_color_func=get_next_color)
    assert result_edgecolors == expected_edgecolors


def test_parse_scatter_color_args_error():
    def get_next_color():
        return 'blue'  # currently unused

    with pytest.raises(ValueError,
                       match="RGBA values should be within 0-1 range"):
        c = np.array([[0.1, 0.2, 0.7], [0.2, 0.4, 1.4]])  # value > 1
        mpl.axes.Axes._parse_scatter_color_args(
            c, None, kwargs={}, xsize=2, get_next_color_func=get_next_color)


def test_as_mpl_axes_api():
    # tests the _as_mpl_axes api
    from matplotlib.projections.polar import PolarAxes

    class Polar:
        def __init__(self):
            self.theta_offset = 0

        def _as_mpl_axes(self):
            # implement the matplotlib axes interface
            return PolarAxes, {'theta_offset': self.theta_offset}

    prj = Polar()
    prj2 = Polar()
    prj2.theta_offset = np.pi
    prj3 = Polar()

    # testing axes creation with plt.axes
    ax = plt.axes([0, 0, 1, 1], projection=prj)
    assert type(ax) == PolarAxes
    with pytest.warns(
            MatplotlibDeprecationWarning,
            match=r'Calling gca\(\) with keyword arguments was deprecated'):
        ax_via_gca = plt.gca(projection=prj)
    assert ax_via_gca is ax
    plt.close()

    # testing axes creation with gca
    with pytest.warns(
            MatplotlibDeprecationWarning,
            match=r'Calling gca\(\) with keyword arguments was deprecated'):
        ax = plt.gca(projection=prj)
    assert type(ax) == mpl.axes._subplots.subplot_class_factory(PolarAxes)
    with pytest.warns(
            MatplotlibDeprecationWarning,
            match=r'Calling gca\(\) with keyword arguments was deprecated'):
        ax_via_gca = plt.gca(projection=prj)
    assert ax_via_gca is ax
    # try getting the axes given a different polar projection
    with pytest.warns(
            MatplotlibDeprecationWarning,
            match=r'Calling gca\(\) with keyword arguments was deprecated'):
        ax_via_gca = plt.gca(projection=prj2)
    assert ax_via_gca is ax
    assert ax.get_theta_offset() == 0
    # try getting the axes given an == (not is) polar projection
    with pytest.warns(
            MatplotlibDeprecationWarning,
            match=r'Calling gca\(\) with keyword arguments was deprecated'):
        ax_via_gca = plt.gca(projection=prj3)
    assert ax_via_gca is ax
    plt.close()

    # testing axes creation with subplot
    ax = plt.subplot(121, projection=prj)
    assert type(ax) == mpl.axes._subplots.subplot_class_factory(PolarAxes)
    plt.close()


def test_pyplot_axes():
    # test focusing of Axes in other Figure
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    plt.sca(ax1)
    assert ax1 is plt.gca()
    assert fig1 is plt.gcf()
    plt.close(fig1)
    plt.close(fig2)


@image_comparison(['log_scales'])
def test_log_scales():
    fig, ax = plt.subplots()
    ax.plot(np.log(np.linspace(0.1, 100)))
    ax.set_yscale('log', base=5.5)
    ax.invert_yaxis()
    ax.set_xscale('log', base=9.0)


def test_log_scales_no_data():
    _, ax = plt.subplots()
    ax.set(xscale="log", yscale="log")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    assert ax.get_xlim() == ax.get_ylim() == (1, 10)


def test_log_scales_invalid():
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    with pytest.warns(UserWarning, match='Attempted to set non-positive'):
        ax.set_xlim(-1, 10)
    ax.set_yscale('log')
    with pytest.warns(UserWarning, match='Attempted to set non-positive'):
        ax.set_ylim(-1, 10)


@image_comparison(['stackplot_test_image', 'stackplot_test_image'])
def test_stackplot():
    fig = plt.figure()
    x = np.linspace(0, 10, 10)
    y1 = 1.0 * x
    y2 = 2.0 * x + 1
    y3 = 3.0 * x + 2
    ax = fig.add_subplot(1, 1, 1)
    ax.stackplot(x, y1, y2, y3)
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 70))

    # Reuse testcase from above for a labeled data test
    data = {"x": x, "y1": y1, "y2": y2, "y3": y3}
    fig, ax = plt.subplots()
    ax.stackplot("x", "y1", "y2", "y3", data=data)
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 70))


@image_comparison(['stackplot_test_baseline'], remove_text=True)
def test_stackplot_baseline():
    np.random.seed(0)

    def layers(n, m):
        a = np.zeros((m, n))
        for i in range(n):
            for j in range(5):
                x = 1 / (.1 + np.random.random())
                y = 2 * np.random.random() - .5
                z = 10 / (.1 + np.random.random())
                a[:, i] += x * np.exp(-((np.arange(m) / m - y) * z) ** 2)
        return a

    d = layers(3, 100)
    d[50, :] = 0  # test for fixed weighted wiggle (issue #6313)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].stackplot(range(100), d.T, baseline='zero')
    axs[0, 1].stackplot(range(100), d.T, baseline='sym')
    axs[1, 0].stackplot(range(100), d.T, baseline='wiggle')
    axs[1, 1].stackplot(range(100), d.T, baseline='weighted_wiggle')


def _bxp_test_helper(
        stats_kwargs={}, transform_stats=lambda s: s, bxp_kwargs={}):
    np.random.seed(937)
    logstats = mpl.cbook.boxplot_stats(
        np.random.lognormal(mean=1.25, sigma=1., size=(37, 4)), **stats_kwargs)
    fig, ax = plt.subplots()
    if bxp_kwargs.get('vert', True):
        ax.set_yscale('log')
    else:
        ax.set_xscale('log')
    # Work around baseline images generate back when bxp did not respect the
    # boxplot.boxprops.linewidth rcParam when patch_artist is False.
    if not bxp_kwargs.get('patch_artist', False):
        mpl.rcParams['boxplot.boxprops.linewidth'] = \
            mpl.rcParams['lines.linewidth']
    ax.bxp(transform_stats(logstats), **bxp_kwargs)


@image_comparison(['bxp_baseline.png'],
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_baseline():
    _bxp_test_helper()


@image_comparison(['bxp_rangewhis.png'],
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_rangewhis():
    _bxp_test_helper(stats_kwargs=dict(whis=[0, 100]))


@image_comparison(['bxp_percentilewhis.png'],
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_percentilewhis():
    _bxp_test_helper(stats_kwargs=dict(whis=[5, 95]))


@image_comparison(['bxp_with_xlabels.png'],
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_with_xlabels():
    def transform(stats):
        for s, label in zip(stats, list('ABCD')):
            s['label'] = label
        return stats

    _bxp_test_helper(transform_stats=transform)


@image_comparison(['bxp_horizontal.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default',
                  tol=0.1)
def test_bxp_horizontal():
    _bxp_test_helper(bxp_kwargs=dict(vert=False))


@image_comparison(['bxp_with_ylabels.png'],
                  savefig_kwarg={'dpi': 40},
                  style='default',
                  tol=0.1)
def test_bxp_with_ylabels():
    def transform(stats):
        for s, label in zip(stats, list('ABCD')):
            s['label'] = label
        return stats

    _bxp_test_helper(transform_stats=transform, bxp_kwargs=dict(vert=False))


@image_comparison(['bxp_patchartist.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_patchartist():
    _bxp_test_helper(bxp_kwargs=dict(patch_artist=True))


@image_comparison(['bxp_custompatchartist.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 100},
                  style='default')
def test_bxp_custompatchartist():
    _bxp_test_helper(bxp_kwargs=dict(
        patch_artist=True,
        boxprops=dict(facecolor='yellow', edgecolor='green', ls=':')))


@image_comparison(['bxp_customoutlier.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_customoutlier():
    _bxp_test_helper(bxp_kwargs=dict(
        flierprops=dict(linestyle='none', marker='d', mfc='g')))


@image_comparison(['bxp_withmean_custompoint.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_showcustommean():
    _bxp_test_helper(bxp_kwargs=dict(
        showmeans=True,
        meanprops=dict(linestyle='none', marker='d', mfc='green'),
    ))


@image_comparison(['bxp_custombox.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_custombox():
    _bxp_test_helper(bxp_kwargs=dict(
        boxprops=dict(linestyle='--', color='b', lw=3)))


@image_comparison(['bxp_custommedian.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_custommedian():
    _bxp_test_helper(bxp_kwargs=dict(
        medianprops=dict(linestyle='--', color='b', lw=3)))


@image_comparison(['bxp_customcap.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_customcap():
    _bxp_test_helper(bxp_kwargs=dict(
        capprops=dict(linestyle='--', color='g', lw=3)))


@image_comparison(['bxp_customwhisker.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_customwhisker():
    _bxp_test_helper(bxp_kwargs=dict(
        whiskerprops=dict(linestyle='-', color='m', lw=3)))


@image_comparison(['bxp_withnotch.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_shownotches():
    _bxp_test_helper(bxp_kwargs=dict(shownotches=True))


@image_comparison(['bxp_nocaps.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_nocaps():
    _bxp_test_helper(bxp_kwargs=dict(showcaps=False))


@image_comparison(['bxp_nobox.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_nobox():
    _bxp_test_helper(bxp_kwargs=dict(showbox=False))


@image_comparison(['bxp_no_flier_stats.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_no_flier_stats():
    def transform(stats):
        for s in stats:
            s.pop('fliers', None)
        return stats

    _bxp_test_helper(transform_stats=transform,
                     bxp_kwargs=dict(showfliers=False))


@image_comparison(['bxp_withmean_point.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_showmean():
    _bxp_test_helper(bxp_kwargs=dict(showmeans=True, meanline=False))


@image_comparison(['bxp_withmean_line.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_showmeanasline():
    _bxp_test_helper(bxp_kwargs=dict(showmeans=True, meanline=True))


@image_comparison(['bxp_scalarwidth.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_scalarwidth():
    _bxp_test_helper(bxp_kwargs=dict(widths=.25))


@image_comparison(['bxp_customwidths.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_customwidths():
    _bxp_test_helper(bxp_kwargs=dict(widths=[0.10, 0.25, 0.65, 0.85]))


@image_comparison(['bxp_custompositions.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_bxp_custompositions():
    _bxp_test_helper(bxp_kwargs=dict(positions=[1, 5, 6, 7]))


def test_bxp_bad_widths():
    with pytest.raises(ValueError):
        _bxp_test_helper(bxp_kwargs=dict(widths=[1]))


def test_bxp_bad_positions():
    with pytest.raises(ValueError):
        _bxp_test_helper(bxp_kwargs=dict(positions=[2, 3]))


@image_comparison(['boxplot', 'boxplot'], tol=1.28, style='default')
def test_boxplot():
    # Randomness used for bootstrapping.
    np.random.seed(937)

    x = np.linspace(-7, 7, 140)
    x = np.hstack([-25, x, 25])
    fig, ax = plt.subplots()

    ax.boxplot([x, x], bootstrap=10000, notch=1)
    ax.set_ylim((-30, 30))

    # Reuse testcase from above for a labeled data test
    data = {"x": [x, x]}
    fig, ax = plt.subplots()
    ax.boxplot("x", bootstrap=10000, notch=1, data=data)
    ax.set_ylim((-30, 30))


@image_comparison(['boxplot_sym2.png'], remove_text=True, style='default')
def test_boxplot_sym2():
    # Randomness used for bootstrapping.
    np.random.seed(937)

    x = np.linspace(-7, 7, 140)
    x = np.hstack([-25, x, 25])
    fig, [ax1, ax2] = plt.subplots(1, 2)

    ax1.boxplot([x, x], bootstrap=10000, sym='^')
    ax1.set_ylim((-30, 30))

    ax2.boxplot([x, x], bootstrap=10000, sym='g')
    ax2.set_ylim((-30, 30))


@image_comparison(['boxplot_sym.png'],
                  remove_text=True,
                  savefig_kwarg={'dpi': 40},
                  style='default')
def test_boxplot_sym():
    x = np.linspace(-7, 7, 140)
    x = np.hstack([-25, x, 25])
    fig, ax = plt.subplots()

    ax.boxplot([x, x], sym='gs')
    ax.set_ylim((-30, 30))


@image_comparison(['boxplot_autorange_false_whiskers.png',
                   'boxplot_autorange_true_whiskers.png'],
                  style='default')
def test_boxplot_autorange_whiskers():
    # Randomness used for bootstrapping.
    np.random.seed(937)

    x = np.ones(140)
    x = np.hstack([0, x, 2])

    fig1, ax1 = plt.subplots()
    ax1.boxplot([x, x], bootstrap=10000, notch=1)
    ax1.set_ylim((-5, 5))

    fig2, ax2 = plt.subplots()
    ax2.boxplot([x, x], bootstrap=10000, notch=1, autorange=True)
    ax2.set_ylim((-5, 5))


def _rc_test_bxp_helper(ax, rc_dict):
    x = np.linspace(-7, 7, 140)
    x = np.hstack([-25, x, 25])
    with matplotlib.rc_context(rc_dict):
        ax.boxplot([x, x])
    return ax


@image_comparison(['boxplot_rc_parameters'],
                  savefig_kwarg={'dpi': 100}, remove_text=True,
                  tol=1, style='default')
def test_boxplot_rc_parameters():
    # Randomness used for bootstrapping.
    np.random.seed(937)

    fig, ax = plt.subplots(3)

    rc_axis0 = {
        'boxplot.notch': True,
        'boxplot.whiskers': [5, 95],
        'boxplot.bootstrap': 10000,

        'boxplot.flierprops.color': 'b',
        'boxplot.flierprops.marker': 'o',
        'boxplot.flierprops.markerfacecolor': 'g',
        'boxplot.flierprops.markeredgecolor': 'b',
        'boxplot.flierprops.markersize': 5,
        'boxplot.flierprops.linestyle': '--',
        'boxplot.flierprops.linewidth': 2.0,

        'boxplot.boxprops.color': 'r',
        'boxplot.boxprops.linewidth': 2.0,
        'boxplot.boxprops.linestyle': '--',

        'boxplot.capprops.color': 'c',
        'boxplot.capprops.linewidth': 2.0,
        'boxplot.capprops.linestyle': '--',

        'boxplot.medianprops.color': 'k',
        'boxplot.medianprops.linewidth': 2.0,
        'boxplot.medianprops.linestyle': '--',
    }

    rc_axis1 = {
        'boxplot.vertical': False,
        'boxplot.whiskers': [0, 100],
        'boxplot.patchartist': True,
    }

    rc_axis2 = {
        'boxplot.whiskers': 2.0,
        'boxplot.showcaps': False,
        'boxplot.showbox': False,
        'boxplot.showfliers': False,
        'boxplot.showmeans': True,
        'boxplot.meanline': True,

        'boxplot.meanprops.color': 'c',
        'boxplot.meanprops.linewidth': 2.0,
        'boxplot.meanprops.linestyle': '--',

        'boxplot.whiskerprops.color': 'r',
        'boxplot.whiskerprops.linewidth': 2.0,
        'boxplot.whiskerprops.linestyle': '-.',
    }
    dict_list = [rc_axis0, rc_axis1, rc_axis2]
    for axis, rc_axis in zip(ax, dict_list):
        _rc_test_bxp_helper(axis, rc_axis)

    assert (matplotlib.patches.PathPatch in
            [type(t) for t in ax[1].get_children()])


@image_comparison(['boxplot_with_CIarray.png'],
                  remove_text=True, savefig_kwarg={'dpi': 40}, style='default')
def test_boxplot_with_CIarray():
    # Randomness used for bootstrapping.
    np.random.seed(937)

    x = np.linspace(-7, 7, 140)
    x = np.hstack([-25, x, 25])
    fig, ax = plt.subplots()
    CIs = np.array([[-1.5, 3.], [-1., 3.5]])

    # show a boxplot with Matplotlib medians and confidence intervals, and
    # another with manual values
    ax.boxplot([x, x], bootstrap=10000, usermedians=[None, 1.0],
               conf_intervals=CIs, notch=1)
    ax.set_ylim((-30, 30))


@image_comparison(['boxplot_no_inverted_whisker.png'],
                  remove_text=True, savefig_kwarg={'dpi': 40}, style='default')
def test_boxplot_no_weird_whisker():
    x = np.array([3, 9000, 150, 88, 350, 200000, 1400, 960],
                 dtype=np.float64)
    ax1 = plt.axes()
    ax1.boxplot(x)
    ax1.set_yscale('log')
    ax1.yaxis.grid(False, which='minor')
    ax1.xaxis.grid(False)


def test_boxplot_bad_medians():
    x = np.linspace(-7, 7, 140)
    x = np.hstack([-25, x, 25])
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.boxplot(x, usermedians=[1, 2])
    with pytest.raises(ValueError):
        ax.boxplot([x, x], usermedians=[[1, 2], [1, 2]])


def test_boxplot_bad_ci():
    x = np.linspace(-7, 7, 140)
    x = np.hstack([-25, x, 25])
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.boxplot([x, x], conf_intervals=[[1, 2]])
    with pytest.raises(ValueError):
        ax.boxplot([x, x], conf_intervals=[[1, 2], [1]])


def test_boxplot_zorder():
    x = np.arange(10)
    fix, ax = plt.subplots()
    assert ax.boxplot(x)['boxes'][0].get_zorder() == 2
    assert ax.boxplot(x, zorder=10)['boxes'][0].get_zorder() == 10


def test_boxplot_marker_behavior():
    plt.rcParams['lines.marker'] = 's'
    plt.rcParams['boxplot.flierprops.marker'] = 'o'
    plt.rcParams['boxplot.meanprops.marker'] = '^'
    fig, ax = plt.subplots()
    test_data = np.arange(100)
    test_data[-1] = 150  # a flier point
    bxp_handle = ax.boxplot(test_data, showmeans=True)
    for bxp_lines in ['whiskers', 'caps', 'boxes', 'medians']:
        for each_line in bxp_handle[bxp_lines]:
            # Ensure that the rcParams['lines.marker'] is overridden by ''
            assert each_line.get_marker() == ''

    # Ensure that markers for fliers and means aren't overridden with ''
    assert bxp_handle['fliers'][0].get_marker() == 'o'
    assert bxp_handle['means'][0].get_marker() == '^'


@image_comparison(['boxplot_mod_artists_after_plotting.png'],
                  remove_text=True, savefig_kwarg={'dpi': 40}, style='default')
def test_boxplot_mod_artist_after_plotting():
    x = [0.15, 0.11, 0.06, 0.06, 0.12, 0.56, -0.56]
    fig, ax = plt.subplots()
    bp = ax.boxplot(x, sym="o")
    for key in bp:
        for obj in bp[key]:
            obj.set_color('green')


@image_comparison(['violinplot_vert_baseline.png',
                   'violinplot_vert_baseline.png'])
def test_vert_violinplot_baseline():
    # First 9 digits of frac(sqrt(2))
    np.random.seed(414213562)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax = plt.axes()
    ax.violinplot(data, positions=range(4), showmeans=0, showextrema=0,
                  showmedians=0)

    # Reuse testcase from above for a labeled data test
    data = {"d": data}
    fig, ax = plt.subplots()
    ax.violinplot("d", positions=range(4), showmeans=0, showextrema=0,
                  showmedians=0, data=data)


@image_comparison(['violinplot_vert_showmeans.png'])
def test_vert_violinplot_showmeans():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(3))
    np.random.seed(732050807)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), showmeans=1, showextrema=0,
                  showmedians=0)


@image_comparison(['violinplot_vert_showextrema.png'])
def test_vert_violinplot_showextrema():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(5))
    np.random.seed(236067977)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), showmeans=0, showextrema=1,
                  showmedians=0)


@image_comparison(['violinplot_vert_showmedians.png'])
def test_vert_violinplot_showmedians():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(7))
    np.random.seed(645751311)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), showmeans=0, showextrema=0,
                  showmedians=1)


@image_comparison(['violinplot_vert_showall.png'])
def test_vert_violinplot_showall():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(11))
    np.random.seed(316624790)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), showmeans=1, showextrema=1,
                  showmedians=1,
                  quantiles=[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])


@image_comparison(['violinplot_vert_custompoints_10.png'])
def test_vert_violinplot_custompoints_10():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(13))
    np.random.seed(605551275)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), showmeans=0, showextrema=0,
                  showmedians=0, points=10)


@image_comparison(['violinplot_vert_custompoints_200.png'])
def test_vert_violinplot_custompoints_200():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(17))
    np.random.seed(123105625)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), showmeans=0, showextrema=0,
                  showmedians=0, points=200)


@image_comparison(['violinplot_horiz_baseline.png'])
def test_horiz_violinplot_baseline():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(19))
    np.random.seed(358898943)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), vert=False, showmeans=0,
                  showextrema=0, showmedians=0)


@image_comparison(['violinplot_horiz_showmedians.png'])
def test_horiz_violinplot_showmedians():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(23))
    np.random.seed(795831523)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), vert=False, showmeans=0,
                  showextrema=0, showmedians=1)


@image_comparison(['violinplot_horiz_showmeans.png'])
def test_horiz_violinplot_showmeans():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(29))
    np.random.seed(385164807)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), vert=False, showmeans=1,
                  showextrema=0, showmedians=0)


@image_comparison(['violinplot_horiz_showextrema.png'])
def test_horiz_violinplot_showextrema():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(31))
    np.random.seed(567764362)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), vert=False, showmeans=0,
                  showextrema=1, showmedians=0)


@image_comparison(['violinplot_horiz_showall.png'])
def test_horiz_violinplot_showall():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(37))
    np.random.seed(82762530)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), vert=False, showmeans=1,
                  showextrema=1, showmedians=1,
                  quantiles=[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])


@image_comparison(['violinplot_horiz_custompoints_10.png'])
def test_horiz_violinplot_custompoints_10():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(41))
    np.random.seed(403124237)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), vert=False, showmeans=0,
                  showextrema=0, showmedians=0, points=10)


@image_comparison(['violinplot_horiz_custompoints_200.png'])
def test_horiz_violinplot_custompoints_200():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(43))
    np.random.seed(557438524)
    data = [np.random.normal(size=100) for _ in range(4)]
    ax.violinplot(data, positions=range(4), vert=False, showmeans=0,
                  showextrema=0, showmedians=0, points=200)


def test_violinplot_bad_positions():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(47))
    np.random.seed(855654600)
    data = [np.random.normal(size=100) for _ in range(4)]
    with pytest.raises(ValueError):
        ax.violinplot(data, positions=range(5))


def test_violinplot_bad_widths():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(53))
    np.random.seed(280109889)
    data = [np.random.normal(size=100) for _ in range(4)]
    with pytest.raises(ValueError):
        ax.violinplot(data, positions=range(4), widths=[1, 2, 3])


def test_violinplot_bad_quantiles():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(73))
    np.random.seed(544003745)
    data = [np.random.normal(size=100)]

    # Different size quantile list and plots
    with pytest.raises(ValueError):
        ax.violinplot(data, quantiles=[[0.1, 0.2], [0.5, 0.7]])


def test_violinplot_outofrange_quantiles():
    ax = plt.axes()
    # First 9 digits of frac(sqrt(79))
    np.random.seed(888194417)
    data = [np.random.normal(size=100)]

    # Quantile value above 100
    with pytest.raises(ValueError):
        ax.violinplot(data, quantiles=[[0.1, 0.2, 0.3, 1.05]])

    # Quantile value below 0
    with pytest.raises(ValueError):
        ax.violinplot(data, quantiles=[[-0.05, 0.2, 0.3, 0.75]])


@check_figures_equal(extensions=["png"])
def test_violinplot_single_list_quantiles(fig_test, fig_ref):
    # Ensures quantile list for 1D can be passed in as single list
    # First 9 digits of frac(sqrt(83))
    np.random.seed(110433579)
    data = [np.random.normal(size=100)]

    # Test image
    ax = fig_test.subplots()
    ax.violinplot(data, quantiles=[0.1, 0.3, 0.9])

    # Reference image
    ax = fig_ref.subplots()
    ax.violinplot(data, quantiles=[[0.1, 0.3, 0.9]])


@check_figures_equal(extensions=["png"])
def test_violinplot_pandas_series(fig_test, fig_ref, pd):
    np.random.seed(110433579)
    s1 = pd.Series(np.random.normal(size=7), index=[9, 8, 7, 6, 5, 4, 3])
    s2 = pd.Series(np.random.normal(size=9), index=list('ABCDEFGHI'))
    s3 = pd.Series(np.random.normal(size=11))
    fig_test.subplots().violinplot([s1, s2, s3])
    fig_ref.subplots().violinplot([s1.values, s2.values, s3.values])


def test_manage_xticks():
    _, ax = plt.subplots()
    ax.set_xlim(0, 4)
    old_xlim = ax.get_xlim()
    np.random.seed(0)
    y1 = np.random.normal(10, 3, 20)
    y2 = np.random.normal(3, 1, 20)
    ax.boxplot([y1, y2], positions=[1, 2], manage_ticks=False)
    new_xlim = ax.get_xlim()
    assert_array_equal(old_xlim, new_xlim)


def test_boxplot_not_single():
    fig, ax = plt.subplots()
    ax.boxplot(np.random.rand(100), positions=[3])
    ax.boxplot(np.random.rand(100), positions=[5])
    fig.canvas.draw()
    assert ax.get_xlim() == (2.5, 5.5)
    assert list(ax.get_xticks()) == [3, 5]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["3", "5"]


def test_tick_space_size_0():
    # allow font size to be zero, which affects ticks when there is
    # no other text in the figure.
    plt.plot([0, 1], [0, 1])
    matplotlib.rcParams.update({'font.size': 0})
    b = io.BytesIO()
    plt.savefig(b, dpi=80, format='raw')


@image_comparison(['errorbar_basic', 'errorbar_mixed', 'errorbar_basic'])
def test_errorbar():
    x = np.arange(0.1, 4, 0.5)
    y = np.exp(-x)

    yerr = 0.1 + 0.2*np.sqrt(x)
    xerr = 0.1 + yerr

    # First illustrate basic pyplot interface, using defaults where possible.
    fig = plt.figure()
    ax = fig.gca()
    ax.errorbar(x, y, xerr=0.2, yerr=0.4)
    ax.set_title("Simplest errorbars, 0.2 in x, 0.4 in y")

    # Now switch to a more OO interface to exercise more features.
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0, 0]
    ax.errorbar(x, y, yerr=yerr, fmt='o')
    ax.set_title('Vert. symmetric')

    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)

    ax = axs[0, 1]
    ax.errorbar(x, y, xerr=xerr, fmt='o', alpha=0.4)
    ax.set_title('Hor. symmetric w/ alpha')

    ax = axs[1, 0]
    ax.errorbar(x, y, yerr=[yerr, 2*yerr], xerr=[xerr, 2*xerr], fmt='--o')
    ax.set_title('H, V asymmetric')

    ax = axs[1, 1]
    ax.set_yscale('log')
    # Here we have to be careful to keep all y values positive:
    ylower = np.maximum(1e-2, y - yerr)
    yerr_lower = y - ylower

    ax.errorbar(x, y, yerr=[yerr_lower, 2*yerr], xerr=xerr,
                fmt='o', ecolor='g', capthick=2)
    ax.set_title('Mixed sym., log y')

    fig.suptitle('Variable errorbars')

    # Reuse the first testcase from above for a labeled data test
    data = {"x": x, "y": y}
    fig = plt.figure()
    ax = fig.gca()
    ax.errorbar("x", "y", xerr=0.2, yerr=0.4, data=data)
    ax.set_title("Simplest errorbars, 0.2 in x, 0.4 in y")


def test_errorbar_colorcycle():

    f, ax = plt.subplots()
    x = np.arange(10)
    y = 2*x

    e1, _, _ = ax.errorbar(x, y, c=None)
    e2, _, _ = ax.errorbar(x, 2*y, c=None)
    ln1, = ax.plot(x, 4*y)

    assert mcolors.to_rgba(e1.get_color()) == mcolors.to_rgba('C0')
    assert mcolors.to_rgba(e2.get_color()) == mcolors.to_rgba('C1')
    assert mcolors.to_rgba(ln1.get_color()) == mcolors.to_rgba('C2')


@check_figures_equal()
def test_errorbar_cycle_ecolor(fig_test, fig_ref):
    x = np.arange(0.1, 4, 0.5)
    y = [np.exp(-x+n) for n in range(4)]

    axt = fig_test.subplots()
    axr = fig_ref.subplots()

    for yi, color in zip(y, ['C0', 'C1', 'C2', 'C3']):
        axt.errorbar(x, yi, yerr=(yi * 0.25), linestyle='-',
                     marker='o', ecolor='black')
        axr.errorbar(x, yi, yerr=(yi * 0.25), linestyle='-',
                     marker='o', color=color, ecolor='black')


def test_errorbar_shape():
    fig = plt.figure()
    ax = fig.gca()

    x = np.arange(0.1, 4, 0.5)
    y = np.exp(-x)
    yerr1 = 0.1 + 0.2*np.sqrt(x)
    yerr = np.vstack((yerr1, 2*yerr1)).T
    xerr = 0.1 + yerr

    with pytest.raises(ValueError):
        ax.errorbar(x, y, yerr=yerr, fmt='o')
    with pytest.raises(ValueError):
        ax.errorbar(x, y, xerr=xerr, fmt='o')
    with pytest.raises(ValueError):
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='o')


@image_comparison(['errorbar_limits'])
def test_errorbar_limits():
    x = np.arange(0.5, 5.5, 0.5)
    y = np.exp(-x)
    xerr = 0.1
    yerr = 0.2
    ls = 'dotted'

    fig, ax = plt.subplots()

    # standard error bars
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, ls=ls, color='blue')

    # including upper limits
    uplims = np.zeros_like(x)
    uplims[[1, 5, 9]] = True
    ax.errorbar(x, y+0.5, xerr=xerr, yerr=yerr, uplims=uplims, ls=ls,
                color='green')

    # including lower limits
    lolims = np.zeros_like(x)
    lolims[[2, 4, 8]] = True
    ax.errorbar(x, y+1.0, xerr=xerr, yerr=yerr, lolims=lolims, ls=ls,
                color='red')

    # including upper and lower limits
    ax.errorbar(x, y+1.5, marker='o', ms=8, xerr=xerr, yerr=yerr,
                lolims=lolims, uplims=uplims, ls=ls, color='magenta')

    # including xlower and xupper limits
    xerr = 0.2
    yerr = np.full_like(x, 0.2)
    yerr[[3, 6]] = 0.3
    xlolims = lolims
    xuplims = uplims
    lolims = np.zeros_like(x)
    uplims = np.zeros_like(x)
    lolims[[6]] = True
    uplims[[3]] = True
    ax.errorbar(x, y+2.1, marker='o', ms=8, xerr=xerr, yerr=yerr,
                xlolims=xlolims, xuplims=xuplims, uplims=uplims,
                lolims=lolims, ls='none', mec='blue', capsize=0,
                color='cyan')
    ax.set_xlim((0, 5.5))
    ax.set_title('Errorbar upper and lower limits')


def test_errobar_nonefmt():
    # Check that passing 'none' as a format still plots errorbars
    x = np.arange(5)
    y = np.arange(5)

    plotline, _, barlines = plt.errorbar(x, y, xerr=1, yerr=1, fmt='none')
    assert plotline is None
    for errbar in barlines:
        assert np.all(errbar.get_color() == mcolors.to_rgba('C0'))


def test_errorbar_line_specific_kwargs():
    # Check that passing line-specific keyword arguments will not result in
    # errors.
    x = np.arange(5)
    y = np.arange(5)

    plotline, _, _ = plt.errorbar(x, y, xerr=1, yerr=1, ls='None',
                                  marker='s', fillstyle='full',
                                  drawstyle='steps-mid',
                                  dash_capstyle='round',
                                  dash_joinstyle='miter',
                                  solid_capstyle='butt',
                                  solid_joinstyle='bevel')
    assert plotline.get_fillstyle() == 'full'
    assert plotline.get_drawstyle() == 'steps-mid'


@check_figures_equal(extensions=['png'])
def test_errorbar_with_prop_cycle(fig_test, fig_ref):
    ax = fig_ref.subplots()
    ax.errorbar(x=[2, 4, 10], y=[0, 1, 2], yerr=0.5,
                ls='--', marker='s', mfc='k')
    ax.errorbar(x=[2, 4, 10], y=[2, 3, 4], yerr=0.5, color='tab:green',
                ls=':', marker='s', mfc='y')
    ax.errorbar(x=[2, 4, 10], y=[4, 5, 6], yerr=0.5, fmt='tab:blue',
                ls='-.', marker='o', mfc='c')
    ax.set_xlim(1, 11)

    _cycle = cycler(ls=['--', ':', '-.'], marker=['s', 's', 'o'],
                    mfc=['k', 'y', 'c'], color=['b', 'g', 'r'])
    plt.rc("axes", prop_cycle=_cycle)
    ax = fig_test.subplots()
    ax.errorbar(x=[2, 4, 10], y=[0, 1, 2], yerr=0.5)
    ax.errorbar(x=[2, 4, 10], y=[2, 3, 4], yerr=0.5, color='tab:green')
    ax.errorbar(x=[2, 4, 10], y=[4, 5, 6], yerr=0.5, fmt='tab:blue')
    ax.set_xlim(1, 11)


def test_errorbar_every_invalid():
    x = np.linspace(0, 1, 15)
    y = x * (1-x)
    yerr = y/6

    ax = plt.figure().subplots()

    with pytest.raises(ValueError, match='not a tuple of two integers'):
        ax.errorbar(x, y, yerr, errorevery=(1, 2, 3))
    with pytest.raises(ValueError, match='not a tuple of two integers'):
        ax.errorbar(x, y, yerr, errorevery=(1.3, 3))
    with pytest.raises(ValueError, match='not a valid NumPy fancy index'):
        ax.errorbar(x, y, yerr, errorevery=[False, True])
    with pytest.raises(ValueError, match='not a recognized value'):
        ax.errorbar(x, y, yerr, errorevery='foobar')


@check_figures_equal()
def test_errorbar_every(fig_test, fig_ref):
    x = np.linspace(0, 1, 15)
    y = x * (1-x)
    yerr = y/6

    ax_ref = fig_ref.subplots()
    ax_test = fig_test.subplots()

    for color, shift in zip('rgbk', [0, 0, 2, 7]):
        y += .02

        # Check errorevery using an explicit offset and step.
        ax_test.errorbar(x, y, yerr, errorevery=(shift, 4),
                         capsize=4, c=color)

        # Using manual errorbars
        # n.b. errorbar draws the main plot at z=2.1 by default
        ax_ref.plot(x, y, c=color, zorder=2.1)
        ax_ref.errorbar(x[shift::4], y[shift::4], yerr[shift::4],
                        capsize=4, c=color, fmt='none')

    # Check that markevery is propagated to line, without affecting errorbars.
    ax_test.errorbar(x, y + 0.1, yerr, markevery=(1, 4), capsize=4, fmt='o')
    ax_ref.plot(x[1::4], y[1::4] + 0.1, 'o', zorder=2.1)
    ax_ref.errorbar(x, y + 0.1, yerr, capsize=4, fmt='none')

    # Check that passing a slice to markevery/errorevery works.
    ax_test.errorbar(x, y + 0.2, yerr, errorevery=slice(2, None, 3),
                     markevery=slice(2, None, 3),
                     capsize=4, c='C0', fmt='o')
    ax_ref.plot(x[2::3], y[2::3] + 0.2, 'o', c='C0', zorder=2.1)
    ax_ref.errorbar(x[2::3], y[2::3] + 0.2, yerr[2::3],
                    capsize=4, c='C0', fmt='none')

    # Check that passing an iterable to markevery/errorevery works.
    ax_test.errorbar(x, y + 0.2, yerr, errorevery=[False, True, False] * 5,
                     markevery=[False, True, False] * 5,
                     capsize=4, c='C1', fmt='o')
    ax_ref.plot(x[1::3], y[1::3] + 0.2, 'o', c='C1', zorder=2.1)
    ax_ref.errorbar(x[1::3], y[1::3] + 0.2, yerr[1::3],
                    capsize=4, c='C1', fmt='none')


@pytest.mark.parametrize('elinewidth', [[1, 2, 3],
                                        np.array([1, 2, 3]),
                                        1])
def test_errorbar_linewidth_type(elinewidth):
    plt.errorbar([1, 2, 3], [1, 2, 3], yerr=[1, 2, 3], elinewidth=elinewidth)


@image_comparison(['hist_stacked_stepfilled', 'hist_stacked_stepfilled'])
def test_hist_stacked_stepfilled():
    # make some data
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    fig, ax = plt.subplots()
    ax.hist((d1, d2), histtype="stepfilled", stacked=True)

    # Reuse testcase from above for a labeled data test
    data = {"x": (d1, d2)}
    fig, ax = plt.subplots()
    ax.hist("x", histtype="stepfilled", stacked=True, data=data)


@image_comparison(['hist_offset'])
def test_hist_offset():
    # make some data
    d1 = np.linspace(0, 10, 50)
    d2 = np.linspace(1, 3, 20)
    fig, ax = plt.subplots()
    ax.hist(d1, bottom=5)
    ax.hist(d2, bottom=15)


@image_comparison(['hist_step.png'], remove_text=True)
def test_hist_step():
    # make some data
    d1 = np.linspace(1, 3, 20)
    fig, ax = plt.subplots()
    ax.hist(d1, histtype="step")
    ax.set_ylim(0, 10)
    ax.set_xlim(-1, 5)


@image_comparison(['hist_step_horiz.png'])
def test_hist_step_horiz():
    # make some data
    d1 = np.linspace(0, 10, 50)
    d2 = np.linspace(1, 3, 20)
    fig, ax = plt.subplots()
    ax.hist((d1, d2), histtype="step", orientation="horizontal")


@image_comparison(['hist_stacked_weights'])
def test_hist_stacked_weighted():
    # make some data
    d1 = np.linspace(0, 10, 50)
    d2 = np.linspace(1, 3, 20)
    w1 = np.linspace(0.01, 3.5, 50)
    w2 = np.linspace(0.05, 2., 20)
    fig, ax = plt.subplots()
    ax.hist((d1, d2), weights=(w1, w2), histtype="stepfilled", stacked=True)


@pytest.mark.parametrize("use_line_collection", [True, False],
                         ids=['w/ line collection', 'w/o line collection'])
@image_comparison(['stem.png'], style='mpl20', remove_text=True)
def test_stem(use_line_collection):
    x = np.linspace(0.1, 2 * np.pi, 100)

    fig, ax = plt.subplots()
    # Label is a single space to force a legend to be drawn, but to avoid any
    # text being drawn
    ax.stem(x, np.cos(x),
            linefmt='C2-.', markerfmt='k+', basefmt='C1-.', label=' ',
            use_line_collection=use_line_collection)
    ax.legend()


def test_stem_args():
    fig, ax = plt.subplots()

    x = list(range(10))
    y = list(range(10))

    # Test the call signatures
    ax.stem(y)
    ax.stem(x, y)
    ax.stem(x, y, 'r--')
    ax.stem(x, y, 'r--', basefmt='b--')


def test_stem_dates():
    fig, ax = plt.subplots(1, 1)
    xs = [dateutil.parser.parse("2013-9-28 11:00:00"),
          dateutil.parser.parse("2013-9-28 12:00:00")]
    ys = [100, 200]
    ax.stem(xs, ys, "*-")


@pytest.mark.parametrize("use_line_collection", [True, False],
                         ids=['w/ line collection', 'w/o line collection'])
@image_comparison(['stem_orientation.png'], style='mpl20', remove_text=True)
def test_stem_orientation(use_line_collection):
    x = np.linspace(0.1, 2*np.pi, 50)

    fig, ax = plt.subplots()
    ax.stem(x, np.cos(x),
            linefmt='C2-.', markerfmt='kx', basefmt='C1-.',
            use_line_collection=use_line_collection, orientation='horizontal')


@image_comparison(['hist_stacked_stepfilled_alpha'])
def test_hist_stacked_stepfilled_alpha():
    # make some data
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    fig, ax = plt.subplots()
    ax.hist((d1, d2), histtype="stepfilled", stacked=True, alpha=0.5)


@image_comparison(['hist_stacked_step'])
def test_hist_stacked_step():
    # make some data
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    fig, ax = plt.subplots()
    ax.hist((d1, d2), histtype="step", stacked=True)


@image_comparison(['hist_stacked_normed'])
def test_hist_stacked_density():
    # make some data
    d1 = np.linspace(1, 3, 20)
    d2 = np.linspace(0, 10, 50)
    fig, ax = plt.subplots()
    ax.hist((d1, d2), stacked=True, density=True)


@image_comparison(['hist_step_bottom.png'], remove_text=True)
def test_hist_step_bottom():
    # make some data
    d1 = np.linspace(1, 3, 20)
    fig, ax = plt.subplots()
    ax.hist(d1, bottom=np.arange(10), histtype="stepfilled")


def test_hist_stepfilled_geometry():
    bins = [0, 1, 2, 3]
    data = [0, 0, 1, 1, 1, 2]
    _, _, (polygon, ) = plt.hist(data,
                                 bins=bins,
                                 histtype='stepfilled')
    xy = [[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 1], [3, 1],
          [3, 0], [2, 0], [2, 0], [1, 0], [1, 0], [0, 0]]
    assert_array_equal(polygon.get_xy(), xy)


def test_hist_step_geometry():
    bins = [0, 1, 2, 3]
    data = [0, 0, 1, 1, 1, 2]
    _, _, (polygon, ) = plt.hist(data,
                                 bins=bins,
                                 histtype='step')
    xy = [[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 1], [3, 1], [3, 0]]
    assert_array_equal(polygon.get_xy(), xy)


def test_hist_stepfilled_bottom_geometry():
    bins = [0, 1, 2, 3]
    data = [0, 0, 1, 1, 1, 2]
    _, _, (polygon, ) = plt.hist(data,
                                 bins=bins,
                                 bottom=[1, 2, 1.5],
                                 histtype='stepfilled')
    xy = [[0, 1], [0, 3], [1, 3], [1, 5], [2, 5], [2, 2.5], [3, 2.5],
          [3, 1.5], [2, 1.5], [2, 2], [1, 2], [1, 1], [0, 1]]
    assert_array_equal(polygon.get_xy(), xy)


def test_hist_step_bottom_geometry():
    bins = [0, 1, 2, 3]
    data = [0, 0, 1, 1, 1, 2]
    _, _, (polygon, ) = plt.hist(data,
                                 bins=bins,
                                 bottom=[1, 2, 1.5],
                                 histtype='step')
    xy = [[0, 1], [0, 3], [1, 3], [1, 5], [2, 5], [2, 2.5], [3, 2.5], [3, 1.5]]
    assert_array_equal(polygon.get_xy(), xy)


def test_hist_stacked_stepfilled_geometry():
    bins = [0, 1, 2, 3]
    data_1 = [0, 0, 1, 1, 1, 2]
    data_2 = [0, 1, 2]
    _, _, patches = plt.hist([data_1, data_2],
                             bins=bins,
                             stacked=True,
                             histtype='stepfilled')

    assert len(patches) == 2

    polygon,  = patches[0]
    xy = [[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 1], [3, 1],
          [3, 0], [2, 0], [2, 0], [1, 0], [1, 0], [0, 0]]
    assert_array_equal(polygon.get_xy(), xy)

    polygon,  = patches[1]
    xy = [[0, 2], [0, 3], [1, 3], [1, 4], [2, 4], [2, 2], [3, 2],
          [3, 1], [2, 1], [2, 3], [1, 3], [1, 2], [0, 2]]
    assert_array_equal(polygon.get_xy(), xy)


def test_hist_stacked_step_geometry():
    bins = [0, 1, 2, 3]
    data_1 = [0, 0, 1, 1, 1, 2]
    data_2 = [0, 1, 2]
    _, _, patches = plt.hist([data_1, data_2],
                             bins=bins,
                             stacked=True,
                             histtype='step')

    assert len(patches) == 2

    polygon,  = patches[0]
    xy = [[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 1], [3, 1], [3, 0]]
    assert_array_equal(polygon.get_xy(), xy)

    polygon,  = patches[1]
    xy = [[0, 2], [0, 3], [1, 3], [1, 4], [2, 4], [2, 2], [3, 2], [3, 1]]
    assert_array_equal(polygon.get_xy(), xy)


def test_hist_stacked_stepfilled_bottom_geometry():
    bins = [0, 1, 2, 3]
    data_1 = [0, 0, 1, 1, 1, 2]
    data_2 = [0, 1, 2]
    _, _, patches = plt.hist([data_1, data_2],
                             bins=bins,
                             stacked=True,
                             bottom=[1, 2, 1.5],
                             histtype='stepfilled')

    assert len(patches) == 2

    polygon,  = patches[0]
    xy = [[0, 1], [0, 3], [1, 3], [1, 5], [2, 5], [2, 2.5], [3, 2.5],
          [3, 1.5], [2, 1.5], [2, 2], [1, 2], [1, 1], [0, 1]]
    assert_array_equal(polygon.get_xy(), xy)

    polygon,  = patches[1]
    xy = [[0, 3], [0, 4], [1, 4], [1, 6], [2, 6], [2, 3.5], [3, 3.5],
          [3, 2.5], [2, 2.5], [2, 5], [1, 5], [1, 3], [0, 3]]
    assert_array_equal(polygon.get_xy(), xy)


def test_hist_stacked_step_bottom_geometry():
    bins = [0, 1, 2, 3]
    data_1 = [0, 0, 1, 1, 1, 2]
    data_2 = [0, 1, 2]
    _, _, patches = plt.hist([data_1, data_2],
                             bins=bins,
                             stacked=True,
                             bottom=[1, 2, 1.5],
                             histtype='step')

    assert len(patches) == 2

    polygon,  = patches[0]
    xy = [[0, 1], [0, 3], [1, 3], [1, 5], [2, 5], [2, 2.5], [3, 2.5], [3, 1.5]]
    assert_array_equal(polygon.get_xy(), xy)

    polygon,  = patches[1]
    xy = [[0, 3], [0, 4], [1, 4], [1, 6], [2, 6], [2, 3.5], [3, 3.5], [3, 2.5]]
    assert_array_equal(polygon.get_xy(), xy)


@image_comparison(['hist_stacked_bar'])
def test_hist_stacked_bar():
    # make some data
    d = [[100, 100, 100, 100, 200, 320, 450, 80, 20, 600, 310, 800],
         [20, 23, 50, 11, 100, 420], [120, 120, 120, 140, 140, 150, 180],
         [60, 60, 60, 60, 300, 300, 5, 5, 5, 5, 10, 300],
         [555, 555, 555, 30, 30, 30, 30, 30, 100, 100, 100, 100, 30, 30],
         [30, 30, 30, 30, 400, 400, 400, 400, 400, 400, 400, 400]]
    colors = [(0.5759849696758961, 1.0, 0.0), (0.0, 1.0, 0.350624650815206),
              (0.0, 1.0, 0.6549834156005998), (0.0, 0.6569064625276622, 1.0),
              (0.28302699607823545, 0.0, 1.0), (0.6849123462299822, 0.0, 1.0)]
    labels = ['green', 'orange', ' yellow', 'magenta', 'black']
    fig, ax = plt.subplots()
    ax.hist(d, bins=10, histtype='barstacked', align='mid', color=colors,
            label=labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1)


def test_hist_barstacked_bottom_unchanged():
    b = np.array([10, 20])
    plt.hist([[0, 1], [0, 1]], 2, histtype="barstacked", bottom=b)
    assert b.tolist() == [10, 20]


def test_hist_emptydata():
    fig, ax = plt.subplots()
    ax.hist([[], range(10), range(10)], histtype="step")


def test_hist_labels():
    # test singleton labels OK
    fig, ax = plt.subplots()
    _, _, bars = ax.hist([0, 1], label=0)
    assert bars[0].get_label() == '0'
    _, _, bars = ax.hist([0, 1], label=[0])
    assert bars[0].get_label() == '0'
    _, _, bars = ax.hist([0, 1], label=None)
    assert bars[0].get_label() == '_nolegend_'
    _, _, bars = ax.hist([0, 1], label='0')
    assert bars[0].get_label() == '0'
    _, _, bars = ax.hist([0, 1], label='00')
    assert bars[0].get_label() == '00'


@image_comparison(['transparent_markers'], remove_text=True)
def test_transparent_markers():
    np.random.seed(0)
    data = np.random.random(50)

    fig, ax = plt.subplots()
    ax.plot(data, 'D', mfc='none', markersize=100)


@image_comparison(['rgba_markers'], remove_text=True)
def test_rgba_markers():
    fig, axs = plt.subplots(ncols=2)
    rcolors = [(1, 0, 0, 1), (1, 0, 0, 0.5)]
    bcolors = [(0, 0, 1, 1), (0, 0, 1, 0.5)]
    alphas = [None, 0.2]
    kw = dict(ms=100, mew=20)
    for i, alpha in enumerate(alphas):
        for j, rcolor in enumerate(rcolors):
            for k, bcolor in enumerate(bcolors):
                axs[i].plot(j+1, k+1, 'o', mfc=bcolor, mec=rcolor,
                            alpha=alpha, **kw)
                axs[i].plot(j+1, k+3, 'x', mec=rcolor, alpha=alpha, **kw)
    for ax in axs:
        ax.axis([-1, 4, 0, 5])


@image_comparison(['mollweide_grid'], remove_text=True)
def test_mollweide_grid():
    # test that both horizontal and vertical gridlines appear on the Mollweide
    # projection
    fig = plt.figure()
    ax = fig.add_subplot(projection='mollweide')
    ax.grid()


def test_mollweide_forward_inverse_closure():
    # test that the round-trip Mollweide forward->inverse transformation is an
    # approximate identity
    fig = plt.figure()
    ax = fig.add_subplot(projection='mollweide')

    # set up 1-degree grid in longitude, latitude
    lon = np.linspace(-np.pi, np.pi, 360)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, 180)
    lon, lat = np.meshgrid(lon, lat)
    ll = np.vstack((lon.flatten(), lat.flatten())).T

    # perform forward transform
    xy = ax.transProjection.transform(ll)

    # perform inverse transform
    ll2 = ax.transProjection.inverted().transform(xy)

    # compare
    np.testing.assert_array_almost_equal(ll, ll2, 3)


def test_mollweide_inverse_forward_closure():
    # test that the round-trip Mollweide inverse->forward transformation is an
    # approximate identity
    fig = plt.figure()
    ax = fig.add_subplot(projection='mollweide')

    # set up grid in x, y
    x = np.linspace(0, 1, 500)
    x, y = np.meshgrid(x, x)
    xy = np.vstack((x.flatten(), y.flatten())).T

    # perform inverse transform
    ll = ax.transProjection.inverted().transform(xy)

    # perform forward transform
    xy2 = ax.transProjection.transform(ll)

    # compare
    np.testing.assert_array_almost_equal(xy, xy2, 3)


@image_comparison(['test_alpha'], remove_text=True)
def test_alpha():
    np.random.seed(0)
    data = np.random.random(50)

    fig, ax = plt.subplots()

    # alpha=.5 markers, solid line
    ax.plot(data, '-D', color=[1, 0, 0], mfc=[1, 0, 0, .5],
            markersize=20, lw=10)

    # everything solid by kwarg
    ax.plot(data + 2, '-D', color=[1, 0, 0, .5], mfc=[1, 0, 0, .5],
            markersize=20, lw=10,
            alpha=1)

    # everything alpha=.5 by kwarg
    ax.plot(data + 4, '-D', color=[1, 0, 0], mfc=[1, 0, 0],
            markersize=20, lw=10,
            alpha=.5)

    # everything alpha=.5 by colors
    ax.plot(data + 6, '-D', color=[1, 0, 0, .5], mfc=[1, 0, 0, .5],
            markersize=20, lw=10)

    # alpha=.5 line, solid markers
    ax.plot(data + 8, '-D', color=[1, 0, 0, .5], mfc=[1, 0, 0],
            markersize=20, lw=10)


@image_comparison(['eventplot', 'eventplot'], remove_text=True)
def test_eventplot():
    np.random.seed(0)

    data1 = np.random.random([32, 20]).tolist()
    data2 = np.random.random([6, 20]).tolist()
    data = data1 + data2
    num_datasets = len(data)

    colors1 = [[0, 1, .7]] * len(data1)
    colors2 = [[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [1, .75, 0],
               [1, 0, 1],
               [0, 1, 1]]
    colors = colors1 + colors2

    lineoffsets1 = 12 + np.arange(0, len(data1)) * .33
    lineoffsets2 = [-15, -3, 1, 1.5, 6, 10]
    lineoffsets = lineoffsets1.tolist() + lineoffsets2

    linelengths1 = [.33] * len(data1)
    linelengths2 = [5, 2, 1, 1, 3, 1.5]
    linelengths = linelengths1 + linelengths2

    fig = plt.figure()
    axobj = fig.add_subplot()
    colls = axobj.eventplot(data, colors=colors, lineoffsets=lineoffsets,
                            linelengths=linelengths)

    num_collections = len(colls)
    assert num_collections == num_datasets

    # Reuse testcase from above for a labeled data test
    data = {"pos": data, "c": colors, "lo": lineoffsets, "ll": linelengths}
    fig = plt.figure()
    axobj = fig.add_subplot()
    colls = axobj.eventplot("pos", colors="c", lineoffsets="lo",
                            linelengths="ll", data=data)
    num_collections = len(colls)
    assert num_collections == num_datasets


@image_comparison(['test_eventplot_defaults.png'], remove_text=True)
def test_eventplot_defaults():
    """
    test that eventplot produces the correct output given the default params
    (see bug #3728)
    """
    np.random.seed(0)

    data1 = np.random.random([32, 20]).tolist()
    data2 = np.random.random([6, 20]).tolist()
    data = data1 + data2

    fig = plt.figure()
    axobj = fig.add_subplot()
    axobj.eventplot(data)


@pytest.mark.parametrize(('colors'), [
    ('0.5',),  # string color with multiple characters: not OK before #8193 fix
    ('tab:orange', 'tab:pink', 'tab:cyan', 'bLacK'),  # case-insensitive
    ('red', (0, 1, 0), None, (1, 0, 1, 0.5)),  # a tricky case mixing types
])
def test_eventplot_colors(colors):
    """Test the *colors* parameter of eventplot. Inspired by issue #8193."""
    data = [[0], [1], [2], [3]]  # 4 successive events of different nature

    # Build the list of the expected colors
    expected = [c if c is not None else 'C0' for c in colors]
    # Convert the list into an array of RGBA values
    # NB: ['rgbk'] is not a valid argument for to_rgba_array, while 'rgbk' is.
    if len(expected) == 1:
        expected = expected[0]
    expected = np.broadcast_to(mcolors.to_rgba_array(expected), (len(data), 4))

    fig, ax = plt.subplots()
    if len(colors) == 1:  # tuple with a single string (like '0.5' or 'rgbk')
        colors = colors[0]
    collections = ax.eventplot(data, colors=colors)

    for coll, color in zip(collections, expected):
        assert_allclose(coll.get_color(), color)


@image_comparison(['test_eventplot_problem_kwargs.png'], remove_text=True)
def test_eventplot_problem_kwargs(recwarn):
    """
    test that 'singular' versions of LineCollection props raise an
    MatplotlibDeprecationWarning rather than overriding the 'plural' versions
    (e.g., to prevent 'color' from overriding 'colors', see issue #4297)
    """
    np.random.seed(0)

    data1 = np.random.random([20]).tolist()
    data2 = np.random.random([10]).tolist()
    data = [data1, data2]

    fig = plt.figure()
    axobj = fig.add_subplot()

    axobj.eventplot(data,
                    colors=['r', 'b'],
                    color=['c', 'm'],
                    linewidths=[2, 1],
                    linewidth=[1, 2],
                    linestyles=['solid', 'dashed'],
                    linestyle=['dashdot', 'dotted'])

    assert len(recwarn) == 3
    assert all(issubclass(wi.category, MatplotlibDeprecationWarning)
               for wi in recwarn)


def test_empty_eventplot():
    fig, ax = plt.subplots(1, 1)
    ax.eventplot([[]], colors=[(0.0, 0.0, 0.0, 0.0)])
    plt.draw()


@pytest.mark.parametrize('data', [[[]], [[], [0, 1]], [[0, 1], []]])
@pytest.mark.parametrize('orientation', [None, 'vertical', 'horizontal'])
def test_eventplot_orientation(data, orientation):
    """Introduced when fixing issue #6412."""
    opts = {} if orientation is None else {'orientation': orientation}
    fig, ax = plt.subplots(1, 1)
    ax.eventplot(data, **opts)
    plt.draw()


@image_comparison(['marker_styles.png'], remove_text=True)
def test_marker_styles():
    fig, ax = plt.subplots()
    # Since generation of the test image, None was removed but 'none' was
    # added. By moving 'none' to the front (=former sorted place of None)
    # we can avoid regenerating the test image. This can be removed if the
    # test image has to be regenerated for other reasons.
    markers = sorted(matplotlib.markers.MarkerStyle.markers,
                     key=lambda x: str(type(x))+str(x))
    markers.remove('none')
    markers = ['none', *markers]
    for y, marker in enumerate(markers):
        ax.plot((y % 2)*5 + np.arange(10)*10, np.ones(10)*10*y, linestyle='',
                marker=marker, markersize=10+y/5, label=marker)


@image_comparison(['rc_markerfill.png'])
def test_markers_fillstyle_rcparams():
    fig, ax = plt.subplots()
    x = np.arange(7)
    for idx, (style, marker) in enumerate(
            [('top', 's'), ('bottom', 'o'), ('none', '^')]):
        matplotlib.rcParams['markers.fillstyle'] = style
        ax.plot(x+idx, marker=marker)


@image_comparison(['vertex_markers.png'], remove_text=True)
def test_vertex_markers():
    data = list(range(10))
    marker_as_tuple = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    marker_as_list = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    fig, ax = plt.subplots()
    ax.plot(data, linestyle='', marker=marker_as_tuple, mfc='k')
    ax.plot(data[::-1], linestyle='', marker=marker_as_list, mfc='b')
    ax.set_xlim([-1, 10])
    ax.set_ylim([-1, 10])


@image_comparison(['vline_hline_zorder', 'errorbar_zorder'],
                  tol=0 if platform.machine() == 'x86_64' else 0.02)
def test_eb_line_zorder():
    x = list(range(10))

    # First illustrate basic pyplot interface, using defaults where possible.
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, lw=10, zorder=5)
    ax.axhline(1, color='red', lw=10, zorder=1)
    ax.axhline(5, color='green', lw=10, zorder=10)
    ax.axvline(7, color='m', lw=10, zorder=7)
    ax.axvline(2, color='k', lw=10, zorder=3)

    ax.set_title("axvline and axhline zorder test")

    # Now switch to a more OO interface to exercise more features.
    fig = plt.figure()
    ax = fig.gca()
    x = list(range(10))
    y = np.zeros(10)
    yerr = list(range(10))
    ax.errorbar(x, y, yerr=yerr, zorder=5, lw=5, color='r')
    for j in range(10):
        ax.axhline(j, lw=5, color='k', zorder=j)
        ax.axhline(-j, lw=5, color='k', zorder=j)
 'c' acceptable as PathCollection facecolors?
                colors = mcolors.to_rgba_array(c)
            except (TypeError, ValueError) as err:
                if "RGBA values should be within 0-1 range" in str(err):
                    raise
                else:
                    if not valid_shape:
                        raise invalid_shape_exception(c.size, xsize) from err
                    # Both the mapping *and* the RGBA conversion failed: pretty
                    # severe failure => one may appreciate a verbose feedback.
                    raise ValueError(
                        f"'c' argument must be a color, a sequence of colors, "
                        f"or a sequence of numbers, not {c}") from err
            else:
                if len(colors) not in (0, 1, xsize):
                    # NB: remember that a single color is also acceptable.
                    # Besides *colors* will be an empty array if c == 'none'.
                    raise invalid_shape_exception(len(colors), xsize)
        else:
            colors = None  # use cmap, norm after collection is created
        return c, colors, edgecolors

    @_preprocess_data(replace_names=["x", "y", "s", "linewidths",
                                     "edgecolors", "c", "facecolor",
                                     "facecolors", "color"],
                      label_namer="y")
    def scatter(self, x, y, s=None, c=None, marker=None, cmap=None, norm=None,
                vmin=None, vmax=None, alpha=None, linewidths=None, *,
                edgecolors=None, plotnonfinite=False, **kwargs):
        """
        A scatter plot of *y* vs. *x* with varying marker size and/or color.

        Parameters
        ----------
        x, y : float or array-like, shape (n, )
            The data positions.

        s : float or array-like, shape (n, ), optional
            The marker size in points**2.
            Default is ``rcParams['lines.markersize'] ** 2``.

        c : array-like or list of colors or color, optional
            The marker colors. Possible values:

            - A scalar or sequence of n numbers to be mapped to colors using
              *cmap* and *norm*.
            - A 2D array in which the rows are RGB or RGBA.
            - A sequence of colors of length n.
            - A single color format string.

            Note that *c* should not be a single numeric RGB or RGBA sequence
            because that is indistinguishable from an array of values to be
            colormapped. If you want to specify the same RGB or RGBA value for
            all points, use a 2D array with a single row.  Otherwise, value-
            matching will have precedence in case of a size matching with *x*
            and *y*.

            If you wish to specify a single color for all points
            prefer the *color* keyword argument.

            Defaults to `None`. In that case the marker color is determined
            by the value of *color*, *facecolor* or *facecolors*. In case
            those are not specified or `None`, the marker color is determined
            by the next color of the ``Axes``' current "shape and fill" color
            cycle. This cycle defaults to :rc:`axes.prop_cycle`.

        marker : `~.markers.MarkerStyle`, default: :rc:`scatter.marker`
            The marker style. *marker* can be either an instance of the class
            or the text shorthand for a particular marker.
            See :mod:`matplotlib.markers` for more information about marker
            styles.

        cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            A `.Colormap` instance or registered colormap name. *cmap* is only
            used if *c* is an array of floats.

        norm : `~matplotlib.colors.Normalize`, default: None
            If *c* is an array of floats, *norm* is used to scale the color
            data, *c*, in the range 0 to 1, in order to map into the colormap
            *cmap*.
            If *None*, use the default `.colors.Normalize`.

        vmin, vmax : float, default: None
            *vmin* and *vmax* are used in conjunction with the default norm to
            map the color array *c* to the colormap *cmap*. If None, the
            respective min and max of the color array is used.
            It is an error to use *vmin*/*vmax* when *norm* is given.

        alpha : float, default: None
            The alpha blending value, between 0 (transparent) and 1 (opaque).

        linewidths : float or array-like, default: :rc:`lines.linewidth`
            The linewidth of the marker edges. Note: The default *edgecolors*
            is 'face'. You may want to change this as well.

        edgecolors : {'face', 'none', *None*} or color or sequence of color, \
default: :rc:`scatter.edgecolors`
            The edge color of the marker. Possible values:

            - 'face': The edge color will always be the same as the face color.
            - 'none': No patch boundary will be drawn.
            - A color or sequence of colors.

            For non-filled markers, *edgecolors* is ignored. Instead, the color
            is determined like with 'face', i.e. from *c*, *colors*, or
            *facecolors*.

        plotnonfinite : bool, default: False
            Whether to plot points with nonfinite *c* (i.e. ``inf``, ``-inf``
            or ``nan``). If ``True`` the points are drawn with the *bad*
            colormap color (see `.Colormap.set_bad`).

        Returns
        -------
        `~matplotlib.collections.PathCollection`

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs : `~matplotlib.collections.Collection` properties

        See Also
        --------
        plot : To plot scatter plots when markers are identical in size and
            color.

        Notes
        -----
        * The `.plot` function will be faster for scatterplots where markers
          don't vary in size or color.

        * Any or all of *x*, *y*, *s*, and *c* may be masked arrays, in which
          case all masks will be combined and only unmasked points will be
          plotted.

        * Fundamentally, scatter works with 1D arrays; *x*, *y*, *s*, and *c*
          may be input as N-D arrays, but within scatter they will be
          flattened. The exception is *c*, which will be flattened only if its
          size matches the size of *x* and *y*.

        """
        # Process **kwargs to handle aliases, conflicts with explicit kwargs:

        x, y = self._process_unit_info([("x", x), ("y", y)], kwargs)

        # np.ma.ravel yields an ndarray, not a masked array,
        # unless its argument is a masked array.
        x = np.ma.ravel(x)
        y = np.ma.ravel(y)
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        if s is None:
            s = (20 if rcParams['_internal.classic_mode'] else
                 rcParams['lines.markersize'] ** 2.0)
        s = np.ma.ravel(s)
        if (len(s) not in (1, x.size) or
                (not np.issubdtype(s.dtype, np.floating) and
                 not np.issubdtype(s.dtype, np.integer))):
            raise ValueError(
                "s must be a scalar, "
                "or float array-like with the same size as x and y")

        # get the original edgecolor the user passed before we normalize
        orig_edgecolor = edgecolors
        if edgecolors is None:
            orig_edgecolor = kwargs.get('edgecolor', None)
        c, colors, edgecolors = \
            self._parse_scatter_color_args(
                c, edgecolors, kwargs, x.size,
                get_next_color_func=self._get_patches_for_fill.get_next_color)

        if plotnonfinite and colors is None:
            c = np.ma.masked_invalid(c)
            x, y, s, edgecolors, linewidths = \
                cbook._combine_masks(x, y, s, edgecolors, linewidths)
        else:
            x, y, s, c, colors, edgecolors, linewidths = \
                cbook._combine_masks(
                    x, y, s, c, colors, edgecolors, linewidths)
        # Unmask edgecolors if it was actually a single RGB or RGBA.
        if (x.size in (3, 4)
                and np.ma.is_masked(edgecolors)
                and not np.ma.is_masked(orig_edgecolor)):
            edgecolors = edgecolors.data

        scales = s   # Renamed for readability below.

        # load default marker from rcParams
        if marker is None:
            marker = rcParams['scatter.marker']

        if isinstance(marker, mmarkers.MarkerStyle):
            marker_obj = marker
        else:
            marker_obj = mmarkers.MarkerStyle(marker)

        path = marker_obj.get_path().transformed(
            marker_obj.get_transform())
        if not marker_obj.is_filled():
            if orig_edgecolor is not None:
                _api.warn_external(
                    f"You passed a edgecolor/edgecolors ({orig_edgecolor!r}) "
                    f"for an unfilled marker ({marker!r}).  Matplotlib is "
                    "ignoring the edgecolor in favor of the facecolor.  This "
                    "behavior may change in the future."
                )
            # We need to handle markers that can not be filled (like
            # '+' and 'x') differently than markers that can be
            # filled, but have their fillstyle set to 'none'.  This is
            # to get:
            #
            #  - respecting the fillestyle if set
            #  - maintaining back-compatibility for querying the facecolor of
            #    the un-fillable markers.
            #
            # While not an ideal situation, but is better than the
            # alternatives.
            if marker_obj.get_fillstyle() == 'none':
                # promote the facecolor to be the edgecolor
                edgecolors = colors
                # set the facecolor to 'none' (at the last chance) because
                # we can not fill a path if the facecolor is non-null
                # (which is defendable at the renderer level).
                colors = 'none'
            else:
                # if we are not nulling the face color we can do this
                # simpler
                edgecolors = 'face'

            if linewidths is None:
                linewidths = rcParams['lines.linewidth']
            elif np.iterable(linewidths):
                linewidths = [
                    lw if lw is not None else rcParams['lines.linewidth']
                    for lw in linewidths]

        offsets = np.ma.column_stack([x, y])

        collection = mcoll.PathCollection(
                (path,), scales,
                facecolors=colors,
                edgecolors=edgecolors,
                linewidths=linewidths,
                offsets=offsets,
                transOffset=kwargs.pop('transform', self.transData),
                alpha=alpha
                )
        collection.set_transform(mtransforms.IdentityTransform())
        collection.update(kwargs)

        if colors is None:
            collection.set_array(c)
            collection.set_cmap(cmap)
            collection.set_norm(norm)
            collection._scale_norm(norm, vmin, vmax)

        # Classic mode only:
        # ensure there are margins to allow for the
        # finite size of the symbols.  In v2.x, margins
        # are present by default, so we disable this
        # scatter-specific override.
        if rcParams['_internal.classic_mode']:
            if self._xmargin < 0.05 and x.size > 0:
                self.set_xmargin(0.05)
            if self._ymargin < 0.05 and x.size > 0:
                self.set_ymargin(0.05)

        self.add_collection(collection)
        self._request_autoscale_view()

        return collection

    @_preprocess_data(replace_names=["x", "y", "C"], label_namer="y")
    @docstring.dedent_interpd
    def hexbin(self, x, y, C=None, gridsize=100, bins=None,
               xscale='linear', yscale='linear', extent=None,
               cmap=None, norm=None, vmin=None, vmax=None,
               alpha=None, linewidths=None, edgecolors='face',
               reduce_C_function=np.mean, mincnt=None, marginals=False,
               **kwargs):
        """
        Make a 2D hexagonal binning plot of points *x*, *y*.

        If *C* is *None*, the value of the hexagon is determined by the number
        of points in the hexagon. Otherwise, *C* specifies values at the
        coordinate (x[i], y[i]). For each hexagon, these values are reduced
        using *reduce_C_function*.

        Parameters
        ----------
        x, y : array-like
            The data positions. *x* and *y* must be of the same length.

        C : array-like, optional
            If given, these values are accumulated in the bins. Otherwise,
            every point has a value of 1. Must be of the same length as *x*
            and *y*.

        gridsize : int or (int, int), default: 100
            If a single int, the number of hexagons in the *x*-direction.
            The number of hexagons in the *y*-direction is chosen such that
            the hexagons are approximately regular.

            Alternatively, if a tuple (*nx*, *ny*), the number of hexagons
            in the *x*-direction and the *y*-direction.

        bins : 'log' or int or sequence, default: None
            Discretization of the hexagon values.

            - If *None*, no binning is applied; the color of each hexagon
              directly corresponds to its count value.
            - If 'log', use a logarithmic scale for the colormap.
              Internally, :math:`log_{10}(i+1)` is used to determine the
              hexagon color. This is equivalent to ``norm=LogNorm()``.
            - If an integer, divide the counts in the specified number
              of bins, and color the hexagons accordingly.
            - If a sequence of values, the values of the lower bound of
              the bins to be used.

        xscale : {'linear', 'log'}, default: 'linear'
            Use a linear or log10 scale on the horizontal axis.

        yscale : {'linear', 'log'}, default: 'linear'
            Use a linear or log10 scale on the vertical axis.

        mincnt : int > 0, default: *None*
            If not *None*, only display cells with more than *mincnt*
            number of points in the cell.

        marginals : bool, default: *False*
            If marginals is *True*, plot the marginal density as
            colormapped rectangles along the bottom of the x-axis and
            left of the y-axis.

        extent : 4-tuple of float, default: *None*
            The limits of the bins (xmin, xmax, ymin, ymax).
            The default assigns the limits based on
            *gridsize*, *x*, *y*, *xscale* and *yscale*.

            If *xscale* or *yscale* is set to 'log', the limits are
            expected to be the exponent for a power of 10. E.g. for
            x-limits of 1 and 50 in 'linear' scale and y-limits
            of 10 and 1000 in 'log' scale, enter (1, 50, 1, 3).

        Returns
        -------
        `~matplotlib.collections.PolyCollection`
            A `.PolyCollection` defining the hexagonal bins.

            - `.PolyCollection.get_offsets` contains a Mx2 array containing
              the x, y positions of the M hexagon centers.
            - `.PolyCollection.get_array` contains the values of the M
              hexagons.

            If *marginals* is *True*, horizontal
            bar and vertical bar (both PolyCollections) will be attached
            to the return collection as attributes *hbar* and *vbar*.

        Other Parameters
        ----------------
        cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            The Colormap instance or registered colormap name used to map
            the bin values to colors.

        norm : `~matplotlib.colors.Normalize`, optional
            The Normalize instance scales the bin values to the canonical
            colormap range [0, 1] for mapping to colors. By default, the data
            range is mapped to the colorbar range using linear scaling.

        vmin, vmax : float, default: None
            The colorbar range. If *None*, suitable min/max values are
            automatically chosen by the `.Normalize` instance (defaults to
            the respective min/max values of the bins in case of the default
            linear scaling).
            It is an error to use *vmin*/*vmax* when *norm* is given.

        alpha : float between 0 and 1, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).

        linewidths : float, default: *None*
            If *None*, defaults to 1.0.

        edgecolors : {'face', 'none', *None*} or color, default: 'face'
            The color of the hexagon edges. Possible values are:

            - 'face': Draw the edges in the same color as the fill color.
            - 'none': No edges are drawn. This can sometimes lead to unsightly
              unpainted pixels between the hexagons.
            - *None*: Draw outlines in the default color.
            - An explicit color.

        reduce_C_function : callable, default: `numpy.mean`
            The function to aggregate *C* within the bins. It is ignored if
            *C* is not given. This must have the signature::

                def reduce_C_function(C: array) -> float

            Commonly used functions are:

            - `numpy.mean`: average of the points
            - `numpy.sum`: integral of the point values
            - `numpy.amax`: value taken from the largest point

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs : `~matplotlib.collections.PolyCollection` properties
            All other keyword arguments are passed on to `.PolyCollection`:

            %(PolyCollection:kwdoc)s

        See Also
        --------
        hist2d : 2D histogram rectangular bins
        """
        self._process_unit_info([("x", x), ("y", y)], kwargs, convert=False)

        x, y, C = cbook.delete_masked_points(x, y, C)

        # Set the size of the hexagon grid
        if np.iterable(gridsize):
            nx, ny = gridsize
        else:
            nx = gridsize
            ny = int(nx / math.sqrt(3))
        # Count the number of data in each hexagon
        x = np.array(x, float)
        y = np.array(y, float)

        if marginals:
            xorig = x.copy()
            yorig = y.copy()

        if xscale == 'log':
            if np.any(x <= 0.0):
                raise ValueError("x contains non-positive values, so can not"
                                 " be log-scaled")
            x = np.log10(x)
        if yscale == 'log':
            if np.any(y <= 0.0):
                raise ValueError("y contains non-positive values, so can not"
                                 " be log-scaled")
            y = np.log10(y)
        if extent is not None:
            xmin, xmax, ymin, ymax = extent
        else:
            xmin, xmax = (np.min(x), np.max(x)) if len(x) else (0, 1)
            ymin, ymax = (np.min(y), np.max(y)) if len(y) else (0, 1)

            # to avoid issues with singular data, expand the min/max pairs
            xmin, xmax = mtransforms.nonsingular(xmin, xmax, expander=0.1)
            ymin, ymax = mtransforms.nonsingular(ymin, ymax, expander=0.1)

        # In the x-direction, the hexagons exactly cover the region from
        # xmin to xmax. Need some padding to avoid roundoff errors.
        padding = 1.e-9 * (xmax - xmin)
        xmin -= padding
        xmax += padding
        sx = (xmax - xmin) / nx
        sy = (ymax - ymin) / ny

        x = (x - xmin) / sx
        y = (y - ymin) / sy
        ix1 = np.round(x).astype(int)
        iy1 = np.round(y).astype(int)
        ix2 = np.floor(x).astype(int)
        iy2 = np.floor(y).astype(int)

        nx1 = nx + 1
        ny1 = ny + 1
        nx2 = nx
        ny2 = ny
        n = nx1 * ny1 + nx2 * ny2

        d1 = (x - ix1) ** 2 + 3.0 * (y - iy1) ** 2
        d2 = (x - ix2 - 0.5) ** 2 + 3.0 * (y - iy2 - 0.5) ** 2
        bdist = (d1 < d2)
        if C is None:
            lattice1 = np.zeros((nx1, ny1))
            lattice2 = np.zeros((nx2, ny2))
            c1 = (0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1) & bdist
            c2 = (0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2) & ~bdist
            np.add.at(lattice1, (ix1[c1], iy1[c1]), 1)
            np.add.at(lattice2, (ix2[c2], iy2[c2]), 1)
            if mincnt is not None:
                lattice1[lattice1 < mincnt] = np.nan
                lattice2[lattice2 < mincnt] = np.nan
            accum = np.concatenate([lattice1.ravel(), lattice2.ravel()])
            good_idxs = ~np.isnan(accum)

        else:
            if mincnt is None:
                mincnt = 0

            # create accumulation arrays
            lattice1 = np.empty((nx1, ny1), dtype=object)
            for i in range(nx1):
                for j in range(ny1):
                    lattice1[i, j] = []
            lattice2 = np.empty((nx2, ny2), dtype=object)
            for i in range(nx2):
                for j in range(ny2):
                    lattice2[i, j] = []

            for i in range(len(x)):
                if bdist[i]:
                    if 0 <= ix1[i] < nx1 and 0 <= iy1[i] < ny1:
                        lattice1[ix1[i], iy1[i]].append(C[i])
                else:
                    if 0 <= ix2[i] < nx2 and 0 <= iy2[i] < ny2:
                        lattice2[ix2[i], iy2[i]].append(C[i])

            for i in range(nx1):
                for j in range(ny1):
                    vals = lattice1[i, j]
                    if len(vals
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/0d89bf6cf4f2a8f4999b3998b54c6ed8caaca8c5/lib/matplotlib/axes/_axes.py/left.py


@check_figures_equal(extensions=['png'])
def test_twin_remove(fig_test, fig_ref):
    ax_test = fig_test.add_subplot()
    ax_twinx = ax_test.twinx()
    ax_twiny = ax_test.twiny()
    ax_twinx.remove()
    ax_twiny.remove()

    ax_ref = fig_ref.add_subplot()
    # Ideally we also undo tick changes when calling ``remove()``, but for now
    # manually set the ticks of the reference image to match the test image
    ax_ref.xaxis.tick_bottom()
    ax_ref.yaxis.tick_left()


@image_comparison(['twin_spines.png'], remove_text=True

=======

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/0d89bf6cf4f2a8f4999b3998b54c6ed8caaca8c5/lib/matplotlib/axes/_axes.py/right.py
) > mincnt:
                        lattice1[i, j] = reduce_C_function(vals)
                    else:
                        lattice1[i, j] = np.nan
            for i in range(nx2):
                for j in range(ny2):
                    vals = lattice2[i, j]
                    if len(vals) > mincnt:
                        lattice2[i, j] = reduce_C_function(vals)
                    else:
                        lattice2[i, j] = np.nan

            accum = np.concatenate([lattice1.astype(float).ravel(),
                                    lattice2.astype(float).ravel()])
            good_idxs = ~np.isnan(accum)

        offsets = np.zeros((n, 2), float)
        offsets[:nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
        offsets[:nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
        offsets[nx1 * ny1:, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
        offsets[nx1 * ny1:, 1] = np.tile(np.arange(ny2), nx2) + 0.5
        offsets[:, 0] *= sx
        offsets[:, 1] *= sy
        offsets[:, 0] += xmin
        offsets[:, 1] += ymin
        # remove accumulation bins with no data
        offsets = offsets[good_idxs, :]
        accum = accum[good_idxs]

        polygon = [sx, sy / 3] * np.array(
            [[.5, -.5], [.5, .5], [0., 1.], [-.5, .5], [-.5, -.5], [0., -1.]])

        if linewidths is None:
            linewidths = [1.0]

        if xscale == 'log' or yscale == 'log':
            polygons = np.expand_dims(polygon, 0) + np.expand_dims(offsets, 1)
            if xscale == 'log':
                polygons[:, :, 0] = 10.0 ** polygons[:, :, 0]
                xmin = 10.0 ** xmin
                xmax = 10.0 ** xmax
                self.set_xscale(xscale)
            if yscale == 'log':
                polygons[:, :, 1] = 10.0 ** polygons[:, :, 1]
                ymin = 10.0 ** ymin
                ymax = 10.0 ** ymax
                self.set_yscale(yscale)
            collection = mcoll.PolyCollection(
                polygons,
                edgecolors=edgecolors,
                linewidths=linewidths,
                )
        else:
            collection = mcoll.PolyCollection(
                [polygon],
                edgecolors=edgecolors,
                linewidths=linewidths,
                offsets=offsets,
                transOffset=mtransforms.AffineDeltaTransform(self.transData),
                )

        # Set normalizer if bins is 'log'
        if bins == 'log':
            if norm is not None:
                _api.warn_external("Only one of 'bins' and 'norm' arguments "
                                   f"can be supplied, ignoring bins={bins}")
            else:
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                vmin = vmax = None
            bins = None

        # autoscale the norm with current accum values if it hasn't
        # been set
        if norm is not None:
            if norm.vmin is None and norm.vmax is None:
                norm.autoscale(accum)

        if bins is not None:
            if not np.iterable(bins):
                minimum, maximum = min(accum), max(accum)
                bins -= 1  # one less edge than bins
                bins = minimum + (maximum - minimum) * np.arange(bins) / bins
            bins = np.sort(bins)
            accum = bins.searchsorted(accum)

        collection.set_array(accum)
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        collection.set_alpha(alpha)
        collection.update(kwargs)
        collection._scale_norm(norm, vmin, vmax)

        corners = ((xmin, ymin), (xmax, ymax))
        self.update_datalim(corners)
        self._request_autoscale_view(tight=True)

        # add the collection last
        self.add_collection(collection, autolim=False)
        if not marginals:
            return collection

        # Process marginals
        if C is None:
            C = np.ones(len(x))

        def coarse_bin(x, y, bin_edges):
            """
            Sort x-values into bins defined by *bin_edges*, then for all the
            corresponding y-values in each bin use *reduce_c_function* to
            compute the bin value.
            """
            nbins = len(bin_edges) - 1
            # Sort x-values into bins
            bin_idxs = np.searchsorted(bin_edges, x) - 1
            mus = np.zeros(nbins) * np.nan
            for i in range(nbins):
                # Get y-values for each bin
                yi = y[bin_idxs == i]
                if len(yi) > 0:
                    mus[i] = reduce_C_function(yi)
            return mus

        if xscale == 'log':
            bin_edges = np.geomspace(xmin, xmax, nx + 1)
        else:
            bin_edges = np.linspace(xmin, xmax, nx + 1)
        xcoarse = coarse_bin(xorig, C, bin_edges)

        verts, values = [], []
        for bin_left, bin_right, val in zip(
                bin_edges[:-1], bin_edges[1:], xcoarse):
            if np.isnan(val):
                continue
            verts.append([(bin_left, 0),
                          (bin_left, 0.05),
                          (bin_right, 0.05),
                          (bin_right, 0)])
            values.append(val)

        values = np.array(values)
        trans = self.get_xaxis_transform(which='grid')

        hbar = mcoll.PolyCollection(verts, transform=trans, edgecolors='face')

        hbar.set_array(values)
        hbar.set_cmap(cmap)
        hbar.set_norm(norm)
        hbar.set_alpha(alpha)
        hbar.update(kwargs)
        self.add_collection(hbar, autolim=False)

        if yscale == 'log':
            bin_edges = np.geomspace(ymin, ymax, 2 * ny + 1)
        else:
            bin_edges = np.linspace(ymin, ymax, 2 * ny + 1)
        ycoarse = coarse_bin(yorig, C, bin_edges)

        verts, values = [], []
        for bin_bottom, bin_top, val in zip(
                bin_edges[:-1], bin_edges[1:], ycoarse):
            if np.isnan(val):
                continue
            verts.append([(0, bin_bottom),
                          (0, bin_top),
                          (0.05, bin_top),
                          (0.05, bin_bottom)])
            values.append(val)

        values = np.array(values)

        trans = self.get_yaxis_transform(which='grid')

        vbar = mcoll.PolyCollection(verts, transform=trans, edgecolors='face')
        vbar.set_array(values)
        vbar.set_cmap(cmap)
        vbar.set_norm(norm)
        vbar.set_alpha(alpha)
        vbar.update(kwargs)
        self.add_collection(vbar, autolim=False)

        collection.hbar = hbar
        collection.vbar = vbar

        def on_changed(collection):
            hbar.set_cmap(collection.get_cmap())
            hbar.set_clim(collection.get_clim())
            vbar.set_cmap(collection.get_cmap())
            vbar.set_clim(collection.get_clim())

        collection.callbacks.connect('changed', on_changed)

        return collection

    @docstring.dedent_interpd
    def arrow(self, x, y, dx, dy, **kwargs):
        """
        Add an arrow to the Axes.

        This draws an arrow from ``(x, y)`` to ``(x+dx, y+dy)``.

        Parameters
        ----------
        %(FancyArrow)s

        Returns
        -------
        `.FancyArrow`
            The created `.FancyArrow` object.

        Notes
        -----
        The resulting arrow is affected by the Axes aspect ratio and limits.
        This may produce an arrow whose head is not square with its stem. To
        create an arrow whose head is square with its stem,
        use :meth:`annotate` for example:

        >>> ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
        ...             arrowprops=dict(arrowstyle="->"))

        """
        # Strip away units for the underlying patch since units
        # do not make sense to most patch-like code
        x = self.convert_xunits(x)
        y = self.convert_yunits(y)
        dx = self.convert_xunits(dx)
        dy = self.convert_yunits(dy)

        a = mpatches.FancyArrow(x, y, dx, dy, **kwargs)
        self.add_patch(a)
        self._request_autoscale_view()
        return a

    @docstring.copy(mquiver.QuiverKey.__init__)
    def quiverkey(self, Q, X, Y, U, label, **kw):
        qk = mquiver.QuiverKey(Q, X, Y, U, label, **kw)
        self.add_artist(qk)
        return qk

    # Handle units for x and y, if they've been passed
    def _quiver_units(self, args, kw):
        if len(args) > 3:
            x, y = args[0:2]
            x, y = self._process_unit_info([("x", x), ("y", y)], kw)
            return (x, y) + args[2:]
        return args

    # args can by a combination if X, Y, U, V, C and all should be replaced
    @_preprocess_data()
    def quiver(self, *args, **kw):
        # Make sure units are handled for x and y values
        args = self._quiver_units(args, kw)

        q = mquiver.Quiver(self, *args, **kw)

        self.add_collection(q, autolim=True)
        self._request_autoscale_view()
        return q
    quiver.__doc__ = mquiver.Quiver.quiver_doc

    # args can be some combination of X, Y, U, V, C and all should be replaced
    @_preprocess_data()
    @docstring.dedent_interpd
    def barbs(self, *args, **kw):
        """
        %(barbs_doc)s
        """
        # Make sure units are handled for x and y values
        args = self._quiver_units(args, kw)

        b = mquiver.Barbs(self, *args, **kw)
        self.add_collection(b, autolim=True)
        self._request_autoscale_view()
        return b

    # Uses a custom implementation of data-kwarg handling in
    # _process_plot_var_args.
    def fill(self, *args, data=None, **kwargs):
        """
        Plot filled polygons.

        Parameters
        ----------
        *args : sequence of x, y, [color]
            Each polygon is defined by the lists of *x* and *y* positions of
            its nodes, optionally followed by a *color* specifier. See
            :mod:`matplotlib.colors` for supported color specifiers. The
            standard color cycle is used for polygons without a color
            specifier.

            You can plot multiple polygons by providing multiple *x*, *y*,
            *[color]* groups.

            For example, each of the following is legal::

                ax.fill(x, y)                    # a polygon with default color
                ax.fill(x, y, "b")               # a blue polygon
                ax.fill(x, y, x2, y2)            # two polygons
                ax.fill(x, y, "b", x2, y2, "r")  # a blue and a red polygon

        data : indexable object, optional
            An object with labelled data. If given, provide the label names to
            plot in *x* and *y*, e.g.::

                ax.fill("time", "signal",
                        data={"time": [0, 1, 2], "signal": [0, 1, 0]})

        Returns
        -------
        list of `~matplotlib.patches.Polygon`

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Polygon` properties

        Notes
        -----
        Use :meth:`fill_between` if you would like to fill the region between
        two curves.
        """
        # For compatibility(!), get aliases from Line2D rather than Patch.
        kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
        # _get_patches_for_fill returns a generator, convert it to a list.
        patches = [*self._get_patches_for_fill(*args, data=data, **kwargs)]
        for poly in patches:
            self.add_patch(poly)
        self._request_autoscale_view()
        return patches

    def _fill_between_x_or_y(
            self, ind_dir, ind, dep1, dep2=0, *,
            where=None, interpolate=False, step=None, **kwargs):
        # Common implementation between fill_between (*ind_dir*="x") and
        # fill_betweenx (*ind_dir*="y").  *ind* is the independent variable,
        # *dep* the dependent variable.  The docstring below is interpolated
        # to generate both methods' docstrings.
        """
        Fill the area between two {dir} curves.

        The curves are defined by the points (*{ind}*, *{dep}1*) and (*{ind}*,
        *{dep}2*).  This creates one or multiple polygons describing the filled
        area.

        You may exclude some {dir} sections from filling using *where*.

        By default, the edges connect the given points directly.  Use *step*
        if the filling should be a step function, i.e. constant in between
        *{ind}*.

        Parameters
        ----------
        {ind} : array (length N)
            The {ind} coordinates of the nodes defining the curves.

        {dep}1 : array (length N) or scalar
            The {dep} coordinates of the nodes defining the first curve.

        {dep}2 : array (length N) or scalar, default: 0
            The {dep} coordinates of the nodes defining the second curve.

        where : array of bool (length N), optional
            Define *where* to exclude some {dir} regions from being filled.
            The filled regions are defined by the coordinates ``{ind}[where]``.
            More precisely, fill between ``{ind}[i]`` and ``{ind}[i+1]`` if
            ``where[i] and where[i+1]``.  Note that this definition implies
            that an isolated *True* value between two *False* values in *where*
            will not result in filling.  Both sides of the *True* position
            remain unfilled due to the adjacent *False* values.

        interpolate : bool, default: False
            This option is only relevant if *where* is used and the two curves
            are crossing each other.

            Semantically, *where* is often used for *{dep}1* > *{dep}2* or
            similar.  By default, the nodes of the polygon defining the filled
            region will only be placed at the positions in the *{ind}* array.
            Such a polygon cannot describe the above semantics close to the
            intersection.  The {ind}-sections containing the intersection are
            simply clipped.

            Setting *interpolate* to *True* will calculate the actual
            intersection point and extend the filled region up to this point.

        step : {{'pre', 'post', 'mid'}}, optional
            Define *step* if the filling should be a step function,
            i.e. constant in between *{ind}*.  The value determines where the
            step will occur:

            - 'pre': The y value is continued constantly to the left from
              every *x* position, i.e. the interval ``(x[i-1], x[i]]`` has the
              value ``y[i]``.
            - 'post': The y value is continued constantly to the right from
              every *x* position, i.e. the interval ``[x[i], x[i+1])`` has the
              value ``y[i]``.
            - 'mid': Steps occur half-way between the *x* positions.

        Returns
        -------
        `.PolyCollection`
            A `.PolyCollection` containing the plotted polygons.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            All other keyword arguments are passed on to `.PolyCollection`.
            They control the `.Polygon` properties:

            %(PolyCollection:kwdoc)s

        See Also
        --------
        fill_between : Fill between two sets of y-values.
        fill_betweenx : Fill between two sets of x-values.

        Notes
        -----
        .. [notes section required to get data note injection right]
        """

        dep_dir = {"x": "y", "y": "x"}[ind_dir]

        if not rcParams["_internal.classic_mode"]:
            kwargs = cbook.normalize_kwargs(kwargs, mcoll.Collection)
            if not any(c in kwargs for c in ("color", "facecolor")):
                kwargs["facecolor"] = \
                    self._get_patches_for_fill.get_next_color()

        # Handle united data, such as dates
        ind, dep1, dep2 = map(
            ma.masked_invalid, self._process_unit_info(
                [(ind_dir, ind), (dep_dir, dep1), (dep_dir, dep2)], kwargs))

        for name, array in [
                (ind_dir, ind), (f"{dep_dir}1", dep1), (f"{dep_dir}2", dep2)]:
            if array.ndim > 1:
                raise ValueError(f"{name!r} is not 1-dimensional")

        if where is None:
            where = True
        else:
            where = np.asarray(where, dtype=bool)
            if where.size != ind.size:
                raise ValueError(f"where size ({where.size}) does not match "
                                 f"{ind_dir} size ({ind.size})")
        where = where & ~functools.reduce(
            np.logical_or, map(np.ma.getmask, [ind, dep1, dep2]))

        ind, dep1, dep2 = np.broadcast_arrays(
            np.atleast_1d(ind), dep1, dep2, subok=True)

        polys = []
        for idx0, idx1 in cbook.contiguous_regions(where):
            indslice = ind[idx0:idx1]
            dep1slice = dep1[idx0:idx1]
            dep2slice = dep2[idx0:idx1]
            if step is not None:
                step_func = cbook.STEP_LOOKUP_MAP["steps-" + step]
                indslice, dep1slice, dep2slice = \
                    step_func(indslice, dep1slice, dep2slice)

            if not len(indslice):
                continue

            N = len(indslice)
            pts = np.zeros((2 * N + 2, 2))

            if interpolate:
                def get_interp_point(idx):
                    im1 = max(idx - 1, 0)
                    ind_values = ind[im1:idx+1]
                    diff_values = dep1[im1:idx+1] - dep2[im1:idx+1]
                    dep1_values = dep1[im1:idx+1]

                    if len(diff_values) == 2:
                        if np.ma.is_masked(diff_values[1]):
                            return ind[im1], dep1[im1]
                        elif np.ma.is_masked(diff_values[0]):
                            return ind[idx], dep1[idx]

                    diff_order = diff_values.argsort()
                    diff_root_ind = np.interp(
                        0, diff_values[diff_order], ind_values[diff_order])
                    ind_order = ind_values.argsort()
                    diff_root_dep = np.interp(
                        diff_root_ind,
                        ind_values[ind_order], dep1_values[ind_order])
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/0d89bf6cf4f2a8f4999b3998b54c6ed8caaca8c5/lib/matplotlib/axes/_axes.py/left.py


def test_mismatched_ticklabels()

=======
                    return diff_root_ind, diff_root_dep

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/0d89bf6cf4f2a8f4999b3998b54c6ed8caaca8c5/lib/matplotlib/axes/_axes.py/right.py
:
    fig, ax = plt.subplots()
    ax.plot(np.arange(10))
    ax.xaxis.set_ticks([1.5, 2.5])
    with pytest.raises(ValueError):
        ax.xaxis.set_ticklabels(['a', 'b', 'c'])


def test_empty_ticks_fixed_loc():
    # Smoke test that [] can be used to unset all tick labels
    fig, ax = plt.subplots()
    ax.bar([1, 2], [1, 2])
    ax.set_xticks([1, 2])
    ax.set_xticklabels([])


@image_comparison(['retain_tick_visibility.png'])
def test_retain_tick_visibility():
    fig, ax = plt.subplots()
    plt.plot([0, 1, 2], [0, -1, 4])
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="y", which="both", length=0)


def test_tick_label_update():
    # test issue 9397

    fig, ax = plt.subplots()

    # Set up a dummy formatter
    def formatter_func(x, pos):
        return "unit value" if x == 1 else ""
    ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter_func))

    # Force some of the x-axis ticks to be outside of the drawn range
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.set_xlim(-0.5, 2.5)

    ax.figure.canvas.draw()
    tick_texts = [tick.get_text() for tick in ax.xaxis.get_ticklabels()]
    assert tick_texts == ["", "", "unit value", "", ""]


@image_comparison(['o_marker_path_snap.png'], savefig_kwarg={'dpi': 72})
def test_o_marker_path_snap():
    fig, ax = plt.subplots()
    ax.margins(.1)
    for ms in range(1, 15):
        ax.plot([1, 2, ], np.ones(2) + ms, 'o', ms=ms)

    for ms in np.linspace(1, 10, 25):
        ax.plot([3, 4, ], np.ones(2) + ms, 'o', ms=ms)


def test_margins():
    # test all ways margins can be called
    data = [1, 10]
    xmin = 0.0
    xmax = len(data) - 1.0
    ymin = min(data)
    ymax = max(data)

    fig1, ax1 = plt.subplots(1, 1)
    ax1.plot(data)
    ax1.margins(1)
    assert ax1.margins() == (1, 1)
    assert ax1.get_xlim() == (xmin - (xmax - xmin) * 1,
                              xmax + (xmax - xmin) * 1)
    assert ax1.get_ylim() == (ymin - (ymax - ymin) * 1,
                              ymax + (ymax - ymin) * 1)

    fig2, ax2 = plt.subplots(1, 1)
    ax2.plot(data)
    ax2.margins(0.5, 2)
    assert ax2.margins() == (0.5, 2)
    assert ax2.get_xlim() == (xmin - (xmax - xmin) * 0.5,
                              xmax + (xmax - xmin) * 0.5)
    assert ax2.get_ylim() == (ymin - (ymax - ymin) * 2,
                              ymax + (ymax - ymin) * 2)

    fig3, ax3 = plt.subplots(1, 1)
    ax3.plot(data)
    ax3.margins(x=-0.2, y=0.5)
    assert ax3.margins() == (-0.2, 0.5)
    assert ax3.get_xlim() == (xmin - (xmax - xmin) * -0.2,
                              xmax + (xmax - xmin) * -0.2)
    assert ax3.get_ylim() == (ymin - (ymax - ymin) * 0.5,
                              ymax + (ymax - ymin) * 0.5)


def test_set_margin_updates_limits():
    mpl.style.use("default")
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    ax.set(xscale="log", xmargin=0)
    assert ax.get_xlim() == (1, 2)


def test_length_one_hist():
    fig, ax = plt.subplots()
    ax.hist(1)
    ax.hist([1])


def test_pathological_hexbin():
    # issue #2863
    mylist = [10] * 100
    fig, ax = plt.subplots(1, 1)
    ax.hexbin(mylist, mylist)
    fig.savefig(io.BytesIO())  # Check that no warning is emitted.


def test_color_None():
    # issue 3855
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], color=None)


def test_color_alias():
    # issues 4157 and 4162
    fig, ax = plt.subplots()
    line = ax.plot([0, 1], c='lime')[0]
    assert 'lime' == line.get_color()


def test_numerical_hist_label():
    fig, ax = plt.subplots()
    ax.hist([range(15)] * 5, label=range(5))
    ax.legend()


def test_unicode_hist_label():
    fig, ax = plt.subplots()
    a = (b'\xe5\xbe\x88\xe6\xbc\x82\xe4\xba\xae, ' +
         b'r\xc3\xb6m\xc3\xa4n ch\xc3\xa4r\xc3\xa1ct\xc3\xa8rs')
    b = b'\xd7\xa9\xd7\x9c\xd7\x95\xd7\x9d'
    labels = [a.decode('utf-8'),
              'hi aardvark',
              b.decode('utf-8'),
              ]

    ax.hist([range(15)] * 3, label=labels)
    ax.legend()


def test_move_offsetlabel():
    data = np.random.random(10) * 1e-22

    fig, ax = plt.subplots()
    ax.plot(data)
    fig.canvas.draw()
    before = ax.yaxis.offsetText.get_position()
    assert ax.yaxis.offsetText.get_horizontalalignment() == 'left'
    ax.yaxis.tick_right()
    fig.canvas.draw()
    after = ax.yaxis.offsetText.get_position()
    assert after[0] > before[0] and after[1] == before[1]
    assert ax.yaxis.offsetText.get_horizontalalignment() == 'right'

    fig, ax = plt.subplots()
    ax.plot(data)
    fig.canvas.draw()
    before = ax.xaxis.offsetText.get_position()
    assert ax.xaxis.offsetText.get_verticalalignment() == 'top'
    ax.xaxis.tick_top()
    fig.canvas.draw()
    after = ax.xaxis.offsetText.get_position()
    assert after[0] == before[0] and after[1] > before[1]
    assert ax.xaxis.offsetText.get_verticalalignment() == 'bottom'


@image_comparison(['rc_spines.png'], savefig_kwarg={'dpi': 40})
def test_rc_spines():
    rc_dict = {
        'axes.spines.left': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.spines.bottom': False}
    with matplotlib.rc_context(rc_dict):
        plt.subplots()  # create a figure and axes with the spine properties


@image_comparison(['rc_grid.png'], savefig_kwarg={'dpi': 40})
def test_rc_grid():
    fig = plt.figure()
    rc_dict0 = {
        'axes.grid': True,
        'axes.grid.axis': 'both'
    }
    rc_dict1 = {
        'axes.grid': True,
        'axes.grid.axis': 'x'
    }
    rc_dict2 = {
        'axes.grid': True,
        'axes.grid.axis': 'y'
    }
    dict_list = [rc_dict0, rc_dict1, rc_dict2]

    for i, rc_dict in enumerate(dict_list, 1):
        with matplotlib.rc_context(rc_dict):
            fig.add_subplot(3, 1, i)


def test_rc_tick():
    d = {'xtick.bottom': False, 'xtick.top': True,
         'ytick.left': True, 'ytick.right': False}
    with plt.rc_context(rc=d):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        xax = ax1.xaxis
        yax = ax1.yaxis
        # tick1On bottom/left
        assert not xax._major_tick_kw['tick1On']
        assert xax._major_tick_kw['tick2On']
        assert not xax._minor_tick_kw['tick1On']
        assert xax._minor_tick_kw['tick2On']

        assert yax._major_tick_kw['tick1On']
        assert not yax._major_tick_kw['tick2On']
        assert yax._minor_tick_kw['tick1On']
        assert not yax._minor_tick_kw['tick2On']


def test_rc_major_minor_tick():
    d = {'xtick.top': True, 'ytick.right': True,  # Enable all ticks
         'xtick.bottom': True, 'ytick.left': True,
         # Selectively disable
         'xtick.minor.bottom': False, 'xtick.major.bottom': False,
         'ytick.major.left': False, 'ytick.minor.left': False}
    with plt.rc_context(rc=d):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        xax = ax1.xaxis
        yax = ax1.yaxis
        # tick1On bottom/left
        assert not xax._major_tick_kw['tick1On']
        assert xax._major_tick_kw['tick2On']
        assert not xax._minor_tick_kw['tick1On']
        assert xax._minor_tick_kw['tick2On']

        assert not yax._major_tick_kw['tick1On']
        assert yax._major_tick_kw['tick2On']
        assert not yax._minor_tick_kw['tick1On']
        assert yax._minor_tick_kw['tick2On']


def test_square_plot():
    x = np.arange(4)
    y = np.array([1., 3., 5., 7.])
    fig, ax = plt.subplots()
    ax.plot(x, y, 'mo')
    ax.axis('square')
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert np.diff(xlim) == np.diff(ylim)
    assert ax.get_aspect() == 1
    assert_array_almost_equal(
        ax.get_position(original=True).extents, (0.125, 0.1, 0.9, 0.9))
    assert_array_almost_equal(
        ax.get_position(original=False).extents, (0.2125, 0.1, 0.8125, 0.9))


def test_bad_plot_args():
    with pytest.raises(ValueError):
        plt.plot(None)
    with pytest.raises(ValueError):
        plt.plot(None, None)
    with pytest.raises(ValueError):
        plt.plot(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        plt.plot((np.arange(5).reshape((1, -1)), np.arange(5).reshape(-1, 1)))


@pytest.mark.parametrize(
    "xy, cls", [
        ((), mpl.image.AxesImage),  # (0, N)
        (((3, 7), (2, 6)), mpl.image.AxesImage),  # (xmin, xmax)
        ((range(5), range(4)), mpl.image.AxesImage),  # regular grid
        (([1, 2, 4, 8, 16], [0, 1, 2, 3]),  # irregular grid
         mpl.image.PcolorImage),
        ((np.random.random((4, 5)), np.random.random((4, 5))),  # 2D coords
         mpl.collections.QuadMesh),
    ]
)
@pytest.mark.parametrize(
    "data", [np.arange(12).reshape((3, 4)), np.random.rand(3, 4, 3)]
)
def test_pcolorfast(xy, data, cls):
    fig, ax = plt.subplots()
    assert type(ax.pcolorfast(*xy, data)) == cls


def test_shared_scale():
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    axs[0, 0].set_xscale("log")
    axs[0, 0].set_yscale("log")

    for ax in axs.flat:
        assert ax.get_yscale() == 'log'
        assert ax.get_xscale() == 'log'

    axs[1, 1].set_xscale("linear")
    axs[1, 1].set_yscale("linear")

    for ax in axs.flat:
        assert ax.get_yscale() == 'linear'
        assert ax.get_xscale() == 'linear'


def test_shared_bool():
    with pytest.raises(TypeError):
        plt.subplot(sharex=True)
    with pytest.raises(TypeError):
        plt.subplot(sharey=True)


def test_violin_point_mass():
    """Violin plot should handle point mass pdf gracefully."""
    plt.violinplot(np.array([0, 0]))


def generate_errorbar_inputs():
    base_xy = cycler('x', [np.arange(5)]) + cycler('y', [np.ones(5)])
    err_cycler = cycler('err', [1,
                                [1, 1, 1, 1, 1],
                                [[1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1]],
                                np.ones(5),
                                np.ones((2, 5)),
                                None
                                ])
    xerr_cy = cycler('xerr', err_cycler)
    yerr_cy = cycler('yerr', err_cycler)

    empty = ((cycler('x', [[]]) + cycler('y', [[]])) *
             cycler('xerr', [[], None]) * cycler('yerr', [[], None]))
    xerr_only = base_xy * xerr_cy
    yerr_only = base_xy * yerr_cy
    both_err = base_xy * yerr_cy * xerr_cy

    return [*xerr_only, *yerr_only, *both_err, *empty]


@pytest.mark.parametrize('kwargs', generate_errorbar_inputs())
def test_errorbar_inputs_shotgun(kwargs):
    ax = plt.gca()
    eb = ax.errorbar(**kwargs)
    eb.remove()


@image_comparison(["dash_offset"], remove_text=True)
def test_dash_offset():
    fig, ax = plt.subplots()
    x = np.linspace(0, 10)
    y = np.ones_like(x)
    for j in range(0, 100, 2):
        ax.plot(x, j*y, ls=(j, (10, 10)), lw=5, color='k')


def test_title_pad():
    # check that title padding puts the title in the right
    # place...
    fig, ax = plt.subplots()
    ax.set_title('aardvark', pad=30.)
    m = ax.titleOffsetTrans.get_matrix()
    assert m[1, -1] == (30. / 72. * fig.dpi)
    ax.set_title('aardvark', pad=0.)
    m = ax.titleOffsetTrans.get_matrix()
    assert m[1, -1] == 0.
    # check that it is reverted...
    ax.set_title('aardvark', pad=None)
    m = ax.titleOffsetTrans.get_matrix()
    assert m[1, -1] == (matplotlib.rcParams['axes.titlepad'] / 72. * fig.dpi)


def test_title_location_roundtrip():
    fig, ax = plt.subplots()
    # set default title location
    plt.rcParams['axes.titlelocation'] = 'center'
    ax.set_title('aardvark')
    ax.set_title('left', loc='left')
    ax.set_title('right', loc='right')

    assert 'left' == ax.get_title(loc='left')
    assert 'right' == ax.get_title(loc='right')
    assert 'aardvark' == ax.get_title(loc='center')

    with pytest.raises(ValueError):
        ax.get_title(loc='foo')
    with pytest.raises(ValueError):
        ax.set_title('fail', loc='foo')


@image_comparison(["loglog.png"], remove_text=True, tol=0.02)
def test_loglog():
    fig, ax = plt.subplots()
    x = np.arange(1, 11)
    ax.loglog(x, x**3, lw=5)
    ax.tick_params(length=25, width=2)
    ax.tick_params(length=15, width=2, which='minor')


@image_comparison(["test_loglog_nonpos.png"], remove_text=True, style='mpl20')
def test_loglog_nonpos():
    fig, axs = plt.subplots(3, 3)
    x = np.arange(1, 11)
    y = x**3
    y[7] = -3.
    x[4] = -10
    for (mcy, mcx), ax in zip(product(['mask', 'clip', ''], repeat=2),
                              axs.flat):
        if mcx == mcy:
            if mcx:
                ax.loglog(x, y**3, lw=2, nonpositive=mcx)
            else:
                ax.loglog(x, y**3, lw=2)
        else:
            ax.loglog(x, y**3, lw=2)
            if mcx:
                ax.set_xscale("log", nonpositive=mcx)
            if mcy:
                ax.set_yscale("log", nonpositive=mcy)


@mpl.style.context('default')
def test_axes_margins():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2, 3])
    assert ax.get_ybound()[0] != 0

    fig, ax = plt.subplots()
    ax.bar([0, 1, 2, 3], [1, 1, 1, 1])
    assert ax.get_ybound()[0] == 0

    fig, ax = plt.subplots()
    ax.barh([0, 1, 2, 3], [1, 1, 1, 1])
    assert ax.get_xbound()[0] == 0

    fig, ax = plt.subplots()
    ax.pcolor(np.zeros((10, 10)))
    assert ax.get_xbound() == (0, 10)
    assert ax.get_ybound() == (0, 10)

    fig, ax = plt.subplots()
    ax.pcolorfast(np.zeros((10, 10)))
    assert ax.get_xbound() == (0, 10)
    assert ax.get_ybound() == (0, 10)

    fig, ax = plt.subplots()
    ax.hist(np.arange(10))
    assert ax.get_ybound()[0] == 0

    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)))
    assert ax.get_xbound() == (-0.5, 9.5)
    assert ax.get_ybound() == (-0.5, 9.5)


@pytest.fixture(params=['x', 'y'])
def shared_axis_remover(request):
    def _helper_x(ax):
        ax2 = ax.twinx()
        ax2.remove()
        ax.set_xlim(0, 15)
        r = ax.xaxis.get_major_locator()()
        assert r[-1] > 14

    def _helper_y(ax):
        ax2 = ax.twiny()
        ax2.remove()
        ax.set_ylim(0, 15)
        r = ax.yaxis.get_major_locator()()
        assert r[-1] > 14

    return {"x": _helper_x, "y": _helper_y}[request.param]


@pytest.fixture(params=['gca', 'subplots', 'subplots_shared', 'add_axes'])
def shared_axes_generator(request):
    # test all of the ways to get fig/ax sets
    if request.param == 'gca':
        fig = plt.figure()
        ax = fig.gca()
    elif request.param == 'subplots':
        fig, ax = plt.subplots()
    elif request.param == 'subplots_shared':
        fig, ax_lst = plt.subplots(2, 2, sharex='all', sharey='all')
        ax = ax_lst[0][0]
    elif request.param == 'add_axes':
        fig = plt.figure()
        ax = fig.add_axes([.1, .1, .8, .8])
    return fig, ax


def test_remove_shared_axes(shared_axes_generator, shared_axis_remover):
    # test all of the ways to get fig/ax sets
    fig, ax = shared_axes_generator
    shared_axis_remover(ax)


def test_remove_shared_axes_relim():
    fig, ax_lst = plt.subplots(2, 2, sharex='all', sharey='all')
    ax = ax_lst[0][0]
    orig_xlim = ax_lst[0][1].get_xlim()
    ax.remove()
    ax.set_xlim(0, 5)
    assert_array_equal(ax_lst[0][1].get_xlim(), orig_xlim)


def test_shared_axes_autoscale():
    l = np.arange(-80, 90, 40)
    t = np.random.random_sample((l.size, l.size))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    ax1.set_xlim(-1000, 1000)
    ax1.set_ylim(-1000, 1000)
    ax1.contour(l, l, t)

    ax2.contour(l, l, t)
    assert not ax1.get_autoscalex_on() and not ax2.get_autoscalex_on()
    assert not ax1.get_autoscaley_on() and not ax2.get_autoscaley_on()
    assert ax1.get_xlim() == ax2.get_xlim() == (-1000, 1000)
    assert ax1.get_ylim() == ax2.get_ylim() == (-1000, 1000)


def test_adjust_numtick_aspect():
    fig, ax = plt.subplots()
    ax.yaxis.get_major_locator().set_params(nbins='auto')
    ax.set_xlim(0, 1000)
    ax.set_aspect('equal')
    fig.canvas.draw()
    assert len(ax.yaxis.get_major_locator()()) == 2
    ax.set_ylim(0, 1000)
    fig.canvas.draw()
    assert len(ax.yaxis.get_major_locator()()) > 2


@image_comparison(["auto_numticks.png"], style='default')
def test_auto_numticks():
    # Make tiny, empty subplots, verify that there are only 3 ticks.
    plt.subplots(4, 4)


@image_comparison(["auto_numticks_log.png"], style='default')
def test_auto_numticks_log():
    # Verify that there are not too many ticks with a large log range.
    fig, ax = plt.subplots()
    matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
    ax.loglog([1e-20, 1e5], [1e-16, 10])


def test_broken_barh_empty():
    fig, ax = plt.subplots()
    ax.broken_barh([], (.1, .5))


def test_broken_barh_timedelta():
    """Check that timedelta works as x, dx pair for this method."""
    fig, ax = plt.subplots()
    d0 = datetime.datetime(2018, 11, 9, 0, 0, 0)
    pp = ax.broken_barh([(d0, datetime.timedelta(hours=1))], [1, 2])
    assert pp.get_paths()[0].vertices[0, 0] == mdates.date2num(d0)
    assert pp.get_paths()[0].vertices[2, 0] == mdates.date2num(d0) + 1 / 24


def test_pandas_pcolormesh(pd):
    time = pd.date_range('2000-01-01', periods=10)
    depth = np.arange(20)
    data = np.random.rand(19, 9)

    fig, ax = plt.subplots()
    ax.pcolormesh(time, depth, data)


def test_pandas_indexing_dates(pd):
    dates = np.arange('2005-02', '2005-03', dtype='datetime64[D]')
    values = np.sin(np.array(range(len(dates))))
    df = pd.DataFrame({'dates': dates, 'values': values})

    ax = plt.gca()

    without_zero_index = df[np.array(df.index) % 2 == 1].copy()
    ax.plot('dates', 'values', data=without_zero_index)


def test_pandas_errorbar_indexing(pd):
    df = pd.DataFrame(np.random.uniform(size=(5, 4)),
                      columns=['x', 'y', 'xe', 'ye'],
                      index=[1, 2, 3, 4, 5])
    fig, ax = plt.subplots()
    ax.errorbar('x', 'y', xerr='xe', yerr='ye', data=df)


def test_pandas_index_shape(pd):
    df = pd.DataFrame({"XX": [4, 5, 6], "YY": [7, 1, 2]})
    fig, ax = plt.subplots()
    ax.plot(df.index, df['YY'])


def test_pandas_indexing_hist(pd):
    ser_1 = pd.Series(data=[1, 2, 2, 3, 3, 4, 4, 4, 4, 5])
    ser_2 = ser_1.iloc[1:]
    fig, ax = plt.subplots()
    ax.hist(ser_2)


def test_pandas_bar_align_center(pd):
    # Tests fix for issue 8767
    df = pd.DataFrame({'a': range(2), 'b': range(2)})

    fig, ax = plt.subplots(1)

    ax.bar(df.loc[df['a'] == 1, 'b'],
           df.loc[df['a'] == 1, 'b'],
           align='center')

    fig.canvas.draw()


def test_tick_apply_tickdir_deprecation():
    # Remove this test when the deprecation expires.
    import matplotlib.axis as maxis
    ax = plt.axes()

    tick = maxis.XTick(ax, 0)
    with pytest.warns(MatplotlibDeprecationWarning,
                      match="The apply_tickdir function was deprecated in "
                            "Matplotlib 3.5"):
        tick.apply_tickdir('out')

    tick = maxis.YTick(ax, 0)
    with pytest.warns(MatplotlibDeprecationWarning,
                      match="The apply_tickdir function was deprecated in "
                            "Matplotlib 3.5"):
        tick.apply_tickdir('out')


def test_axis_set_tick_params_labelsize_labelcolor():
    # Tests fix for issue 4346
    axis_1 = plt.subplot()
    axis_1.yaxis.set_tick_params(labelsize=30, labelcolor='red',
                                 direction='out')

    # Expected values after setting the ticks
    assert axis_1.yaxis.majorTicks[0]._size == 4.0
    assert axis_1.yaxis.majorTicks[0].tick1line.get_color() == 'k'
    assert axis_1.yaxis.majorTicks[0].label1.get_size() == 30.0
    assert axis_1.yaxis.majorTicks[0].label1.get_color() == 'red'


def test_axes_tick_params_gridlines():
    # Now treating grid params like other Tick params
    ax = plt.subplot()
    ax.tick_params(grid_color='b', grid_linewidth=5, grid_alpha=0.5,
                   grid_linestyle='dashdot')
    for axis in ax.xaxis, ax.yaxis:
        assert axis.majorTicks[0].gridline.get_color() == 'b'
        assert axis.majorTicks[0].gridline.get_linewidth() == 5
        assert axis.majorTicks[0].gridline.get_alpha() == 0.5
        assert axis.majorTicks[0].gridline.get_linestyle() == '-.'


def test_axes_tick_params_ylabelside():
    # Tests fix for issue 10267
    ax = plt.subplot()
    ax.tick_params(labelleft=False, labelright=True,
                   which='major')
    ax.tick_params(labelleft=False, labelright=True,
                   which='minor')
    # expects left false, right true
    assert ax.yaxis.majorTicks[0].label1.get_visible() is False
    assert ax.yaxis.majorTicks[0].label2.get_visible() is True
    assert ax.yaxis.minorTicks[0].label1.get_visible() is False
    assert ax.yaxis.minorTicks[0].label2.get_visible() is True


def test_axes_tick_params_xlabelside():
    # Tests fix for issue 10267
    ax = plt.subplot()
    ax.tick_params(labeltop=True, labelbottom=False,
                   which='major')
    ax.tick_params(labeltop=True, labelbottom=False,
                   which='minor')
    # expects top True, bottom False
    # label1.get_visible() mapped to labelbottom
    # label2.get_visible() mapped to labeltop
    assert ax.xaxis.majorTicks[0].label1.get_visible() is False
    assert ax.xaxis.majorTicks[0].label2.get_visible() is True
    assert ax.xaxis.minorTicks[0].label1.get_visible() is False
    assert ax.xaxis.minorTicks[0].label2.get_visible() is True


def test_none_kwargs():
    ax = plt.figure().subplots()
    ln, = ax.plot(range(32), linestyle=None)
    assert ln.get_linestyle() == '-'


def test_bar_uint8():
    xs = [0, 1, 2, 3]
    b = plt.bar(np.array(xs, dtype=np.uint8), [2, 3, 4, 5], align="edge")
    for (patch, x) in zip(b.patches, xs):
        assert patch.xy[0] == x


@image_comparison(['date_timezone_x.png'], tol=1.0)
def test_date_timezone_x():
    # Tests issue 5575
    time_index = [datetime.datetime(2016, 2, 22, hour=x,
                                    tzinfo=dateutil.tz.gettz('Canada/Eastern'))
                  for x in range(3)]

    # Same Timezone
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 1, 1)
    plt.plot_date(time_index, [3] * 3, tz='Canada/Eastern')

    # Different Timezone
    plt.subplot(2, 1, 2)
    plt.plot_date(time_index, [3] * 3, tz='UTC')


@image_comparison(['date_timezone_y.png'])
def test_date_timezone_y():
    # Tests issue 5575
    time_index = [datetime.datetime(2016, 2, 22, hour=x,
                                    tzinfo=dateutil.tz.gettz('Canada/Eastern'))
                  for x in range(3)]

    # Same Timezone
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 1, 1)
    plt.plot_date([3] * 3,
                  time_index, tz='Canada/Eastern', xdate=False, ydate=True)

    # Different Timezone
    plt.subplot(2, 1, 2)
    plt.plot_date([3] * 3, time_index, tz='UTC', xdate=False, ydate=True)


@image_comparison(['date_timezone_x_and_y.png'], tol=1.0)
def test_date_timezone_x_and_y():
    # Tests issue 5575
    UTC = datetime.timezone.utc
    time_index = [datetime.datetime(2016, 2, 22, hour=x, tzinfo=UTC)
                  for x in range(3)]

    # Same Timezone
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 1, 1)
    plt.plot_date(time_index, time_index, tz='UTC', ydate=True)

    # Different Timezone
    plt.subplot(2, 1, 2)
    plt.plot_date(time_index, time_index, tz='US/Eastern', ydate=True)


@image_comparison(['axisbelow.png'], remove_text=True)
def test_axisbelow():
    # Test 'line' setting added in 6287.
    # Show only grids, not frame or ticks, to make this test
    # independent of future change to drawing order of those elements.
    axs = plt.figure().subplots(ncols=3, sharex=True, sharey=True)
    settings = (False, 'line', True)

    for ax, setting in zip(axs, settings):
        ax.plot((0, 10), (0, 10), lw=10, color='m')
        circ = mpatches.Circle((3, 3), color='r')
        ax.add_patch(circ)
        ax.grid(color='c', linestyle='-', linewidth=3)
        ax.tick_params(top=False, bottom=False,
                       left=False, right=False)
        ax.spines[:].set_visible(False)
        ax.set_axisbelow(setting)


def test_titletwiny():
    plt.style.use('mpl20')
    fig, ax = plt.subplots(dpi=72)
    ax2 = ax.twiny()
    xlabel2 = ax2.set_xlabel('Xlabel2')
    title = ax.set_title('Title')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # ------- Test that title is put above Xlabel2 (Xlabel2 at top) ----------
    bbox_y0_title = title.get_window_extent(renderer).y0  # bottom of title
    bbox_y1_xlabel2 = xlabel2.get_window_extent(renderer).y1  # top of xlabel2
    y_diff = bbox_y0_title - bbox_y1_xlabel2
    assert np.isclose(y_diff, 3)


def test_titlesetpos():
    # Test that title stays put if we set it manually
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.8)
    ax2 = ax.twiny()
    ax.set_xlabel('Xlabel')
    ax2.set_xlabel('Xlabel2')
    ax.set_title('Title')
    pos = (0.5, 1.11)
    ax.title.set_position(pos)
    renderer = fig.canvas.get_renderer()
    ax._update_title_position(renderer)
    assert ax.title.get_position() == pos


def test_title_xticks_top():
    # Test that title moves if xticks on top of axes.
    mpl.rcParams['axes.titley'] = None
    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position('top')
    ax.set_title('xlabel top')
    fig.canvas.draw()
    assert ax.title.get_position()[1] > 1.04


def test_title_xticks_top_both():
    # Test that title moves if xticks on top of axes.
    mpl.rcParams['axes.titley'] = None
    fig, ax = plt.subplots()
    ax.tick_params(axis="x",
                   bottom=True, top=True, labelbottom=True, labeltop=True)
    ax.set_title('xlabel top')
    fig.canvas.draw()
    assert ax.title.get_position()[1] > 1.04


def test_title_no_move_off_page():
    # If an axes is off the figure (ie. if it is cropped during a save)
    # make sure that the automatic title repositioning does not get done.
    mpl.rcParams['axes.titley'] = None
    fig = plt.figure()
    ax = fig.add_axes([0.1, -0.5, 0.8, 0.2])
    ax.tick_params(axis="x",
                   bottom=True, top=True, labelbottom=True, labeltop=True)
    tt = ax.set_title('Boo')
    fig.canvas.draw()
    assert tt.get_position()[1] == 1.0


def test_offset_label_color():
    # Tests issue 6440
    fig, ax = plt.subplots()
    ax.plot([1.01e9, 1.02e9, 1.03e9])
    ax.yaxis.set_tick_params(labelcolor='red')
    assert ax.yaxis.get_offset_text().get_color() == 'red'


def test_offset_text_visible():
    fig, ax = plt.subplots()
    ax.plot([1.01e9, 1.02e9, 1.03e9])
    ax.yaxis.set_tick_params(label1On=False, label2On=True)
    assert ax.yaxis.get_offset_text().get_visible()
    ax.yaxis.set_tick_params(label2On=False)
    assert not ax.yaxis.get_offset_text().get_visible()


def test_large_offset():
    fig, ax = plt.subplots()
    ax.plot((1 + np.array([0, 1.e-12])) * 1.e27)
    fig.canvas.draw()


def test_barb_units():
    fig, ax = plt.subplots()
    dates = [datetime.datetime(2017, 7, 15, 18, i) for i in range(0, 60, 10)]
    y = np.linspace(0, 5, len(dates))
    u = v = np.linspace(0, 50, len(dates))
    ax.barbs(dates, y, u, v)


def test_quiver_units():
    fig, ax = plt.subplots()
    dates = [datetime.datetime(2017, 7, 15, 18, i) for i in range(0, 60, 10)]
    y = np.linspace(0, 5, len(dates))
    u = v = np.linspace(0, 50, len(dates))
    ax.quiver(dates, y, u, v)


def test_bar_color_cycle():
    to_rgb = mcolors.to_rgb
    fig, ax = plt.subplots()
    for j in range(5):
        ln, = ax.plot(range(3))
        brs = ax.bar(range(3), range(3))
        for br in brs:
            assert to_rgb(ln.get_color()) == to_rgb(br.get_facecolor())


def test_tick_param_label_rotation():
    fix, (ax, ax2) = plt.subplots(1, 2)
    ax.plot([0, 1], [0, 1])
    ax2.plot([0, 1], [0, 1])
    ax.xaxis.set_tick_params(which='both', rotation=75)
    ax.yaxis.set_tick_params(which='both', rotation=90)
    for text in ax.get_xticklabels(which='both'):
        assert text.get_rotation() == 75
    for text in ax.get_yticklabels(which='both'):
        assert text.get_rotation() == 90

    ax2.tick_params(axis='x', labelrotation=53)
    ax2.tick_params(axis='y', rotation=35)
    for text in ax2.get_xticklabels(which='major'):
        assert text.get_rotation() == 53
    for text in ax2.get_yticklabels(which='major'):
        assert text.get_rotation() == 35


@mpl.style.context('default')
def test_fillbetween_cycle():
    fig, ax = plt.subplots()

    for j in range(3):
        cc = ax.fill_between(range(3), range(3))
        target = mcolors.to_rgba('C{}'.format(j))
        assert tuple(cc.get_facecolors().squeeze()) == tuple(target)

    for j in range(3, 6):
        cc = ax.fill_betweenx(range(3), range(3))
        target = mcolors.to_rgba('C{}'.format(j))
        assert tuple(cc.get_facecolors().squeeze()) == tuple(target)

    target = mcolors.to_rgba('k')

    for al in ['facecolor', 'facecolors', 'color']:
        cc = ax.fill_between(range(3), range(3), **{al: 'k'})
        assert tuple(cc.get_facecolors().squeeze()) == tuple(target)

    edge_target = mcolors.to_rgba('k')
    for j, el in enumerate(['edgecolor', 'edgecolors'], start=6):
        cc = ax.fill_between(range(3), range(3), **{el: 'k'})
        face_target = mcolors.to_rgba('C{}'.format(j))
        assert tuple(cc.get_facecolors().squeeze()) == tuple(face_target)
        assert tuple(cc.get_edgecolors().squeeze()) == tuple(edge_target)


def test_log_margins():
    plt.rcParams['axes.autolimit_mode'] = 'data'
    fig, ax = plt.subplots()
    margin = 0.05
    ax.set_xmargin(margin)
    ax.semilogx([10, 100], [10, 100])
    xlim0, xlim1 = ax.get_xlim()
    transform = ax.xaxis.get_transform()
fine the rows and columns of
            the image.

            This parameter can only be passed positionally.

        X, Y : tuple or array-like, default: ``(0, N)``, ``(0, M)``
            *X* and *Y* are used to specify the coordinates of the
            quadrilaterals. There are different ways to do this:

            - Use tuples ``X=(xmin, xmax)`` and ``Y=(ymin, ymax)`` to define
              a *uniform rectangular grid*.

              The tuples define the outer edges of the grid. All individual
              quadrilaterals will be of the same size. This is the fastest
              version.

            - Use 1D arrays *X*, *Y* to specify a *non-uniform rectangular
              grid*.

              In this case *X* and *Y* have to be monotonic 1D arrays of length
              *N+1* and *M+1*, specifying the x and y boundaries of the cells.

              The speed is intermediate. Note: The grid is checked, and if
              found to be uniform the fast version is used.

            - Use 2D arrays *X*, *Y* if you need an *arbitrary quadrilateral
              grid* (i.e. if the quadrilaterals are not rectangular).

              In this case *X* and *Y* are 2D arrays with shape (M + 1, N + 1),
              specifying the x and y coordinates of the corners of the colored
              quadrilaterals.

              This is the most general, but the slowest to render.  It may
              produce faster and more compact output using ps, pdf, and
              svg backends, however.

            These arguments can only be passed positionally.

        cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            A Colormap instance or registered colormap name. The colormap
            maps the *C* values to colors.

        norm : `~matplotlib.colors.Normalize`, optional
            The Normalize instance scales the data values to the canonical
            colormap range [0, 1] for mapping to colors. By default, the data
            range is mapped to the colorbar range using linear scaling.

        vmin, vmax : float, default: None
            The colorbar range. If *None*, suitable min/max values are
            automatically chosen by the `.Normalize` instance (defaults to
            the respective min/max values of *C* in case of the default linear
            scaling).
            It is an error to use *vmin*/*vmax* when *norm* is given.

        alpha : float, default: None
            The alpha blending value, between 0 (transparent) and 1 (opaque).

        snap : bool, default: False
            Whether to snap the mesh to pixel boundaries.

        Returns
        -------
        `.AxesImage` or `.PcolorImage` or `.QuadMesh`
            The return type depends on the type of grid:

            - `.AxesImage` for a regular rectangular grid.
            - `.PcolorImage` for a non-regular rectangular grid.
            - `.QuadMesh` for a non-rectangular grid.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Supported additional parameters depend on the type of grid.
            See return types of *image* for further description.
        """

        C = args[-1]
        nr, nc = np.shape(C)[:2]
        if len(args) == 1:
            style = "image"
            x = [0, nc]
            y = [0, nr]
        elif len(args) == 3:
            x, y = args[:2]
            x = np.asarray(x)
            y = np.asarray(y)
            if x.ndim == 1 and y.ndim == 1:
                if x.size == 2 and y.size == 2:
                    style = "image"
                else:
                    dx = np.diff(x)
                    dy = np.diff(y)
                    if (np.ptp(dx) < 0.01 * abs(dx.mean()) and
                            np.ptp(dy) < 0.01 * abs(dy.mean())):
                        style = "image"
                    else:
                        style = "pcolorimage"
            elif x.ndim == 2 and y.ndim == 2:
                style = "quadmesh"
            else:
                raise TypeError("arguments do not match valid signatures")
        else:
            raise TypeError("need 1 argument or 3 arguments")

        if style == "quadmesh":
            # data point in each cell is value at lower left corner
            coords = np.stack([x, y], axis=-1)
            if np.ndim(C) == 2:
                qm_kwargs = {"array": np.ma.ravel(C)}
            elif np.ndim(C) == 3:
                qm_kwargs = {"color": np.ma.reshape(C, (-1, C.shape[-1]))}
            else:
                raise ValueError("C must be 2D or 3D")
            collection = mcoll.QuadMesh(
                coords, **qm_kwargs,
                alpha=alpha, cmap=cmap, norm=norm,
                antialiased=False, edgecolors="none")
            self.add_collection(collection, autolim=False)
            xl, xr, yb, yt = x.min(), x.max(), y.min(), y.max()
            ret = collection

        else:  # It's one of the two image styles.
            extent = xl, xr, yb, yt = x[0], x[-1], y[0], y[-1]
            if style == "image":
                im = mimage.AxesImage(
                    self, cmap, norm,
                    data=C, alpha=alpha, extent=extent,
                    interpolation='nearest', origin='lower',
                    **kwargs)
            elif style == "pcolorimage":
                im = mimage.PcolorImage(
                    self, x, y, C,
                    cmap=cmap, norm=norm, alpha=alpha, extent=extent,
                    **kwargs)
            self.add_image(im)
            ret = im

        if np.ndim(C) == 2:  # C.ndim == 3 is RGB(A) so doesn't need scaling.
            ret._scale_norm(norm, vmin, vmax)

        if ret.get_clip_path() is None:
            # image does not already have clipping set, clip to axes patch
            ret.set_clip_path(self.patch)

        ret.sticky_edges.x[:] = [xl, xr]
        ret.sticky_edges.y[:] = [yb, yt]
        self.update_datalim(np.array([[xl, yb], [xr, yt]]))
        self._request_autoscale_view(tight=True)
        return ret

    @_preprocess_data()
    @docstring.dedent_interpd
    def contour(self, *args, **kwargs):
        """
        Plot contour lines.

        Call signature::

            contour([X, Y,] Z, [levels], **kwargs)
        %(contour_doc)s
        """
        kwargs['filled'] = False
        contours = mcontour.QuadContourSet(self, *args, **kwargs)
        self._request_autoscale_view()
        return contours

    @_preprocess_data()
    @docstring.dedent_interpd
    def contourf(self, *args, **kwargs):
        """
        Plot filled contours.

        Call signature::

            contourf([X, Y,] Z, [levels], **kwargs)
        %(contour_doc)s
        """
        kwargs['filled'] = True
        contours = mcontour.QuadContourSet(self, *args, **kwargs)
        self._request_autoscale_view()
        return contours

    def clabel(self, CS, levels=None, **kwargs):
        """
        Label a contour plot.

        Adds labels to line contours in given `.ContourSet`.

        Parameters
        ----------
        CS : `.ContourSet` instance
            Line contours to label.

        levels : array-like, optional
            A list of level values, that should be labeled. The list must be
            a subset of ``CS.levels``. If not given, all levels are labeled.

        **kwargs
            All other parameters are documented in `~.ContourLabeler.clabel`.
        """
        return CS.clabel(levels, **kwargs)

    #### Data analysis

    @_preprocess_data(replace_names=["x", 'weights'], label_namer="x")
    def hist(self, x, bins=None, range=None, density=False, weights=None,
             cumulative=False, bottom=None, histtype='bar', align='mid',
             orientation='vertical', rwidth=None, log=False,
             color=None, label=None, stacked=False, **kwargs):
        """
        Plot a histogram.

        Compute and draw the histogram of *x*.  The return value is a tuple
        (*n*, *bins*, *patches*) or ([*n0*, *n1*, ...], *bins*, [*patches0*,
        *patches1*, ...]) if the input contains multiple data.  See the
        documentation of the *weights* parameter to draw a histogram of
        already-binned data.

        Multiple data can be provided via *x* as a list of datasets
        of potentially different length ([*x0*, *x1*, ...]), or as
        a 2D ndarray in which each column is a dataset.  Note that
        the ndarray form is transposed relative to the list form.

        Masked arrays are not supported.

        The *bins*, *range*, *weights*, and *density* parameters behave as in
        `numpy.histogram`.

        Parameters
        ----------
        x : (n,) array or sequence of (n,) arrays
            Input values, this takes either a single array or a sequence of
            arrays which are not required to be of the same length.

        bins : int or sequence or str, default: :rc:`hist.bins`
            If *bins* is an integer, it defines the number of equal-width bins
            in the range.

            If *bins* is a sequence, it defines the bin edges, including the
            left edge of the first bin and the right edge of the last bin;
            in this case, bins may be unequally spaced.  All but the last
            (righthand-most) bin is half-open.  In other words, if *bins* is::

                [1, 2, 3, 4]

            then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
            the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
            *includes* 4.

            If *bins* is a string, it is one of the binning strategies
            supported by `numpy.histogram_bin_edges`: 'auto', 'fd', 'doane',
            'scott', 'stone', 'rice', 'sturges', or 'sqrt'.

        range : tuple or None, default: None
            The lower and upper range of the bins. Lower and upper outliers
            are ignored. If not provided, *range* is ``(x.min(), x.max())``.
            Range has no effect if *bins* is a sequence.

            If *bins* is a sequence or *range* is specified, autoscaling
            is based on the specified bin range instead of the
            range of x.

        density : bool, default: False
            If ``True``, draw and return a probability density: each bin
            will display the bin's raw count divided by the total number of
            counts *and the bin width*
            (``density = counts / (sum(counts) * np.diff(bins))``),
            so that the area under the histogram integrates to 1
            (``np.sum(density * np.diff(bins)) == 1``).

            If *stacked* is also ``True``, the sum of the histograms is
            normalized to 1.

        weights : (n,) array-like or None, default: None
            An array of weights, of the same shape as *x*.  Each value in
            *x* only contributes its associated weight towards the bin count
            (instead of 1).  If *density* is ``True``, the weights are
            normalized, so that the integral of the density over the range
            remains 1.

            This parameter can be used to draw a histogram of data that has
            already been binned, e.g. using `numpy.histogram` (by treating each
            bin as a single point with a weight equal to its count) ::

                counts, bins = np.histogram(data)
                plt.hist(bins[:-1], bins, weights=counts)

            (or you may alternatively use `~.bar()`).

        cumulative : bool or -1, default: False
            If ``True``, then a histogram is computed where each bin gives the
            counts in that bin plus all bins for smaller values. The last bin
            gives the total number of datapoints.

            If *density* is also ``True`` then the histogram is normalized such
            that the last bin equals 1.

            If *cumulative* is a number less than 0 (e.g., -1), the direction
            of accumulation is reversed.  In this case, if *density* is also
            ``True``, then the histogram is normalized such that the first bin
            equals 1.

        bottom : array-like, scalar, or None, default: None
            Location of the bottom of each bin, ie. bins are drawn from
            ``bottom`` to ``bottom + hist(x, bins)`` If a scalar, the bottom
            of each bin is shifted by the same amount. If an array, each bin
            is shifted independently and the length of bottom must match the
            number of bins. If None, defaults to 0.

        histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, default: 'bar'
            The type of histogram to draw.

            - 'bar' is a traditional bar-type histogram.  If multiple data
              are given the bars are arranged side by side.
            - 'barstacked' is a bar-type histogram where multiple
              data are stacked on top of each other.
            - 'step' generates a lineplot that is by default unfilled.
            - 'stepfilled' generates a lineplot that is by default filled.

        align : {'left', 'mid', 'right'}, default: 'mid'
            The horizontal alignment of the histogram bars.

            - 'left': bars are centered on the left bin edges.
            - 'mid': bars are centered between the bin edges.
            - 'right': bars are centered on the right bin edges.

        orientation : {'vertical', 'horizontal'}, default: 'vertical'
            If 'horizontal', `~.Axes.barh` will be used for bar-type histograms
            and the *bottom* kwarg will be the left edges.

        rwidth : float or None, default: None
            The relative width of the bars as a fraction of the bin width.  If
            ``None``, automatically compute the width.

            Ignored if *histtype* is 'step' or 'stepfilled'.

        log : bool, default: False
            If ``True``, the histogram axis will be set to a log scale.

        color : color or array-like of colors or None, default: None
            Color or sequence of colors, one per dataset.  Default (``None``)
            uses the standard line color sequence.

        label : str or None, default: None
            String, or sequence of strings to match multiple datasets.  Bar
            charts yield multiple patches per dataset, but only the first gets
            the label, so that `~.Axes.legend` will work as expected.

        stacked : bool, default: False
            If ``True``, multiple data are stacked on top of each other If
            ``False`` multiple data are arranged side by side if histtype is
            'bar' or on top of each other if histtype is 'step'

        Returns
        -------
        n : array or list of arrays
            The values of the histogram bins. See *density* and *weights* for a
            description of the possible semantics.  If input *x* is an array,
            then this is an array of length *nbins*. If input is a sequence of
            arrays ``[data1, data2, ...]``, then this is a list of arrays with
            the values of the histograms for each of the arrays in the same
            order.  The dtype of the array *n* (or of its element arrays) will
            always be float even if no weighting or normalization is used.

        bins : array
            The edges of the bins. Length nbins + 1 (nbins left edges and right
            edge of last bin).  Always a single array even when multiple data
            sets are passed in.

        patches : `.BarContainer` or list of a single `.Polygon` or list of \
such objects
            Container of individual artists used to create the histogram
            or list of such containers if there are multiple input datasets.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            `~matplotlib.patches.Patch` properties

        See Also
        --------
        hist2d : 2D histogram with rectangular bins
        hexbin : 2D histogram with hexagonal bins

        Notes
        -----
        For large numbers of bins (>1000), 'step' and 'stepfilled' can be
        significantly faster than 'bar' and 'barstacked'.

        """
        # Avoid shadowing the builtin.
        bin_range = range
        from builtins import range

        if np.isscalar(x):
            x = [x]

        if bins is None:
            bins = rcParams['hist.bins']

        # Validate string inputs here to avoid cluttering subsequent code.
        _api.check_in_list(['bar', 'barstacked', 'step', 'stepfilled'],
                           histtype=histtype)
        _api.check_in_list(['left', 'mid', 'right'], align=align)
        _api.check_in_list(['horizontal', 'vertical'], orientation=orientation)

        if histtype == 'barstacked' and not stacked:
            stacked = True

        # Massage 'x' for processing.
        x = cbook._reshape_2D(x, 'x')
        nx = len(x)  # number of datasets

        # Process unit information.  _process_unit_info sets the unit and
        # converts the first dataset; then we convert each following dataset
        # one at a time.
        if orientation == "vertical":
            convert_units = self.convert_xunits
            x = [*self._process_unit_info([("x", x[0])], kwargs),
                 *map(convert_units, x[1:])]
        else:  # horizontal
            convert_units = self.convert_yunits
            x = [*self._process_unit_info([("y", x[0])], kwargs),
                 *map(convert_units, x[1:])]

        if bin_range is not None:
            bin_range = convert_units(bin_range)

        if not cbook.is_scalar_or_string(bins):
            bins = convert_units(bins)

        # We need to do to 'weights' what was done to 'x'
        if weights is not None:
            w = cbook._reshape_2D(weights, 'weights')
        else:
            w = [None] * nx

        if len(w) != nx:
            raise ValueError('weights should have the same shape as x')

        input_empty = True
        for xi, wi in zip(x, w):
            len_xi = len(xi)
            if wi is not None and len(wi) != len_xi:
                raise ValueError('weights should have the same shape as x')
            if len_xi:
                input_empty = False

        if color is None:
            color = [self._get_lines.get_next_color() for i in range(nx)]
        else:
            color = mcolors.to_rgba_array(color)
            if len(color) != nx:
                raise ValueError(f"The 'color' keyword argument must have one "
                                 f"color per dataset, but {nx} datasets and "
                                 f"{len(color)} colors were provided")

        hist_kwargs = dict()

        # if the bin_range is not given, compute without nan numpy
        # does not do this for us when guessing the range (but will
        # happily ignore nans when computing the histogram).
        if bin_range is None:
            xmin = np.inf
            xmax = -np.inf
            for xi in x:
                if len(xi):
                    # python's min/max ignore nan,
                    # np.minnan returns nan for all nan input
                    xmin = min(xmin, np.nanmin(xi))
                    xmax = max(xmax, np.nanmax(xi))
            if xmin <= xmax:  # Only happens if we have seen a finite value.
                bin_range = (xmin, xmax)

        # If bins are not specified either explicitly or via range,
        # we need to figure out the range required for all datasets,
        # and supply that to np.histogram.
        if not input_empty and len(x) > 1:
            if weights is not None:
                _w = np.concatenate(w)
            else:
                _w = None
            bins = np.histogram_bin_edges(
                np.concatenate(x), bins, bin_range, _w)
        else:
            hist_kwargs['range'] = bin_range

        density = bool(density)
        if density and not stacked:
            hist_kwargs['density'] = density

        # List to store all the top coordinates of the histograms
        tops = []  # Will have shape (n_datasets, n_bins).
        # Loop through datasets
        for i in range(nx):
            # this will automatically overwrite bins,
            # so that each histogram uses the same bins
            m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
            tops.append(m)
        tops = np.array(tops, float)  # causes problems later if it's an int
        if stacked:
            tops = tops.cumsum(axis=0)
            # If a stacked density plot, normalize so the area of all the
            # stacked histograms together is 1
            if density:
                tops = (tops / np.diff(bins)) / tops[-1].sum()
        if cumulative:
            slc = slice(None)
            if isinstance(cumulative, Number) and cumulative < 0:
                slc = slice(None, None, -1)
            if density:
                tops = (tops * np.diff(bins))[:, slc].cumsum(axis=1)[:, slc]
            else:
                tops = tops[:, slc].cumsum(axis=1)[:, slc]

        patches = []

        if histtype.startswith('bar'):

            totwidth = np.diff(bins)

            if rwidth is not None:
                dr = np.clip(rwidth, 0, 1)
            elif (len(tops) > 1 and
                  ((not stacked) or rcParams['_internal.classic_mode'])):
                dr = 0.8
            else:
                dr = 1.0

            if histtype == 'bar' and not stacked:
                width = dr * totwidth / nx
                dw = width
                boffset = -0.5 * dr * totwidth * (1 - 1 / nx)
            elif histtype == 'barstacked' or stacked:
                width = dr * totwidth
                boffset, dw = 0.0, 0.0

            if align == 'mid':
                boffset += 0.5 * totwidth
            elif align == 'right':
                boffset += totwidth

            if orientation == 'horizontal':
                _barfunc = self.barh
                bottom_kwarg = 'left'
            else:  # orientation == 'vertical'
                _barfunc = self.bar
                bottom_kwarg = 'bottom'

            for m, c in zip(tops, color):
                if bottom is None:
                    bottom = np.zeros(len(m))
                if stacked:
                    height = m - bottom
                else:
                    height = m
                bars = _barfunc(bins[:-1]+boffset, height, width,
                                align='center', log=log,
                                color=c, **{bottom_kwarg: bottom})
                patches.append(bars)
                if stacked:
                    bottom = m
                boffset += dw
            # Remove stickies from all bars but the lowest ones, as otherwise
            # margin expansion would be unable to cross the stickies in the
            # middle of the bars.
            for bars in patches[1:]:
                for patch in bars:
                    patch.sticky_edges.x[:] = patch.sticky_edges.y[:] = []

        elif histtype.startswith('step'):
            # these define the perimeter of the polygon
            x = np.zeros(4 * len(bins) - 3)
            y = np.zeros(4 * len(bins) - 3)

            x[0:2*len(bins)-1:2], x[1:2*len(bins)-1:2] = bins, bins[:-1]
            x[2*len(bins)-1:] = x[1:2*len(bins)-1][::-1]

            if bottom is None:
                bottom = 0

            y[1:2*len(bins)-1:2] = y[2:2*len(bins):2] = bottom
            y[2*len(bins)-1:] = y[1:2*len(bins)-1][::-1]

            if log:
                if orientation == 'horizontal':
                    self.set_xscale('log', nonpositive='clip')
                else:  # orientation == 'vertical'
                    self.set_yscale('log', nonpositive='clip')

            if align == 'left':
                x -= 0.5*(bins[1]-bins[0])
            elif align == 'right':
                x += 0.5*(bins[1]-bins[0])

            # If fill kwarg is set, it will be passed to the patch collection,
            # overriding this
            fill = (histtype == 'stepfilled')

            xvals, yvals = [], []
            for m in tops:
                if stacked:
                    # top of the previous polygon becomes the bottom
                    y[2*len(bins)-1:] = y[1:2*len(bins)-1][::-1]
                # set the top of this polygon
                y[1:2*len(bins)-1:2] = y[2:2*len(bins):2] = m + bottom

                # The starting point of the polygon has not yet been
                # updated. So far only the endpoint was adjusted. This
                # assignment closes the polygon. The redundant endpoint is
                # later discarded (for step and stepfilled).
                y[0] = y[-1]

                if orientation == 'horizontal':
                    xvals.append(y.copy())
                    yvals.append(x.copy())
                else:
                    xvals.append(x.copy())
                    yvals.append(y.copy())

            # stepfill is closed, step is not
            split = -1 if fill else 2 * len(bins)
            # add patches in reverse order so that when stacking,
            # items lower in the stack are plotted on top of
            # items higher in the stack
            for x, y, c in reversed(list(zip(xvals, yvals, color))):
                patches.append(self.fill(
                    x[:split], y[:split],
                    closed=True if fill else None,
                    facecolor=c,
                    edgecolor=None if fill else c,
                    fill=fill if fill else None,
                    zorder=None if fill else mlines.Line2D.zorder))
            for patch_list in patches:
                for patch in patch_list:
                    if orientation == 'vertical':
                        patch.sticky_edges.y.append(0)
                    elif orientation == 'horizontal':
                        patch.sticky_edges.x.append(0)

            # we return patches, so put it back in the expected order
            patches.reverse()

        # If None, make all labels None (via zip_longest below); otherwise,
        # cast each element to str, but keep a single str as it.
        labels = [] if label is None else np.atleast_1d(np.asarray(label, str))
        for patch, lbl in itertools.zip_longest(patches, labels):
            if patch:
                p = patch[0]
                p.update(kwargs)
                if lbl is not None:
                    p.set_label(lbl)
                for p in patch[1:]:
                    p.update(kwargs)
                    p.set_label('_nolegend_')

        if nx == 1:
            return tops[0], bins, patches[0]
        else:
            patch_type = ("BarContainer" if histtype.startswith("bar")
                          else "list[Polygon]")
            return tops, bins, cbook.silent_list(patch_type, patches)

    @_preprocess_data()
    def stairs(self, values, edges=None, *,
               orientation='vertical', baseline=0, fill=False, **kwargs):
        """
        A stepwise constant function as a line with bounding edges
        or a filled plot.

        Parameters
        ----------
        values : array-like
            The step heights.

        edges : array-like
            The edge positions, with ``len(edges) == len(vals) + 1``,
            between which the curve takes on vals values.

        orientation : {'vertical', 'horizontal'}, default: 'vertical'
            The direction of the steps. Vertical means that *values* are along
            the y-axis, and edges are along the x-axis.

        baseline : float, array-like or None, default: 0
            The bottom value of the bounding edges or when
            ``fill=True``, position of lower edge. If *fill* is
            True or an array is passed to *baseline*, a closed
            path is drawn.

        fill : bool, default: False
            Whether the area under the step curve should be filled.

        Returns
        -------
        StepPatch : `matplotlib.patches.StepPatch`

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            `~matplotlib.patches.StepPatch` properties

        """

        if 'color' in kwargs:
            _color = kwargs.pop('color')
        else:
            _color = self._get_lines.get_next_color()
        if fill:
            kwargs.setdefault('edgecolor', 'none')
            kwargs.setdefault('facecolor', _color)
        else:
            kwargs.setdefault('edgecolor', _color)

        if edges is None:
            edges = np.arange(len(values) + 1)

        edges, values, baseline = self._process_unit_info(
            [("x", edges), ("y", values), ("y", baseline)], kwargs)

        patch = mpatches.StepPatch(values,
                                   edges,
                                   baseline=baseline,
                                   orientation=orientation,
                                   fill=fill,
                                   **kwargs)
        self.add_patch(patch)
        if baseline is None:
            baseline = 0
        if orientation == 'vertical':
            patch.sticky_edges.y.append(np.min(baseline))
            self.update_datalim([(edges[0], np.min(baseline))])
        else:
            patch.sticky_edges.x.append(np.min(baseline))
            self.update_datalim([(np.min(baseline), edges[0])])
        self._request_autoscale_view()
        return patch

    @_preprocess_data(replace_names=["x", "y", "weights"])
    @docstring.dedent_interpd
    def hist2d(self, x, y, bins=10, range=None, density=False, weights=None,
               cmin=None, cmax=None, **kwargs):
        """
        Make a 2D histogram plot.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input values

        bins : None or int or [int, int] or array-like or [array, array]

            The bin specification:

            - If int, the number of bins for the two dimensions
              (nx=ny=bins).
            - If ``[int, int]``, the number of bins in each dimension
              (nx, ny = bins).
            - If array-like, the bin edges for the two dimensions
              (x_edges=y_edges=bins).
            - If ``[array, array]``, the bin edges in each dimension
              (x_edges, y_edges = bins).

            The default value is 10.

        range : array-like shape(2, 2), optional
            The leftmost and rightmost edges of the bins along each dimension
            (if not specified explicitly in the bins parameters): ``[[xmin,
            xmax], [ymin, ymax]]``. All values outside of this range will be
            considered outliers and not tallied in the histogram.

        density : bool, default: False
            Normalize histogram.  See the documentation for the *density*
            parameter of `~.Axes.hist` for more details.

        weights : array-like, shape (n, ), optional
            An array of values w_i weighing each sample (x_i, y_i).

        cmin, cmax : float, default: None
            All bins that has count less than *cmin* or more than *cmax* will
            not be displayed (set to NaN before passing to imshow) and these
            count values in the return value count histogram will also be set
            to nan upon return.

        Returns
        -------
        h : 2D array
            The bi-dimensional histogram of samples x and y. Values in x are
            histogrammed along the first dimension and values in y are
            histogrammed along the second dimension.
        xedges : 1D array
            The bin edges along the x axis.
        yedges : 1D array
            The bin edges along the y axis.
        image : `~.matplotlib.collections.QuadMesh`

        Other Parameters
        ----------------
        cmap : Colormap or str, optional
            A `.colors.Colormap` instance.  If not set, use rc settings.

        norm : Normalize, optional
            A `.colors.Normalize` instance is used to
            scale luminance data to ``[0, 1]``. If not set, defaults to
            `.colors.Normalize()`.

        vmin/vmax : None or scalar, optional
            Arguments passed to the `~.colors.Normalize` instance.

        alpha : ``0 <= scalar <= 1`` or ``None``, optional
            The alpha blending value.

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Additional parameters are passed along to the
            `~.Axes.pcolormesh` method and `~matplotlib.collections.QuadMesh`
            constructor.

        See Also
        --------
        hist : 1D histogram plotting
        hexbin : 2D histogram with hexagonal bins

        Notes
        -----
        - Currently ``hist2d`` calculates its own axis limits, and any limits
          previously set are ignored.
        - Rendering the histogram with a logarithmic color scale is
          accomplished by passing a `.colors.LogNorm` instance to the *norm*
          keyword argument. Likewise, power-law normalization (similar
          in effect to gamma correction) can be accomplished with
          `.colors.PowerNorm`.
        """

        h, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range,
                                           density=density, weights=weights)

        if cmin is not None:
            h[h < cmin] = None
        if cmax is not None:
            h[h > cmax] = None

        pc = self.pcolormesh(xedges, yedges, h.T, **kwargs)
        self.set_xlim(xedges[0], xedges[-1])
        self.set_ylim(yedges[0], yedges[-1])

        return h, xedges, yedges, pc

    @_preprocess_data(replace_names=["x"])
    @docstring.dedent_interpd
    def psd(self, x, NFFT=None, Fs=None, Fc=None, detrend=None,
            window=None, noverlap=None, pad_to=None,
            sides=None, scale_by_freq=None, return_line=None, **kwargs):
        r"""
        Plot the power spectral density.

        The power spectral density :math:`P_{xx}` by Welch's average
        periodogram method.  The vector *x* is divided into *NFFT* length
        segments.  Each segment is detrended by function *detrend* and
        windowed by function *window*.  *noverlap* gives the length of
        the overlap between segments.  The :math:`|\mathrm{fft}(i)|^2`
        of each segment :math:`i` are averaged to compute :math:`P_{xx}`,
        with a scaling to correct for power loss due to windowing.

        If len(*x*) < *NFFT*, it will be zero padded to *NFFT*.

        Parameters
        ----------
        x : 1-D array or sequence
            Array or sequence containing the data

        %(Spectral)s

        %(PSD)s

        noverlap : int, default: 0 (no overlap)
            The number of points of overlap between segments.

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        return_line : bool, default: False
            Whether to include the line object plotted in the returned values.

        Returns
        -------
        Pxx : 1-D array
            The values for the power spectrum :math:`P_{xx}` before scaling
            (real valued).

        freqs : 1-D array
            The frequencies corresponding to the elements in *Pxx*.

        line : `~matplotlib.lines.Line2D`
            The line created by this function.
            Only returned if *return_line* is True.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s

        See Also
        --------
        specgram
            Differs in the default overlap; in not returning the mean of the
            segment periodograms; in returning the times of the segments; and
            in plotting a colormap instead of a line.
        magnitude_spectrum
            Plots the magnitude spectrum.
        csd
            Plots the spectral density between two signals.

        Notes
        -----
        For plotting, the power is plotted as
        :math:`10\log_{10}(P_{xx})` for decibels, though *Pxx* itself
        is returned.

        References
        ----------
        Bendat & Piersol -- Random Data: Analysis and Measurement Procedures,
        John Wiley & Sons (1986)
        """
        if Fc is None:
            Fc = 0

        pxx, freqs = mlab.psd(x=x, NFFT=NFFT, Fs=Fs, detrend=detrend,
                              window=window, noverlap=noverlap, pad_to=pad_to,
                              sides=sides, scale_by_freq=scale_by_freq)
        freqs += Fc

        if scale_by_freq in (None, True):
            psd_units = 'dB/Hz'
        else:
            psd_units = 'dB'

        line = self.plot(freqs, 10 * np.log10(pxx), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Power Spectral Density (%s)' % psd_units)
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly
        intv = vmax - vmin
        logi = int(np.log10(intv))
        if logi == 0:
            logi = .1
        step = 10 * logi
        ticks = np.arange(math.floor(vmin), math.ceil(vmax) + 1, step)
        self.set_yticks(ticks)

        if return_line is None or not return_line:
            return pxx, freqs
        else:
            return pxx, freqs, line

    @_preprocess_data(replace_names=["x", "y"], label_namer="y")
    @docstring.dedent_interpd
    def csd(self, x, y, NFFT=None, Fs=None, Fc=None, detrend=None,
            window=None, noverlap=None, pad_to=None,
            sides=None, scale_by_freq=None, return_line=None, **kwargs):
        r"""
        Plot the cross-spectral density.

        The cross spectral density :math:`P_{xy}` by Welch's average
        periodogram method.  The vectors *x* and *y* are divided into
        *NFFT* length segments.  Each segment is detrended by function
        *detrend* and windowed by function *window*.  *noverlap* gives
        the length of the overlap between segments.  The product of
        the direct FFTs of *x* and *y* are averaged over each segment
        to compute :math:`P_{xy}`, with a scaling to correct for power
        loss due to windowing.

        If len(*x*) < *NFFT* or len(*y*) < *NFFT*, they will be zero
        padded to *NFFT*.

        Parameters
        ----------
        x, y : 1-D arrays or sequences
            Arrays or sequences containing the data.

        %(Spectral)s

        %(PSD)s

        noverlap : int, default: 0 (no overlap)
            The number of points of overlap between segments.

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        return_line : bool, default: False
            Whether to include the line object plotted in the returned values.

        Returns
        -------
        Pxy : 1-D array
            The values for the cross spectrum :math:`P_{xy}` before scaling
            (complex valued).

        freqs : 1-D array
            The frequencies corresponding to the elements in *Pxy*.

        line : `~matplotlib.lines.Line2D`
            The line created by this function.
            Only returned if *return_line* is True.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s

        See Also
        --------
        psd : is equivalent to setting ``y = x``.

        Notes
        -----
        For plotting, the power is plotted as
        :math:`10 \log_{10}(P_{xy})` for decibels, though :math:`P_{xy}` itself
        is returned.

        References
        ----------
        Bendat & Piersol -- Random Data: Analysis and Measurement Procedures,
        John Wiley & Sons (1986)
        """
        if Fc is None:
            Fc = 0

        pxy, freqs = mlab.csd(x=x, y=y, NFFT=NFFT, Fs=Fs, detrend=detrend,
                              window=window, noverlap=noverlap, pad_to=pad_to,
                              sides=sides, scale_by_freq=scale_by_freq)
        # pxy is complex
        freqs += Fc

        line = self.plot(freqs, 10 * np.log10(np.abs(pxy)), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Cross Spectrum Magnitude (dB)')
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly

        intv = vmax - vmin
        step = 10 * int(np.log10(intv))

        ticks = np.arange(math.floor(vmin), math.ceil(vmax) + 1, step)
        self.set_yticks(ticks)

        if return_line is None or not return_line:
            return pxy, freqs
        else:
            return pxy, freqs, line

    @_preprocess_data(replace_names=["x"])
    @docstring.dedent_interpd
    def magnitude_spectrum(self, x, Fs=None, Fc=None, window=None,
                           pad_to=None, sides=None, scale=None,
                           **kwargs):
        """
        Plot the magnitude spectrum.

        Compute the magnitude spectrum of *x*.  Data is padded to a
        length of *pad_to* and the windowing function *window* is applied to
        the signal.

        Parameters
        ----------
        x : 1-D array or sequence
            Array or sequence containing the data.

        %(Spectral)s

        %(Single_Spectrum)s

        scale : {'default', 'linear', 'dB'}
            The scaling of the values in the *spec*.  'linear' is no scaling.
            'dB' returns the values in dB scale, i.e., the dB amplitude
            (20 * log10). 'default' is 'linear'.

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        Returns
        -------
        spectrum : 1-D array
            The values for the magnitude spectrum before scaling (real valued).

        freqs : 1-D array
            The frequencies corresponding to the elements in *spectrum*.

        line : `~matplotlib.lines.Line2D`
            The line created by this function.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s

        See Also
        --------
        psd
            Plots the power spectral density.
        angle_spectrum
            Plots the angles of the corresponding frequencies.
        phase_spectrum
            Plots the phase (unwrapped angle) of the corresponding frequencies.
        specgram
            Can plot the magnitude spectrum of segments within the signal in a
            colormap.
        """
        if Fc is None:
            Fc = 0

        spec, freqs = mlab.magnitude_spectrum(x=x, Fs=Fs, window=window,
                                              pad_to=pad_to, sides=sides)
        freqs += Fc

        yunits = _api.check_getitem(
            {None: 'energy', 'default': 'energy', 'linear': 'energy',
             'dB': 'dB'},
            scale=scale)
        if yunits == 'energy':
            Z = spec
        else:  # yunits == 'dB'
            Z = 20. * np.log10(spec)

        line, = self.plot(freqs, Z, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Magnitude (%s)' % yunits)

        return spec, freqs, line

    @_preprocess_data(replace_names=["x"])
    @docstring.dedent_interpd
    def angle_spectrum(self, x, Fs=None, Fc=None, window=None,
                       pad_to=None, sides=None, **kwargs):
        """
        Plot the angle spectrum.

        Compute the angle spectrum (wrapped phase spectrum) of *x*.
        Data is padded to a length of *pad_to* and the windowing function
        *window* is applied to the signal.

        Parameters
        ----------
        x : 1-D array or sequence
            Array or sequence containing the data.

        %(Spectral)s

        %(Single_Spectrum)s

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        Returns
        -------
        spectrum : 1-D array
            The values for the angle spectrum in radians (real valued).

        freqs : 1-D array
            The frequencies corresponding to the elements in *spectrum*.

        line : `~matplotlib.lines.Line2D`
            The line created by this function.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s

        See Also
        --------
        magnitude_spectrum
            Plots the magnitudes of the corresponding frequencies.
        phase_spectrum
            Plots the unwrapped version of this function.
        specgram
            Can plot the angle spectrum of segments within the signal in a
            colormap.
        """
        if Fc is None:
            Fc = 0

        spec, freqs = mlab.angle_spectrum(x=x, Fs=Fs, window=window,
                                          pad_to=pad_to, sides=sides)
        freqs += Fc

        lines = self.plot(freqs, spec, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Angle (radians)')

        return spec, freqs, lines[0]

    @_preprocess_data(replace_names=["x"])
    @docstring.dedent_interpd
    def phase_spectrum(self, x, Fs=None, Fc=None, window=None,
                       pad_to=None, sides=None, **kwargs):
        """
        Plot the phase spectrum.

        Compute the phase spectrum (unwrapped angle spectrum) of *x*.
        Data is padded to a length of *pad_to* and the windowing function
        *window* is applied to the signal.

        Parameters
        ----------
        x : 1-D array or sequence
            Array or sequence containing the data

        %(Spectral)s

        %(Single_Spectrum)s

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        Returns
        -------
        spectrum : 1-D array
            The values for the phase spectrum in radians (real valued).

        freqs : 1-D array
            The frequencies corresponding to the elements in *spectrum*.

        line : `~matplotlib.lines.Line2D`
            The line created by this function.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s

        See Also
        --------
        magnitude_spectrum
            Plots the magnitudes of the corresponding frequencies.
        angle_spectrum
            Plots the wrapped version of this function.
        specgram
            Can plot the phase spectrum of segments within the signal in a
            colormap.
        """
        if Fc is None:
            Fc = 0

        spec, freqs = mlab.phase_spectrum(x=x, Fs=Fs, window=window,
                                          pad_to=pad_to, sides=sides)
        freqs += Fc

        lines = self.plot(freqs, spec, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Phase (radians)')

        return spec, freqs, lines[0]

    @_preprocess_data(replace_names=["x", "y"])
    @docstring.dedent_interpd
    def cohere(self, x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
               window=mlab.window_hanning, noverlap=0, pad_to=None,
               sides='default', scale_by_freq=None, **kwargs):
        r"""
        Plot the coherence between *x* and *y*.

        Plot the coherence between *x* and *y*.  Coherence is the
        normalized cross spectral density:

        .. math::

          C_{xy} = \frac{|P_{xy}|^2}{P_{xx}P_{yy}}

        Parameters
        ----------
        %(Spectral)s

        %(PSD)s

        noverlap : int, default: 0 (no overlap)
            The number of points of overlap between blocks.

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        Returns
        -------
        Cxy : 1-D array
            The coherence vector.

        freqs : 1-D array
            The frequencies for the elements in *Cxy*.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s

        References
        ----------
        Bendat & Piersol -- Random Data: Analysis and Measurement Procedures,
        John Wiley & Sons (1986)
        """
        cxy, freqs = mlab.cohere(x=x, y=y, NFFT=NFFT, Fs=Fs, detrend=detrend,
                                 window=window, noverlap=noverlap,
                                 scale_by_freq=scale_by_freq)
        freqs += Fc

        self.plot(freqs, cxy, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Coherence')
        self.grid(True)

        return cxy, freqs

    @_preprocess_data(replace_names=["x"])
    @docstring.dedent_interpd
    def specgram(self, x, NFFT=None, Fs=None, Fc=None, detrend=None,
                 window=None, noverlap=None,
                 cmap=None, xextent=None, pad_to=None, sides=None,
                 scale_by_freq=None, mode=None, scale=None,
                 vmin=None, vmax=None, **kwargs):
        """
        Plot a spectrogram.

        Compute and plot a spectrogram of data in *x*.  Data are split into
        *NFFT* length segments and the spectrum of each section is
        computed.  The windowing function *window* is applied to each
        segment, and the amount of overlap of each segment is
        specified with *noverlap*. The spectrogram is plotted as a colormap
        (using imshow).

        Parameters
        ----------
        x : 1-D array or sequence
            Array or sequence containing the data.

        %(Spectral)s

        %(PSD)s

        mode : {'default', 'psd', 'magnitude', 'angle', 'phase'}
            What sort of spectrum to use.  Default is 'psd', which takes the
            power spectral density.  'magnitude' returns the magnitude
            spectrum.  'angle' returns the phase spectrum without unwrapping.
            'phase' returns the phase spectrum with unwrapping.

        noverlap : int, default: 128
            The number of points of overlap between blocks.

        scale : {'default', 'linear', 'dB'}
            The scaling of the values in the *spec*.  'linear' is no scaling.
            'dB' returns the values in dB scale.  When *mode* is 'psd',
            this is dB power (10 * log10).  Otherwise this is dB amplitude
            (20 * log10). 'default' is 'dB' if *mode* is 'psd' or
            'magnitude' and 'linear' otherwise.  This must be 'linear'
            if *mode* is 'angle' or 'phase'.

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        cmap : `.Colormap`, default: :rc:`image.cmap`

        xextent : *None* or (xmin, xmax)
            The image extent along the x-axis. The default sets *xmin* to the
            left border of the first bin (*spectrum* column) and *xmax* to the
            right border of the last bin. Note that for *noverlap>0* the width
            of the bins is smaller than those of the segments.

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Additional keyword arguments are passed on to `~.axes.Axes.imshow`
            which makes the specgram image. The origin keyword argument
            is not supported.

        Returns
        -------
        spectrum : 2D array
            Columns are the periodograms of successive segments.

        freqs : 1-D array
            The frequencies corresponding to the rows in *spectrum*.

        t : 1-D array
            The times corresponding to midpoints of segments (i.e., the columns
            in *spectrum*).

        im : `.AxesImage`
            The image created by imshow containing the spectrogram.

        See Also
        --------
        psd
            Differs in the default overlap; in returning the mean of the
            segment periodograms; in not returning times; and in generating a
            line plot instead of colormap.
        magnitude_spectrum
            A single spectrum, similar to having a single segment when *mode*
            is 'magnitude'. Plots a line instead of a colormap.
        angle_spectrum
            A single spectrum, similar to having a single segment when *mode*
            is 'angle'. Plots a line instead of a colormap.
        phase_spectrum
            A single spectrum, similar to having a single segment when *mode*
            is 'phase'. Plots a line instead of a colormap.

        Notes
        -----
        The parameters *detrend* and *scale_by_freq* do only apply when *mode*
        is set to 'psd'.
        """
        if NFFT is None:
            NFFT = 256  # same default as in mlab.specgram()
        if Fc is None:
            Fc = 0  # same default as in mlab._spectral_helper()
        if noverlap is None:
            noverlap = 128  # same default as in mlab.specgram()
        if Fs is None:
            Fs = 2  # same default as in mlab._spectral_helper()

        if mode == 'complex':
            raise ValueError('Cannot plot a complex specgram')

        if scale is None or scale == 'default':
            if mode in ['angle', 'phase']:
                scale = 'linear'
            else:
                scale = 'dB'
        elif mode in ['angle', 'phase'] and scale == 'dB':
            raise ValueError('Cannot use dB scale with angle or phase mode')

        spec, freqs, t = mlab.specgram(x=x, NFFT=NFFT, Fs=Fs,
                                       detrend=detrend, window=window,
                                       noverlap=noverlap, pad_to=pad_to,
                                       sides=sides,
                                       scale_by_freq=scale_by_freq,
                                       mode=mode)

        if scale == 'linear':
            Z = spec
        elif scale == 'dB':
            if mode is None or mode == 'default' or mode == 'psd':
                Z = 10. * np.log10(spec)
            else:
                Z = 20. * np.log10(spec)
        else:
            raise ValueError('Unknown scale %s', scale)

        Z = np.flipud(Z)

        if xextent is None:
            # padding is needed for first and last segment:
            pad_xextent = (NFFT-noverlap) / Fs / 2
            xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
        xmin, xmax = xextent
        freqs += Fc
        extent = xmin, xmax, freqs[0], freqs[-1]

        if 'origin' in kwargs:
            raise TypeError("specgram() got an unexpected keyword argument "
                            "'origin'")

        im = self.imshow(Z, cmap, extent=extent, vmin=vmin, vmax=vmax,
                         origin='upper', **kwargs)
        self.axis('auto')

        return spec, freqs, t, im

    @docstring.dedent_interpd
    def spy(self, Z, precision=0, marker=None, markersize=None,
            aspect='equal', origin="upper", **kwargs):
        """
        Plot the sparsity pattern of a 2D array.

        This visualizes the non-zero values of the array.

        Two plotting styles are available: image and marker. Both
        are available for full arrays, but only the marker style
        works for `scipy.sparse.spmatrix` instances.

        **Image style**

        If *marker* and *markersize* are *None*, `~.Axes.imshow` is used. Any
        extra remaining keyword arguments are passed to this method.

        **Marker style**

        If *Z* is a `scipy.sparse.spmatrix` or *marker* or *markersize* are
        *None*, a `.Line2D` object will be returned with the value of marker
        determining the marker type, and any remaining keyword arguments
        passed to `~.Axes.plot`.

        Parameters
        ----------
        Z : (M, N) array-like
            The array to be plotted.

        precision : float or 'present', default: 0
            If *precision* is 0, any non-zero value will be plotted. Otherwise,
            values of :math:`|Z| > precision` will be plotted.

            For `scipy.sparse.spmatrix` instances, you can also
            pass 'present'. In this case any value present in the array
            will be plotted, even if it is identically zero.

        aspect : {'equal', 'auto', None} or float, default: 'equal'
            The aspect ratio of the Axes.  This parameter is particularly
            relevant for images since it determines whether data pixels are
            square.

            This parameter is a shortcut for explicitly calling
            `.Axes.set_aspect`. See there for further details.

            - 'equal': Ensures an aspect ratio of 1. Pixels will be square.
            - 'auto': The Axes is kept fixed and the aspect is adjusted so
              that the data fit in the Axes. In general, this will result in
              non-square pixels.
            - *None*: Use :rc:`image.aspect`.

        origin : {'upper', 'lower'}, default: :rc:`image.origin`
            Place the [0, 0] index of the array in the upper left or lower left
            corner of the Axes. The convention 'upper' is typically used for
            matrices and images.

        Returns
        -------
        `~matplotlib.image.AxesImage` or `.Line2D`
            The return type depends on the plotting style (see above).

        Other Parameters
        ----------------
        **kwargs
            The supported additional parameters depend on the plotting style.

            For the image style, you can pass the following additional
            parameters of `~.Axes.imshow`:

            - *cmap*
            - *alpha*
            - *url*
            - any `.Artist` properties (passed on to the `.AxesImage`)

            For the marker style, you can pass any `.Line2D` property except
            for *linestyle*:

            %(Line2D:kwdoc)s
        """
        if marker is None and markersize is None and hasattr(Z, 'tocoo'):
            marker = 's'
        _api.check_in_list(["upper", "lower"], origin=origin)
        if marker is None and markersize is None:
            Z = np.asarray(Z)
            mask = np.abs(Z) > precision

            if 'cmap' not in kwargs:
                kwargs['cmap'] = mcolors.ListedColormap(['w', 'k'],
                                                        name='binary')
            if 'interpolation' in kwargs:
                raise TypeError(
                    "spy() got an unexpected keyword argument 'interpolation'")
            if 'norm' not in kwargs:
                kwargs['norm'] = mcolors.NoNorm()
            ret = self.imshow(mask, interpolation='nearest',
                              aspect=aspect, origin=origin,
                              **kwargs)
        else:
            if hasattr(Z, 'tocoo'):
                c = Z.tocoo()
                if precision == 'present':
                    y = c.row
                    x = c.col
                else:
                    nonzero = np.abs(c.data) > precision
                    y = c.row[nonzero]
                    x = c.col[nonzero]
            else:
                Z = np.asarray(Z)
                nonzero = np.abs(Z) > precision
                y, x = np.nonzero(nonzero)
            if marker is None:
                marker = 's'
            if markersize is None:
                markersize = 10
            if 'linestyle' in kwargs:
                raise TypeError(
                    "spy() got an unexpected keyword argument 'linestyle'")
            ret = mlines.Line2D(
                x, y, linestyle='None', marker=marker, markersize=markersize,
                **kwargs)
            self.add_line(ret)
            nr, nc = Z.shape
            self.set_xlim(-0.5, nc - 0.5)
            if origin == "upper":
                self.set_ylim(nr - 0.5, -0.5)
            else:
                self.set_ylim(-0.5, nr - 0.5)
            self.set_aspect(aspect)
        self.title.set_y(1.05)
        if origin == "upper":
            self.xaxis.tick_top()
        else:
            self.xaxis.tick_bottom()
        self.xaxis.set_ticks_position('both')
        self.xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        self.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        return ret

    def matshow(self, Z, **kwargs):
        """
        Plot the values of a 2D matrix or array as color-coded image.

        The matrix will be shown the way it would be printed, with the first
        row at the top.  Row and column numbering is zero-based.

        Parameters
        ----------
        Z : (M, N) array-like
            The matrix to be displayed.

        Returns
        -------
        `~matplotlib.image.AxesImage`

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.axes.Axes.imshow` arguments

        See Also
        --------
        imshow : More general function to plot data on a 2D regular raster.

        Notes
        -----
        This is just a convenience function wrapping `.imshow` to set useful
        defaults for displaying a matrix. In particular:

        - Set ``origin='upper'``.
        - Set ``interpolation='nearest'``.
        - Set ``aspect='equal'``.
        - Ticks are placed to the left and above.
        - Ticks are formatted to show integer indices.

        """
        Z = np.asanyarray(Z)
        kw = {'origin': 'upper',
              'interpolation': 'nearest',
              'aspect': 'equal',          # (already the imshow default)
              **kwargs}
        im = self.imshow(Z, **kw)
        self.title.set_y(1.05)
        self.xaxis.tick_top()
        self.xaxis.set_ticks_position('both')
        self.xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        self.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        return im

    @_preprocess_data(replace_names=["dataset"])
    def violinplot(self, dataset, positions=None, vert=True, widths=0.5,
                   showmeans=False, showextrema=True, showmedians=False,
                   quantiles=None, points=100, bw_method=None):
        """
        Make a violin plot.

        Make a violin plot for each column of *dataset* or each vector in
        sequence *dataset*.  Each filled area extends to represent the
        entire data range, with optional lines at the mean, the median,
        the minimum, the maximum, and user-specified quantiles.

        Parameters
        ----------
        dataset : Array or a sequence of vectors.
          The input data.

        positions : array-like, default: [1, 2, ..., n]
          The positions of the violins. The ticks and limits are
          automatically set to match the positions.

        vert : bool, default: True.
          If true, creates a vertical violin plot.
          Otherwise, creates a horizontal violin plot.

        widths : array-like, default: 0.5
          Either a scalar or a vector that sets the maximal width of
          each violin. The default is 0.5, which uses about half of the
          available horizontal space.

        showmeans : bool, default: False
          If `True`, will toggle rendering of the means.

        showextrema : bool, default: True
          If `True`, will toggle rendering of the extrema.

        showmedians : bool, default: False
          If `True`, will toggle rendering of the medians.

        quantiles : array-like, default: None
          If not None, set a list of floats in interval [0, 1] for each violin,
          which stands for the quantiles that will be rendered for that
          violin.

        points : int, default: 100
          Defines the number of points to evaluate each of the
          gaussian kernel density estimations at.

        bw_method : str, scalar or callable, optional
          The method used to calculate the estimator bandwidth.  This can be
          'scott', 'silverman', a scalar constant or a callable.  If a
          scalar, this will be used directly as `kde.factor`.  If a
          callable, it should take a `GaussianKDE` instance as its only
          parameter and return a scalar. If None (default), 'scott' is used.

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        Returns
        -------
        dict
          A dictionary mapping each component of the violinplot to a
          list of the corresponding collection instances created. The
          dictionary has the following keys:

          - ``bodies``: A list of the `~.collections.PolyCollection`
            instances containing the filled area of each violin.

          - ``cmeans``: A `~.collections.LineCollection` instance that marks
            the mean values of each of the violin's distribution.

          - ``cmins``: A `~.collections.LineCollection` instance that marks
            the bottom of each violin's distribution.

          - ``cmaxes``: A `~.collections.LineCollection` instance that marks
            the top of each violin's distribution.

          - ``cbars``: A `~.collections.LineCollection` instance that marks
            the centers of each violin's distribution.

          - ``cmedians``: A `~.collections.LineCollection` instance that
            marks the median values of each of the violin's distribution.

          - ``cquantiles``: A `~.collections.LineCollection` instance created
            to identify the quantile values of each of the violin's
            distribution.

        """

        def _kde_method(X, coords):
            if hasattr(X, 'values'):  # support pandas.Series
                X = X.values
            # fallback gracefully if the vector contains only one value
            if np.all(X[0] == X):
                return (X[0] == coords).astype(float)
            kde = mlab.GaussianKDE(X, bw_method)
            return kde.evaluate(coords)

        vpstats = cbook.violin_stats(dataset, _kde_method, points=points,
                                     quantiles=quantiles)
        return self.violin(vpstats, positions=positions, vert=vert,
                           widths=widths, showmeans=showmeans,
                           showextrema=showextrema, showmedians=showmedians)

    def violin(self, vpstats, positions=None, vert=True, widths=0.5,
               showmeans=False, showextrema=True, showmedians=False):
        """
        Drawing function for violin plots.

        Draw a violin plot for each column of *vpstats*. Each filled area
        extends to represent the entire data range, with optional lines at the
        mean, the median, the minimum, the maximum, and the quantiles values.

        Parameters
        ----------
        vpstats : list of dicts
          A list of dictionaries containing stats for each violin plot.
          Required keys are:

          - ``coords``: A list of scalars containing the coordinates that
            the violin's kernel density estimate were evaluated at.

          - ``vals``: A list of scalars containing the values of the
            kernel density estimate at each of the coordinates given
            in *coords*.

          - ``mean``: The mean value for this violin's dataset.

          - ``median``: The median value for this violin's dataset.

          - ``min``: The minimum value for this violin's dataset.

          - ``max``: The maximum value for this violin's dataset.

          Optional keys are:

          - ``quantiles``: A list of scalars containing the quantile values
            for this violin's dataset.

        positions : array-like, default: [1, 2, ..., n]
          The positions of the violins. The ticks and limits are
          automatically set to match the positions.

        vert : bool, default: True.
          If true, plots the violins vertically.
          Otherwise, plots the violins horizontally.

        widths : array-like, default: 0.5
          Either a scalar or a vector that sets the maximal width of
          each violin. The default is 0.5, which uses about half of the
          available horizontal space.

        showmeans : bool, default: False
          If true, will toggle rendering of the means.

        showextrema : bool, default: True
          If true, will toggle rendering of the extrema.

        showmedians : bool, default: False
          If true, will toggle rendering of the medians.

        Returns
        -------
        dict
          A dictionary mapping each component of the violinplot to a
          list of the corresponding collection instances created. The
          dictionary has the following keys:

          - ``bodies``: A list of the `~.collections.PolyCollection`
            instances containing the filled area of each violin.

          - ``cmeans``: A `~.collections.LineCollection` instance that marks
            the mean values of each of the violin's distribution.

          - ``cmins``: A `~.collections.LineCollection` instance that marks
            the bottom of each violin's distribution.

          - ``cmaxes``: A `~.collections.LineCollection` instance that marks
            the top of each violin's distribution.

          - ``cbars``: A `~.collections.LineCollection` instance that marks
            the centers of each violin's distribution.

          - ``cmedians``: A `~.collections.LineCollection` instance that
            marks the median values of each of the violin's distribution.

          - ``cquantiles``: A `~.collections.LineCollection` instance created
            to identify the quantiles values of each of the violin's
            distribution.
        """

        # Statistical quantities to be plotted on the violins
        means = []
        mins = []
        maxes = []
        medians = []
        quantiles = []

        qlens = []  # Number of quantiles in each dataset.

        artists = {}  # Collections to be returned

        N = len(vpstats)
        datashape_message = ("List of violinplot statistics and `{0}` "
                             "values must have the same length")

        # Validate positions
        if positions is None:
            positions = range(1, N + 1)
        elif len(positions) != N:
            raise ValueError(datashape_message.format("positions"))

        # Validate widths
        if np.isscalar(widths):
            widths = [widths] * N
        elif len(widths) != N:
            raise ValueError(datashape_message.format("widths"))

        # Calculate ranges for statistics lines (shape (2, N)).
        line_ends = [[-0.25], [0.25]] * np.array(widths) + positions

        # Colors.
        if rcParams['_internal.classic_mode']:
            fillcolor = 'y'
            linecolor = 'r'
        else:
            fillcolor = linecolor = self._get_lines.get_next_color()

        # Check whether we are rendering vertically or horizontally
        if vert:
            fill = self.fill_betweenx
            perp_lines = functools.partial(self.hlines, colors=linecolor)
            par_lines = functools.partial(self.vlines, colors=linecolor)
        else:
            fill = self.fill_between
            perp_lines = functools.partial(self.vlines, colors=linecolor)
            par_lines = functools.partial(self.hlines, colors=linecolor)

        # Render violins
        bodies = []
        for stats, pos, width in zip(vpstats, positions, widths):
            # The 0.5 factor reflects the fact that we plot from v-p to v+p.
            vals = np.array(stats['vals'])
            vals = 0.5 * width * vals / vals.max()
            bodies += [fill(stats['coords'], -vals + pos, vals + pos,
                            facecolor=fillcolor, alpha=0.3)]
            means.append(stats['mean'])
            mins.append(stats['min'])
            maxes.append(stats['max'])
            medians.append(stats['median'])
            q = stats.get('quantiles')  # a list of floats, or None
            if q is None:
                q = []
            quantiles.extend(q)
            qlens.append(len(q))
        artists['bodies'] = bodies

        if showmeans:  # Render means
            artists['cmeans'] = perp_lines(means, *line_ends)
        if showextrema:  # Render extrema
            artists['cmaxes'] = perp_lines(maxes, *line_ends)
            artists['cmins'] = perp_lines(mins, *line_ends)
            artists['cbars'] = par_lines(positions, mins, maxes)
        if showmedians:  # Render medians
            artists['cmedians'] = perp_lines(medians, *line_ends)
        if quantiles:  # Render quantiles: each width is repeated qlen times.
            artists['cquantiles'] = perp_lines(
                quantiles, *np.repeat(line_ends, qlens, axis=1))

        return artists

    # Methods that are entirely implemented in other modules.

    table = mtable.table

    # args can by either Y or y1, y2, ... and all should be replaced
    stackplot = _preprocess_data()(mstack.stackplot)

    streamplot = _preprocess_data(
        replace_names=["x", "y", "u", "v", "start_points"])(mstream.streamplot)

    tricontour = mtri.tricontour
    tricontourf = mtri.tricontourf
    tripcolor = mtri.tripcolor
    triplot = mtri.triplot
