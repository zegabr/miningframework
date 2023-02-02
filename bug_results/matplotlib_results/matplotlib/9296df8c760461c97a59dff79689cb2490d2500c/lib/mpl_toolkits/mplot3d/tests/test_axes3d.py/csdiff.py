"""
axes3d.py, original mplot3d version by John Porter
Created: 23 Sep 2005

Parts fixed by Reinier Heeres <reinier@heeres.eu>
Minor additions by Ben Axelrod <baxelrod@coroware.com>
Significant updates and revisions by Ben Root <ben.v.root@gmail.com>

Module containing Axes3D, an object which can plot 3D objects on a
2D matplotlib figure.
"""

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
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation

from . import art3d
from . import proj3d
from . import axis3d


@_docstring.interpd
@_api.define_aliases({
    "xlim": ["xlim3d"], "ylim": ["ylim3d"], "zlim": ["zlim3d"]})
class Axes3D(Axes):
    """
    3D Axes object.

    .. note::

        As a user, you do not instantiate Axes directly, but use Axes creation
        methods instead; e.g. from `.pyplot` or `.Figure`:
        `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` or `.Figure.add_axes`.
    """
    name = '3d'

    _axis_names = ("x", "y", "z")
    Axes._shared_axes["z"] = cbook.Grouper()

    dist = _api.deprecate_privatize_attribute("3.6")
    vvec = _api.deprecate_privatize_attribute("3.7")
    eye = _api.deprecate_privatize_attribute("3.7")
    sx = _api.deprecate_privatize_attribute("3.7")
    sy = _api.deprecate_privatize_attribute("3.7")

    def __init__(
            self, fig, rect=None, *args,
            elev=30, azim=-60, roll=0, sharez=None, proj_type='persp',
            box_aspect=None, computed_zorder=True, focal_length=None,
            **kwargs):
        """
        Parameters
        ----------
        fig : Figure
            The parent figure.
        rect : tuple (left, bottom, width, height), default: None.
            The ``(left, bottom, width, height)`` axes position.
        elev : float, default: 30
            The elevation angle in degrees rotates the camera above and below
            the x-y plane, with a positive angle corresponding to a location
            above the plane.
        azim : float, default: -60
            The azimuthal angle in degrees rotates the camera about the z axis,
            with a positive angle corresponding to a right-handed rotation. In
            other words, a positive azimuth rotates the camera about the origin
            from its location along the +x axis towards the +y axis.
        roll : float, default: 0
            The roll angle in degrees rotates the camera about the viewing
            axis. A positive angle spins the camera clockwise, causing the
            scene to rotate counter-clockwise.
        sharez : Axes3D, optional
            Other Axes to share z-limits with.
        proj_type : {'persp', 'ortho'}
            The projection type, default 'persp'.
        box_aspect : 3-tuple of floats, default: None
            Changes the physical dimensions of the Axes3D, such that the ratio
            of the axis lengths in display units is x:y:z.
            If None, defaults to 4:4:3
        computed_zorder : bool, default: True
            If True, the draw order is computed based on the average position
            of the `.Artist`\\s along the view direction.
            Set to False if you want to manually control the order in which
            Artists are drawn on top of each other using their *zorder*
            attribute. This can be used for fine-tuning if the automatic order
            does not produce the desired result. Note however, that a manual
            zorder will only be correct for a limited view angle. If the figure
            is rotated by the user, it will look wrong from certain angles.
        focal_length : float, default: None
            For a projection type of 'persp', the focal length of the virtual
            camera. Must be > 0. If None, defaults to 1.
            For a projection type of 'ortho', must be set to either None
            or infinity (numpy.inf). If None, defaults to infinity.
            The focal length can be computed from a desired Field Of View via
            the equation: focal_length = 1/tan(FOV/2)

        **kwargs
            Other optional keyword arguments:

            %(Axes3D:kwdoc)s
        """

        if rect is None:
            rect = [0.0, 0.0, 1.0, 1.0]

        self.initial_azim = azim
        self.initial_elev = elev
        self.initial_roll = roll
        self.set_proj_type(proj_type, focal_length)
        self.computed_zorder = computed_zorder

        self.xy_viewLim = Bbox.unit()
        self.zz_viewLim = Bbox.unit()
        self.xy_dataLim = Bbox.unit()
        # z-limits are encoded in the x-component of the Bbox, y is un-used
        self.zz_dataLim = Bbox.unit()

        # inhibit autoscale_view until the axes are defined
        # they can't be defined until Axes.__init__ has been called
        self.view_init(self.initial_elev, self.initial_azim, self.initial_roll)

        self._sharez = sharez
        if sharez is not None:
            self._shared_axes["z"].join(self, sharez)
            self._adjustable = 'datalim'

        if kwargs.pop('auto_add_to_figure', False):
            raise AttributeError(
                'auto_add_to_figure is no longer supported for Axes3D. '
                'Use fig.add_axes(ax) instead.'
            )

        super().__init__(
            fig, rect, frameon=True, box_aspect=box_aspect, *args, **kwargs
        )
        # Disable drawing of axes by base class
        super().set_axis_off()
        # Enable drawing of axes by Axes3D class
        self.set_axis_on()
        self.M = None

        # func used to format z -- fall back on major formatters
        self.fmt_zdata = None

        self.mouse_init()
        self.figure.canvas.callbacks._connect_picklable(
            'motion_notify_event', self._on_move)
        self.figure.canvas.callbacks._connect_picklable(
            'button_press_event', self._button_press)
        self.figure.canvas.callbacks._connect_picklable(
            'button_release_event', self._button_release)
        self.set_top_view()

        self.patch.set_linewidth(0)
        # Calculate the pseudo-data width and height
        pseudo_bbox = self.transLimits.inverted().transform([(0, 0), (1, 1)])
        self._pseudo_w, self._pseudo_h = pseudo_bbox[1] - pseudo_bbox[0]

        # mplot3d currently manages its own spines and needs these turned off
        # for bounding box calculations
        self.spines[:].set_visible(False)

    def set_axis_off(self):
        self._axis3don = False
        self.stale = True

    def set_axis_on(self):
        self._axis3don = True
        self.stale = True

    def convert_zunits(self, z):
        """
        For artists in an Axes, if the zaxis has units support,
        convert *z* using zaxis unit type
        """
        return self.zaxis.convert_units(z)

    def set_top_view(self):
        # this happens to be the right view for the viewing coordinates
        # moved up and to the left slightly to fit labels and axes
        xdwl = 0.95 / self._dist
        xdw = 0.9 / self._dist
        ydwl = 0.95 / self._dist
        ydw = 0.9 / self._dist
        # Set the viewing pane.
        self.viewLim.intervalx = (-xdwl, xdw)
        self.viewLim.intervaly = (-ydwl, ydw)
        self.stale = True

    def _init_axis(self):
        """Init 3D axes; overrides creation of regular X/Y axes."""
        self.xaxis = axis3d.XAxis(self)
        self.yaxis = axis3d.YAxis(self)
        self.zaxis = axis3d.ZAxis(self)

    def get_zaxis(self):
        """Return the ``ZAxis`` (`~.axis3d.Axis`) instance."""
        return self.zaxis

    get_zgridlines = _axis_method_wrapper("zaxis", "get_gridlines")
    get_zticklines = _axis_method_wrapper("zaxis", "get_ticklines")

    w_xaxis = _api.deprecated("3.1", alternative="xaxis", removal="3.8")(
        property(lambda self: self.xaxis))
    w_yaxis = _api.deprecated("3.1", alternative="yaxis", removal="3.8")(
        property(lambda self: self.yaxis))
    w_zaxis = _api.deprecated("3.1", alternative="zaxis", removal="3.8")(
        property(lambda self: self.zaxis))

    @_api.deprecated("3.7")
    def unit_cube(self, vals=None):
        return self._unit_cube(vals)

    def _unit_cube(self, vals=None):
        minx, maxx, miny, maxy, minz, maxz = vals or self.get_w_lims()
        return [(minx, miny, minz),
                (maxx, miny, minz),
                (maxx, maxy, minz),
                (minx, maxy, minz),
                (minx, miny, maxz),
                (maxx, miny, maxz),
                (maxx, maxy, maxz),
                (minx, maxy, maxz)]

    @_api.deprecated("3.7")
    def tunit_cube(self, vals=None, M=None):
        return self._tunit_cube(vals, M)

    def _tunit_cube(self, vals=None, M=None):
        if M is None:
            M = self.M
        xyzs = self._unit_cube(vals)
        tcube = proj3d._proj_points(xyzs, M)
        return tcube

    @_api.deprecated("3.7")
    def tunit_edges(self, vals=None, M=None):
        return self._tunit_edges(vals, M)

    def _tunit_edges(self, vals=None, M=None):
        tc = self._tunit_cube(vals, M)
        edges = [(tc[0], tc[1]),
                 (tc[1], tc[2]),
                 (tc[2], tc[3]),
                 (tc[3], tc[0]),

                 (tc[0], tc[4]),
                 (tc[1], tc[5]),
                 (tc[2], tc[6]),
                 (tc[3], tc[7]),

                 (tc[4], tc[5]),
                 (tc[5], tc[6]),
                 (tc[6], tc[7]),
                 (tc[7], tc[4])]
        return edges

    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
        """
        Set the aspect ratios.

        Parameters
        ----------
        aspect : {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}
            Possible values:

            =========   ==================================================
            value       description
            =========   ==================================================
            'auto'      automatic; fill the position rectangle with data.
            'equal'     adapt all the axes to have equal aspect ratios.
            'equalxy'   adapt the x and y axes to have equal aspect ratios.
            'equalxz'   adapt the x and z axes to have equal aspect ratios.
            'equalyz'   adapt the y and z axes to have equal aspect ratios.
            =========   ==================================================

        adjustable : None or {'box', 'datalim'}, optional
            If not *None*, this defines which parameter will be adjusted to
            meet the required aspect. See `.set_adjustable` for further
            details.

        anchor : None or str or 2-tuple of float, optional
            If not *None*, this defines where the Axes will be drawn if there
            is extra space due to aspect constraints. The most common way to
            specify the anchor are abbreviations of cardinal directions:

            =====   =====================
            value   description
            =====   =====================
            'C'     centered
            'SW'    lower left corner
            'S'     middle of bottom edge
            'SE'    lower right corner
            etc.
            =====   =====================

            See `~.Axes.set_anchor` for further details.

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        mpl_toolkits.mplot3d.axes3d.Axes3D.set_box_aspect
        """
        _api.check_in_list(('auto', 'equal', 'equalxy', 'equalyz', 'equalxz'),
                           aspect=aspect)
        super().set_aspect(
            aspect='auto', adjustable=adjustable, anchor=anchor, share=share)
        self._aspect = aspect

        if aspect in ('equal', 'equalxy', 'equalxz', 'equalyz'):
            ax_indices = self._equal_aspect_axis_indices(aspect)

            view_intervals = np.array([self.xaxis.get_view_interval(),
                                       self.yaxis.get_view_interval(),
                                       self.zaxis.get_view_interval()])
            ptp = np.ptp(view_intervals, axis=1)
            if self._adjustable == 'datalim':
                mean = np.mean(view_intervals, axis=1)
                delta = max(ptp[ax_indices])
                scale = self._box_aspect[ptp == delta][0]
                deltas = delta * self._box_aspect / scale

                for i, set_lim in enumerate((self.set_xlim3d,
                                             self.set_ylim3d,
                                             self.set_zlim3d)):
                    if i in ax_indices:
                        set_lim(mean[i] - deltas[i]/2., mean[i] + deltas[i]/2.)
            else:  # 'box'
                # Change the box aspect such that the ratio of the length of
                # the unmodified axis to the length of the diagonal
                # perpendicular to it remains unchanged.
                box_aspect = np.array(self._box_aspect)
                box_aspect[ax_indices] = ptp[ax_indices]
                remaining_ax_indices = {0, 1, 2}.difference(ax_indices)
                if remaining_ax_indices:
                    remaining = remaining_ax_indices.pop()
                    old_diag = np.linalg.norm(self._box_aspect[ax_indices])
                    new_diag = np.linalg.norm(box_aspect[ax_indices])
                    box_aspect[remaining] *= new_diag / old_diag
                self.set_box_aspect(box_aspect)

    def _equal_aspect_axis_indices(self, aspect):
        """
        Get the indices for which of the x, y, z axes are constrained to have
        equal aspect ratios.

        Parameters
        ----------
        aspect : {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}
            See descriptions in docstring for `.set_aspect()`.
        """
        ax_indices = []  # aspect == 'auto'
        if aspect == 'equal':
            ax_indices = [0, 1, 2]
        elif aspect == 'equalxy':
            ax_indices = [0, 1]
        elif aspect == 'equalxz':
            ax_indices = [0, 2]
        elif aspect == 'equalyz':
            ax_indices = [1, 2]
        return ax_indices

    def set_box_aspect(self, aspect, *, zoom=1):
        """
        Set the Axes box aspect.

        The box aspect is the ratio of height to width in display
        units for each face of the box when viewed perpendicular to
        that face.  This is not to be confused with the data aspect (see
        `~.Axes3D.set_aspect`). The default ratios are 4:4:3 (x:y:z).

        To simulate having equal aspect in data space, set the box
        aspect to match your data range in each dimension.

        *zoom* controls the overall size of the Axes3D in the figure.

        Parameters
        ----------
        aspect : 3-tuple of floats or None
            Changes the physical dimensions of the Axes3D, such that the ratio
            of the axis lengths in display units is x:y:z.
            If None, defaults to (4, 4, 3).

        zoom : float, default: 1
            Control overall size of the Axes3D in the figure. Must be > 0.
        """
        if zoom <= 0:
            raise ValueError(f'Argument zoom = {zoom} must be > 0')

        if aspect is None:
            aspect = np.asarray((4, 4, 3), dtype=float)
        else:
            aspect = np.asarray(aspect, dtype=float)
            _api.check_shape((3,), aspect=aspect)
        # default scale tuned to match the mpl32 appearance.
        aspect *= 1.8294640721620434 * zoom / np.linalg.norm(aspect)

        self._box_aspect = aspect
        self.stale = True

    def apply_aspect(self, position=None):
        if position is None:
            position = self.get_position(original=True)

        # in the superclass, we would go through and actually deal with axis
        # scales and box/datalim. Those are all irrelevant - all we need to do
        # is make sure our coordinate system is square.
        trans = self.get_figure().transSubfigure
        bb = mtransforms.Bbox.unit().transformed(trans)
        # this is the physical aspect of the panel (or figure):
        fig_aspect = bb.height / bb.width

        box_aspect = 1
        pb = position.frozen()
        pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect)
        self._set_position(pb1.anchored(self.get_anchor(), pb), 'active')

    @martist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        self._unstale_viewLim()

        # draw the background patch
        self.patch.draw(renderer)
        self._frameon = False

        # first, set the aspect
        # this is duplicated from `axes._base._AxesBase.draw`
        # but must be called before any of the artist are drawn as
        # it adjusts the view limits and the size of the bounding box
        # of the Axes
        locator = self.get_axes_locator()
        self.apply_aspect(locator(self, renderer) if locator else None)

        # add the projection matrix to the renderer
        self.M = self.get_proj()

        collections_and_patches = (
            artist for artist in self._children
            if isinstance(artist, (mcoll.Collection, mpatches.Patch))
            and artist.get_visible())
        if self.computed_zorder:
            # Calculate projection of collections and patches and zorder
            # them. Make sure they are drawn above the grids.
            zorder_offset = max(axis.get_zorder()
                                for axis in self._axis_map.values()) + 1
            collection_zorder = patch_zorder = zorder_offset

            for artist in sorted(collections_and_patches,
                                 key=lambda artist: artist.do_3d_projection(),
                                 reverse=True):
                if isinstance(artist, mcoll.Collection):
                    artist.zorder = collection_zorder
                    collection_zorder += 1
                elif isinstance(artist, mpatches.Patch):
                    artist.zorder = patch_zorder
                    patch_zorder += 1
        else:
            for artist in collections_and_patches:
                artist.do_3d_projection()

        if self._axis3don:
            # Draw panes first
            for axis in self._axis_map.values():
                axis.draw_pane(renderer)
            # Then axes
            for axis in self._axis_map.values():
                axis.draw(renderer)

        # Then rest
        super().draw(renderer)

    def get_axis_position(self):
        vals = self.get_w_lims()
        tc = self._tunit_cube(vals, self.M)
        xhigh = tc[1][2] > tc[2][2]
        yhigh = tc[3][2] > tc[2][2]
        zhigh = tc[0][2] > tc[2][2]
        return xhigh, yhigh, zhigh

    def update_datalim(self, xys, **kwargs):
        """
        Not implemented in `~mpl_toolkits.mplot3d.axes3d.Axes3D`.
        """
        pass

    get_autoscalez_on = _axis_method_wrapper("zaxis", "_get_autoscale_on")
    set_autoscalez_on = _axis_method_wrapper("zaxis", "_set_autoscale_on")

    def set_zmargin(self, m):
        """
        Set padding of Z data limits prior to autoscaling.

        *m* times the data interval will be added to each end of that interval
        before it is used in autoscaling.  If *m* is negative, this will clip
        the data range instead of expanding it.

        For example, if your data is in the range [0, 2], a margin of 0.1 will
        result in a range [-0.2, 2.2]; a margin of -0.1 will result in a range
        of [0.2, 1.8].

        Parameters
        ----------
        m : float greater than -0.5
        """
        if m <= -0.5:
            raise ValueError("margin must be greater than -0.5")
        self._zmargin = m
        self._request_autoscale_view("z")
        self.stale = True

    def margins(self, *margins, x=None, y=None, z=None, tight=True):
        """
        Set or retrieve autoscaling margins.

        See `.Axes.margins` for full documentation.  Because this function
        applies to 3D Axes, it also takes a *z* argument, and returns
        ``(xmargin, ymargin, zmargin)``.
        """
        if margins and (x is not None or y is not None or z is not None):
            raise TypeError('Cannot pass both positional and keyword '
                            'arguments for x, y, and/or z.')
        elif len(margins) == 1:
            x = y = z = margins[0]
        elif len(margins) == 3:
            x, y, z = margins
        elif margins:
            raise TypeError('Must pass a single positional argument for all '
                            'margins, or one for each margin (x, y, z).')

        if x is None and y is None and z is None:
            if tight is not True:
                _api.warn_external(f'ignoring tight={tight!r} in get mode')
            return self._xmargin, self._ymargin, self._zmargin

        if x is not None:
            self.set_xmargin(x)
        if y is not None:
            self.set_ymargin(y)
        if z is not None:
            self.set_zmargin(z)

        self.autoscale_view(
            tight=tight, scalex=(x is not None), scaley=(y is not None),
            scalez=(z is not None)
        )

    def autoscale(self, enable=True, axis='both', tight=None):
        """
        Convenience method for simple axis view autoscaling.

        See `.Axes.autoscale` for full documentation.  Because this function
        applies to 3D Axes, *axis* can also be set to 'z', and setting *axis*
        to 'both' autoscales all three axes.
        """
        if enable is None:
            scalex = True
            scaley = True
            scalez = True
        else:
            if axis in ['x', 'both']:
                self.set_autoscalex_on(bool(enable))
                scalex = self.get_autoscalex_on()
            else:
                scalex = False
            if axis in ['y', 'both']:
                self.set_autoscaley_on(bool(enable))
                scaley = self.get_autoscaley_on()
            else:
                scaley = False
            if axis in ['z', 'both']:
                self.set_autoscalez_on(bool(enable))
                scalez = self.get_autoscalez_on()
            else:
                scalez = False
        if scalex:
            self._request_autoscale_view("x", tight=tight)
        if scaley:
            self._request_autoscale_view("y", tight=tight)
        if scalez:
            self._request_autoscale_view("z", tight=tight)

    def auto_scale_xyz(self, X, Y, Z=None, had_data=None):
        # This updates the bounding boxes as to keep a record as to what the
        # minimum sized rectangular volume holds the data.
        if np.shape(X) == np.shape(Y):
            self.xy_dataLim.update_from_data_xy(
                np.column_stack([np.ravel(X), np.ravel(Y)]), not had_data)
        else:
            self.xy_dataLim.update_from_data_x(X, not had_data)
            self.xy_dataLim.update_from_data_y(Y, not had_data)
        if Z is not None:
            self.zz_dataLim.update_from_data_x(Z, not had_data)
        # Let autoscale_view figure out how to use this data.
        self.autoscale_view()

    def autoscale_view(self, tight=None, scalex=True, scaley=True,
                       scalez=True):
        """
        Autoscale the view limits using the data limits.

        See `.Axes.autoscale_view` for full documentation.  Because this
        function applies to 3D Axes, it also takes a *scalez* argument.
        """
        # This method looks at the rectangular volume (see above)
        # of data and decides how to scale the view portal to fit it.
        if tight is None:
            _tight = self._tight
            if not _tight:
                # if image data only just use the datalim
                for artist in self._children:
                    if isinstance(artist, mimage.AxesImage):
                        _tight = True
                    elif isinstance(artist, (mlines.Line2D, mpatches.Patch)):
                        _tight = False
                        break
        else:
            _tight = self._tight = bool(tight)

        if scalex and self.get_autoscalex_on():
            self._shared_axes["x"].clean()
            x0, x1 = self.xy_dataLim.intervalx
            xlocator = self.xaxis.get_major_locator()
            x0, x1 = xlocator.nonsingular(x0, x1)
            if self._xmargin > 0:
                delta = (x1 - x0) * self._xmargin
                x0 -= delta
                x1 += delta
            if not _tight:
                x0, x1 = xlocator.view_limits(x0, x1)
            self.set_xbound(x0, x1)

        if scaley and self.get_autoscaley_on():
            self._shared_axes["y"].clean()
            y0, y1 = self.xy_dataLim.intervaly
            ylocator = self.yaxis.get_major_locator()
            y0, y1 = ylocator.nonsingular(y0, y1)
            if self._ymargin > 0:
                delta = (y1 - y0) * self._ymargin
                y0 -= delta
                y1 += delta
            if not _tight:
                y0, y1 = ylocator.view_limits(y0, y1)
            self.set_ybound(y0, y1)

        if scalez and self.get_autoscalez_on():
            self._shared_axes["z"].clean()
            z0, z1 = self.zz_dataLim.intervalx
            zlocator = self.zaxis.get_major_locator()
            z0, z1 = zlocator.nonsingular(z0, z1)
            if self._zmargin > 0:
                delta = (z1 - z0) * self._zmargin
                z0 -= delta
                z1 += delta
            if not _tight:
                z0, z1 = zlocator.view_limits(z0, z1)
            self.set_zbound(z0, z1)

    def get_w_lims(self):
        """Get 3D world limits."""
        minx, maxx = self.get_xlim3d()
        miny, maxy = self.get_ylim3d()
        minz, maxz = self.get_zlim3d()
        return minx, maxx, miny, maxy, minz, maxz

    # set_xlim, set_ylim are directly inherited from base Axes.
    @_api.make_keyword_only("3.6", "emit")
    def set_zlim(self, bottom=None, top=None, emit=True, auto=False,
                 *, zmin=None, zmax=None):
        """
        Set 3D z limits.

        See `.Axes.set_ylim` for full documentation
        """
        if top is None and np.iterable(bottom):
            bottom, top = bottom
        if zmin is not None:
            if bottom is not None:
                raise TypeError("Cannot pass both 'bottom' and 'zmin'")
            bottom = zmin
        if zmax is not None:
            if top is not None:
                raise TypeError("Cannot pass both 'top' and 'zmax'")
            top = zmax
        return self.zaxis._set_lim(bottom, top, emit=emit, auto=auto)

    set_xlim3d = maxes.Axes.set_xlim
    set_ylim3d = maxes.Axes.set_ylim
    set_zlim3d = set_zlim

    def get_xlim(self):
        # docstring inherited
        return tuple(self.xy_viewLim.intervalx)

    def get_ylim(self):
        # docstring inherited
        return tuple(self.xy_viewLim.intervaly)

    def get_zlim(self):
        """Get 3D z limits."""
        return tuple(self.zz_viewLim.intervalx)

    get_zscale = _axis_method_wrapper("zaxis", "get_scale")

    # Redefine all three methods to overwrite their docstrings.
    set_xscale = _axis_method_wrapper("xaxis", "_set_axes_scale")
    set_yscale = _axis_method_wrapper("yaxis", "_set_axes_scale")
    set_zscale = _axis_method_wrapper("zaxis", "_set_axes_scale")
    set_xscale.__doc__, set_yscale.__doc__, set_zscale.__doc__ = map(
        """
        Set the {}-axis scale.

        Parameters
        ----------
        value : {{"linear"}}
            The axis scale type to apply.  3D axes currently only support
            linear scales; other scales yield nonsensical results.

        **kwargs
            Keyword arguments are nominally forwarded to the scale class, but
            none of them is applicable for linear scales.
        """.format,
        ["x", "y", "z"])

    get_zticks = _axis_method_wrapper("zaxis", "get_ticklocs")
    set_zticks = _axis_method_wrapper("zaxis", "set_ticks")
    get_zmajorticklabels = _axis_method_wrapper("zaxis", "get_majorticklabels")
    get_zminorticklabels = _axis_method_wrapper("zaxis", "get_minorticklabels")
    get_zticklabels = _axis_method_wrapper("zaxis", "get_ticklabels")
    set_zticklabels = _axis_method_wrapper(
        "zaxis", "set_ticklabels",
        doc_sub={"Axis.set_ticks": "Axes3D.set_zticks"})

    zaxis_date = _axis_method_wrapper("zaxis", "axis_date")
    if zaxis_date.__doc__:
        zaxis_date.__doc__ += textwrap.dedent("""

        Notes
        -----
        This function is merely provided for completeness, but 3D axes do not
        support dates for ticks, and so this may not work as expected.
        """)

    def clabel(self, *args, **kwargs):
        """Currently not implemented for 3D axes, and returns *None*."""
        return None

    def view_init(self, elev=None, azim=None, roll=None, vertical_axis="z"):
        """
        Set the elevation and azimuth of the axes in degrees (not radians).

        This can be used to rotate the axes programmatically.

        To look normal to the primary planes, the following elevation and
        azimuth angles can be used. A roll angle of 0, 90, 180, or 270 deg
        will rotate these views while keeping the axes at right angles.

        ==========   ====  ====
        view plane   elev  azim
        ==========   ====  ====
        XY           90    -90
        XZ           0     -90
        YZ           0     0
        -XY          -90   90
        -XZ          0     90
        -YZ          0     180
        ==========   ====  ====

        Parameters
        ----------
        elev : float, default: None
            The elevation angle in degrees rotates the camera above the plane
            pierced by the vertical axis, with a positive angle corresponding
            to a location above that plane. For example, with the default
            vertical axis of 'z', the elevation defines the angle of the camera
            location above the x-y plane.
            If None, then the initial value as specified in the `Axes3D`
            constructor is used.
        azim : float, default: None
            The azimuthal angle in degrees rotates the camera about the
            vertical axis, with a positive angle corresponding to a
            right-handed rotation. For example, with the default vertical axis
            of 'z', a positive azimuth rotates the camera about the origin from
            its location along the +x axis towards the +y axis.
            If None, then the initial value as specified in the `Axes3D`
            constructor is used.
        roll : float, default: None
            The roll angle in degrees rotates the camera about the viewing
            axis. A positive angle spins the camera clockwise, causing the
            scene to rotate counter-clockwise.
            If None, then the initial value as specified in the `Axes3D`
            constructor is used.
        vertical_axis : {"z", "x", "y"}, default: "z"
            The axis to align vertically. *azim* rotates about this axis.
        """

        self._dist = 10  # The camera distance from origin. Behaves like zoom

        if elev is None:
            self.elev = self.initial_elev
        else:
            self.elev = elev

        if azim is None:
            self.azim = self.initial_azim
        else:
            self.azim = azim

        if roll is None:
            self.roll = self.initial_roll
        else:
            self.roll = roll

        self._vertical_axis = _api.check_getitem(
            dict(x=0, y=1, z=2), vertical_axis=vertical_axis
        )

    def set_proj_type(self, proj_type, focal_length=None):
        """
        Set the projection type.

        Parameters
        ----------
        proj_type : {'persp', 'ortho'}
            The projection type.
        focal_length : float, default: None
            For a projection type of 'persp', the focal length of the virtual
            camera. Must be > 0. If None, defaults to 1.
            The focal length can be computed from a desired Field Of View via
            the equation: focal_length = 1/tan(FOV/2)
        """
        _api.check_in_list(['persp', 'ortho'], proj_type=proj_type)
        if proj_type == 'persp':
            if focal_length is None:
                focal_length = 1
            elif focal_length <= 0:
                raise ValueError(f"focal_length = {focal_length} must be "
                                 "greater than 0")
            self._focal_length = focal_length
        else:  # 'ortho':
            if focal_length not in (None, np.inf):
                raise ValueError(f"focal_length = {focal_length} must be "
                                 f"None for proj_type = {proj_type}")
            self._focal_length = np.inf

    def _roll_to_vertical(self, arr):
        """Roll arrays to match the different vertical axis."""
        return np.roll(arr, self._vertical_axis - 2)

    def get_proj(self):
        """Create the projection matrix from the current viewing position."""

        # Transform to uniform world coordinates 0-1, 0-1, 0-1
        box_aspect = self._roll_to_vertical(self._box_aspect)
        worldM = proj3d.world_transformation(
            *self.get_xlim3d(),
            *self.get_ylim3d(),
            *self.get_zlim3d(),
            pb_aspect=box_aspect,
        )

        # Look into the middle of the world coordinates:
        R = 0.5 * box_aspect

        # elev: elevation angle in the z plane.
        # azim: azimuth angle in the xy plane.
        # Coordinates for a point that rotates around the box of data.
        # p0, p1 corresponds to rotating the box only around the vertical axis.
        # p2 corresponds to rotating the box only around the horizontal axis.
        elev_rad = np.deg2rad(self.elev)
        azim_rad = np.deg2rad(self.azim)
        p0 = np.cos(elev_rad) * np.cos(azim_rad)
        p1 = np.cos(elev_rad) * np.sin(azim_rad)
        p2 = np.sin(elev_rad)

        # When changing vertical axis the coordinates changes as well.
        # Roll the values to get the same behaviour as the default:
        ps = self._roll_to_vertical([p0, p1, p2])

        # The coordinates for the eye viewing point. The eye is looking
        # towards the middle of the box of data from a distance:
        eye = R + self._dist * ps

        # vvec, self._vvec and self._eye are unused, remove when deprecated
        vvec = R - eye
        self._eye = eye
        self._vvec = vvec / np.linalg.norm(vvec)

        # Calculate the viewing axes for the eye position
        u, v, w = self._calc_view_axes(eye)
        self._view_u = u  # _view_u is towards the right of the screen
        self._view_v = v  # _view_v is towards the top of the screen
        self._view_w = w  # _view_w is out of the screen

        # Generate the view and projection transformation matrices
        if self._focal_length == np.inf:
            # Orthographic projection
            viewM = proj3d._view_transformation_uvw(u, v, w, eye)
            projM = proj3d._ortho_transformation(-self._dist, self._dist)
        else:
            # Perspective projection
            # Scale the eye dist to compensate for the focal length zoom effect
            eye_focal = R + self._dist * ps * self._focal_length
            viewM = proj3d._view_transformation_uvw(u, v, w, eye_focal)
            projM = proj3d._persp_transformation(-self._dist,
                                                 self._dist,
                                                 self._focal_length)

        # Combine all the transformation matrices to get the final projection
        M0 = np.dot(viewM, worldM)
        M = np.dot(projM, M0)
        return M

    def mouse_init(self, rotate_btn=1, pan_btn=2, zoom_btn=3):
        """
        Set the mouse buttons for 3D rotation and zooming.

        Parameters
        ----------
        rotate_btn : int or list of int, default: 1
            The mouse button or buttons to use for 3D rotation of the axes.
        pan_btn : int or list of int, default: 2
            The mouse button or buttons to use to pan the 3D axes.
        zoom_btn : int or list of int, default: 3
            The mouse button or buttons to use to zoom the 3D axes.
        """
        self.button_pressed = None
        # coerce scalars into array-like, then convert into
        # a regular list to avoid comparisons against None
        # which breaks in recent versions of numpy.
        self._rotate_btn = np.atleast_1d(rotate_btn).tolist()
        self._pan_btn = np.atleast_1d(pan_btn).tolist()
        self._zoom_btn = np.atleast_1d(zoom_btn).tolist()

    def disable_mouse_rotation(self):
        """Disable mouse buttons for 3D rotation, panning, and zooming."""
        self.mouse_init(rotate_btn=[], pan_btn=[], zoom_btn=[])

    def can_zoom(self):
        # doc-string inherited
        return True

    def can_pan(self):
        # doc-string inherited
        return True

    def sharez(self, other):
        """
        Share the z-axis with *other*.

        This is equivalent to passing ``sharez=other`` when constructing the
        Axes, and cannot be used if the z-axis is already being shared with
        another Axes.
        """
        _api.check_isinstance(maxes._base._AxesBase, other=other)
        if self._sharez is not None and other is not self._sharez:
            raise ValueError("z-axis is already shared")
        self._shared_axes["z"].join(self, other)
        self._sharez = other
        self.zaxis.major = other.zaxis.major  # Ticker instances holding
        self.zaxis.minor = other.zaxis.minor  # locator and formatter.
        z0, z1 = other.get_zlim()
        self.set_zlim(z0, z1, emit=False, auto=other.get_autoscalez_on())
        self.zaxis._scale = other.zaxis._scale

    def clear(self):
        # docstring inherited.
        super().clear()
        if self._focal_length == np.inf:
            self._zmargin = mpl.rcParams['axes.zmargin']
        else:
            self._zmargin = 0.
        self.grid(mpl.rcParams['axes3d.grid'])

    def _button_press(self, event):
        if event.inaxes == self:
            self.button_pressed = event.button
            self._sx, self._sy = event.xdata, event.ydata
            toolbar = self.figure.canvas.toolbar
            if toolbar and toolbar._nav_stack() is None:
                toolbar.push_current()

    def _button_release(self, event):
        self.button_pressed = None
        toolbar = self.figure.canvas.toolbar
        # backend_bases.release_zoom and backend_bases.release_pan call
        # push_current, so check the navigation mode so we don't call it twice
        if toolbar and self.get_navigate_mode() is None:
            toolbar.push_current()

    def _get_view(self):
        # docstring inherited
        return (self.get_xlim(), self.get_ylim(), self.get_zlim(),
                self.elev, self.azim, self.roll)

    def _set_view(self, view):
        # docstring inherited
        xlim, ylim, zlim, elev, azim, roll = view
        self.set(xlim=xlim, ylim=ylim, zlim=zlim)
        self.elev = elev
        self.azim = azim
        self.roll = roll

    def format_zdata(self, z):
        """
        Return *z* string formatted.  This function will use the
        :attr:`fmt_zdata` attribute if it is callable, else will fall
        back on the zaxis major formatter
        """
        try:
            return self.fmt_zdata(z)
        except (AttributeError, TypeError):
            func = self.zaxis.get_major_formatter().format_data_short
            val = func(z)
            return val

    def format_coord(self, xd, yd):
        """
        Given the 2D view coordinates attempt to guess a 3D coordinate.
        Looks for the nearest edge to the point and then assumes that
        the point is at the same z location as the nearest point on the edge.
        """

        if self.M is None:
            return ''

        if self.button_pressed in self._rotate_btn:
            # ignore xd and yd and display angles instead
            norm_elev = art3d._norm_angle(self.elev)
            norm_azim = art3d._norm_angle(self.azim)
            norm_roll = art3d._norm_angle(self.roll)
            return (f"elevation={norm_elev:.0f}\N{DEGREE SIGN}, "
                    f"azimuth={norm_azim:.0f}\N{DEGREE SIGN}, "
                    f"roll={norm_roll:.0f}\N{DEGREE SIGN}"
                    ).replace("-", "\N{MINUS SIGN}")

        # nearest edge
        p0, p1 = min(self._tunit_edges(),
                     key=lambda edge: proj3d._line2d_seg_dist(
                         (xd, yd), edge[0][:2], edge[1][:2]))

        # scale the z value to match
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        d0 = np.hypot(x0-xd, y0-yd)
        d1 = np.hypot(x1-xd, y1-yd)
        dt = d0+d1
        z = d1/dt * z0 + d0/dt * z1

        x, y, z = proj3d.inv_transform(xd, yd, z, self.M)

        xs = self.format_xdata(x)
        ys = self.format_ydata(y)
        zs = self.format_zdata(z)
        return f'x={xs}, y={ys}, z={zs}'

    def _on_move(self, event):
        """
        Mouse moving.

        By default, button-1 rotates, button-2 pans, and button-3 zooms;
        these buttons can be modified via `mouse_init`.
        """

        if not self.button_pressed:
            return

        if self.get_navigate_mode() is not None:
            # we don't want to rotate if we are zooming/panning
            # from the toolbar
            return

        if self.M is None:
            return

        x, y = event.xdata, event.ydata
        # In case the mouse is out of bounds.
        if x is None or event.inaxes != self:
            return

        dx, dy = x - self._sx, y - self._sy
        w = self._pseudo_w
        h = self._pseudo_h

        # Rotation
        if self.button_pressed in self._rotate_btn:
            # rotate viewing point
            # get the x and y pixel coords
            if dx == 0 and dy == 0:
                return

            roll = np.deg2rad(self.roll)
            delev = -(dy/h)*180*np.cos(roll) + (dx/w)*180*np.sin(roll)
            dazim = -(dy/h)*180*np.sin(roll) - (dx/w)*180*np.cos(roll)
            self.elev = self.elev + delev
            self.azim = self.azim + dazim
            self.stale = True

        elif self.button_pressed in self._pan_btn:
            # Start the pan event with pixel coordinates
            px, py = self.transData.transform([self._sx, self._sy])
            self.start_pan(px, py, 2)
            # pan view (takes pixel coordinate input)
            self.drag_pan(2, None, event.x, event.y)
            self.end_pan()

        # Zoom
        elif self.button_pressed in self._zoom_btn:
            # zoom view (dragging down zooms in)
            scale = h/(h - dy)
            self._scale_axis_limits(scale, scale, scale)

        # Store the event coordinates for the next time through.
        self._sx, self._sy = x, y
        # Always request a draw update at the end of interaction
        self.figure.canvas.draw_idle()

    def drag_pan(self, button, key, x, y):
        # docstring inherited

        # Get the coordinates from the move event
        p = self._pan_start
        (xdata, ydata), (xdata_start, ydata_start) = p.trans_inverse.transform(
            [(x, y), (p.x, p.y)])
        self._sx, self._sy = xdata, ydata
        # Calling start_pan() to set the x/y of this event as the starting
        # move location for the next event
        self.start_pan(x, y, button)
        du, dv = xdata - xdata_start, ydata - ydata_start
        dw = 0
        if key == 'x':
            dv = 0
        elif key == 'y':
            du = 0
        if du == 0 and dv == 0:
            return

        # Transform the pan from the view axes to the data axes
        R = np.array([self._view_u, self._view_v, self._view_w])
        R = -R / self._box_aspect * self._dist
        duvw_projected = R.T @ np.array([du, dv, dw])

        # Calculate pan distance
        minx, maxx, miny, maxy, minz, maxz = self.get_w_lims()
        dx = (maxx - minx) * duvw_projected[0]
        dy = (maxy - miny) * duvw_projected[1]
        dz = (maxz - minz) * duvw_projected[2]

        # Set the new axis limits
        self.set_xlim3d(minx + dx, maxx + dx)
        self.set_ylim3d(miny + dy, maxy + dy)
        self.set_zlim3d(minz + dz, maxz + dz)

    def _calc_view_axes(self, eye):
        """
        Get the unit vectors for the viewing axes in data coordinates.
        `u` is towards the right of the screen
        `v` is towards the top of the screen
        `w` is out of the screen
        """
        elev_rad = np.deg2rad(art3d._norm_angle(self.elev))
        roll_rad = np.deg2rad(art3d._norm_angle(self.roll))

        # Look into the middle of the world coordinates
        R = 0.5 * self._roll_to_vertical(self._box_aspect)

        # Define which axis should be vertical. A negative value
        # indicates the plot is upside down and therefore the values
        # have been reversed:
        V = np.zeros(3)
        V[self._vertical_axis] = -1 if abs(elev_rad) > np.pi/2 else 1

        u, v, w = proj3d._view_axes(eye, R, V, roll_rad)
        return u, v, w

    def _set_view_from_bbox(self, bbox, direction='in',
                            mode=None, twinx=False, twiny=False):
        """
        Zoom in or out of the bounding box.

        Will center the view in the center of the bounding box, and zoom by
        the ratio of the size of the bounding box to the size of the Axes3D.
        """
        (start_x, start_y, stop_x, stop_y) = bbox
        if mode == 'x':
            start_y = self.bbox.min[1]
            stop_y = self.bbox.max[1]
        elif mode == 'y':
            start_x = self.bbox.min[0]
            stop_x = self.bbox.max[0]

        # Clip to bounding box limits
        start_x, stop_x = np.clip(sorted([start_x, stop_x]),
                                  self.bbox.min[0], self.bbox.max[0])
        start_y, stop_y = np.clip(sorted([start_y, stop_y]),
                                  self.bbox.min[1], self.bbox.max[1])

        # Move the center of the view to the center of the bbox
        zoom_center_x = (start_x + stop_x)/2
        zoom_center_y = (start_y + stop_y)/2

        ax_center_x = (self.bbox.max[0] + self.bbox.min[0])/2
        ax_center_y = (self.bbox.max[1] + self.bbox.min[1])/2

        self.start_pan(zoom_center_x, zoom_center_y, 2)
        self.drag_pan(2, None, ax_center_x, ax_center_y)
        self.end_pan()

        # Calculate zoom level
        dx = abs(start_x - stop_x)
        dy = abs(start_y - stop_y)
        scale_u = dx / (self.bbox.max[0] - self.bbox.min[0])
        scale_v = dy / (self.bbox.max[1] - self.bbox.min[1])

        # Keep aspect ratios equal
        scale = max(scale_u, scale_v)

        # Zoom out
        if direction == 'out':
            scale = 1 / scale

        self._zoom_data_limits(scale, scale, scale)

    def _zoom_data_limits(self, scale_u, scale_v, scale_w):
        """
        Zoom in or out of a 3D plot.

        Will scale the data limits by the scale factors. These will be
        transformed to the x, y, z data axes based on the current view angles.
        A scale factor > 1 zooms out and a scale factor < 1 zooms in.

        For an axes that has had its aspect ratio set to 'equal', 'equalxy',
        'equalyz', or 'equalxz', the relevant axes are constrained to zoom
        equally.

        Parameters
        ----------
        scale_u : float
            Scale factor for the u view axis (view screen horizontal).
        scale_v : float
            Scale factor for the v view axis (view screen vertical).
        scale_w : float
            Scale factor for the w view axis (view screen depth).
        """
        scale = np.array([scale_u, scale_v, scale_w])

        # Only perform frame conversion if unequal scale factors
        if not np.allclose(scale, scale_u):
            # Convert the scale factors from the view frame to the data frame
            R = np.array([self._view_u, self._view_v, self._view_w])
            S = scale * np.eye(3)
            scale = np.linalg.norm(R.T @ S, axis=1)

            # Set the constrained scale factors to the factor closest to 1
            if self._aspect in ('equal', 'equalxy', 'equalxz', 'equalyz'):
                ax_idxs = self._equal_aspect_axis_indices(self._aspect)
                min_ax_idxs = np.argmin(np.abs(scale[ax_idxs] - 1))
                scale[ax_idxs] = scale[ax_idxs][min_ax_idxs]

        self._scale_axis_limits(scale[0], scale[1], scale[2])

    def _scale_axis_limits(self, scale_x, scale_y, scale_z):
        """
        Keeping the center of the x, y, and z data axes fixed, scale their
        limits by scale factors. A scale factor > 1 zooms out and a scale
        factor < 1 zooms in.

        Parameters
        ----------
        scale_x : float
            Scale factor for the x data axis.
        scale_y : float
            Scale factor for the y data axis.
        scale_z : float
            Scale factor for the z data axis.
        """
        # Get the axis limits and centers
        minx, maxx, miny, maxy, minz, maxz = self.get_w_lims()
        cx = (maxx + minx)/2
        cy = (maxy + miny)/2
        cz = (maxz + minz)/2

        # Scale the data range
        dx = (maxx - minx)*scale_x
        dy = (maxy - miny)*scale_y
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
            color = kwargs.pop('color', None)
            if color is None:
                color = self._get_lines.get_next_color()
            color = np.array(mcolors.to_rgba(color))

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
    () == (2, 0)
    assert ax.get_zbound() == (0, 2)

    # Set upper bound
    ax.set_zbound(upper=1)
    assert ax.zaxis_inverted()
    assert ax.get_zlim() == (1, 0)
    assert ax.get_zbound() == (0, 1)

    # Set lower bound
    ax.set_zbound(lower=2)
    assert ax.zaxis_inverted()
    assert ax.get_zlim() == (2, 1)
    assert ax.get_zbound() == (1, 2)


def test_set_zlim():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    assert ax.get_zlim() == (0, 1)
    ax.set_zlim(zmax=2)
    assert ax.get_zlim() == (0, 2)
    ax.set_zlim(zmin=1)
    assert ax.get_zlim() == (1, 2)

    with pytest.raises(
            TypeError, match="Cannot pass both 'bottom' and 'zmin'"):
        ax.set_zlim(bottom=0, zmin=1)
    with pytest.raises(
            TypeError, match="Cannot pass both 'top' and 'zmax'"):
        ax.set_zlim(top=0, zmax=1)


def test_shared_axes_retick():
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection="3d")
    ax2 = fig.add_subplot(212, projection="3d", sharez=ax1)
    ax1.plot([0, 1], [0, 1], [0, 2])
    ax2.plot([0, 1], [0, 1], [0, 2])
    ax1.set_zticks([-0.5, 0, 2, 2.5])
    # check that setting ticks on a shared axis is synchronized
    assert ax1.get_zlim() == (-0.5, 2.5)
    assert ax2.get_zlim() == (-0.5, 2.5)


def test_pan():
    """Test mouse panning using the middle mouse button."""

    def convert_lim(dmin, dmax):
        """Convert min/max limits to center and range."""
        center = (dmin + dmax) / 2
        range_ = dmax - dmin
        return center, range_

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(0, 0, 0)
    ax.figure.canvas.draw()

    x_center0, x_range0 = convert_lim(*ax.get_xlim3d())
    y_center0, y_range0 = convert_lim(*ax.get_ylim3d())
    z_center0, z_range0 = convert_lim(*ax.get_zlim3d())

    # move mouse diagonally to pan along all axis.
    ax._button_press(
        mock_event(ax, button=MouseButton.MIDDLE, xdata=0, ydata=0))
    ax._on_move(
        mock_event(ax, button=MouseButton.MIDDLE, xdata=1, ydata=1))

    x_center, x_range = convert_lim(*ax.get_xlim3d())
    y_center, y_range = convert_lim(*ax.get_ylim3d())
    z_center, z_range = convert_lim(*ax.get_zlim3d())

    # Ranges have not changed
    assert x_range == pytest.approx(x_range0)
    assert y_range == pytest.approx(y_range0)
    assert z_range == pytest.approx(z_range0)

    # But center positions have
    assert x_center != pytest.approx(x_center0)
    assert y_center != pytest.approx(y_center0)
    assert z_center != pytest.approx(z_center0)


@pytest.mark.parametrize("tool,button,key,expected",
                         [("zoom", MouseButton.LEFT, None,  # zoom in
                          ((0.00, 0.06), (0.01, 0.07), (0.02, 0.08))),
                          ("zoom", MouseButton.LEFT, 'x',  # zoom in
                          ((-0.01, 0.10), (-0.03, 0.08), (-0.06, 0.06))),
                          ("zoom", MouseButton.LEFT, 'y',  # zoom in
                          ((-0.07, 0.04), (-0.03, 0.08), (0.00, 0.11))),
                          ("zoom", MouseButton.RIGHT, None,  # zoom out
                          ((-0.09, 0.15), (-0.07, 0.17), (-0.06, 0.18))),
                          ("pan", MouseButton.LEFT, None,
                          ((-0.70, -0.58), (-1.03, -0.91), (-1.27, -1.15))),
                          ("pan", MouseButton.LEFT, 'x',
                          ((-0.96, -0.84), (-0.58, -0.46), (-0.06, 0.06))),
                          ("pan", MouseButton.LEFT, 'y',
                          ((0.20, 0.32), (-0.51, -0.39), (-1.27, -1.15)))])
def test_toolbar_zoom_pan(tool, button, key, expected):
    # NOTE: The expected zoom values are rough ballparks of moving in the view
    #       to make sure we are getting the right direction of motion.
    #       The specific values can and should change if the zoom movement
    #       scaling factor gets updated.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(0, 0, 0)
    fig.canvas.draw()
    xlim0, ylim0, zlim0 = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()

    # Mouse from (0, 0) to (1, 1)
    d0 = (0, 0)
    d1 = (1, 1)
    # Convert to screen coordinates ("s").  Events are defined only with pixel
    # precision, so round the pixel values, and below, check against the
    # corresponding xdata/ydata, which are close but not equal to d0/d1.
    s0 = ax.transData.transform(d0).astype(int)
    s1 = ax.transData.transform(d1).astype(int)

    # Set up the mouse movements
    start_event = MouseEvent(
        "button_press_event", fig.canvas, *s0, button, key=key)
    stop_event = MouseEvent(
        "button_release_event", fig.canvas, *s1, button, key=key)

    tb = NavigationToolbar2(fig.canvas)
    if tool == "zoom":
        tb.zoom()
        tb.press_zoom(start_event)
        tb.drag_zoom(stop_event)
        tb.release_zoom(stop_event)
    else:
        tb.pan()
        tb.press_pan(start_event)
        tb.drag_pan(stop_event)
        tb.release_pan(stop_event)

    # Should be close, but won't be exact due to screen integer resolution
    xlim, ylim, zlim = expected
    assert ax.get_xlim3d() == pytest.approx(xlim, abs=0.01)
    assert ax.get_ylim3d() == pytest.approx(ylim, abs=0.01)
    assert ax.get_zlim3d() == pytest.approx(zlim, abs=0.01)

    # Ensure that back, forward, and home buttons work
    tb.back()
    assert ax.get_xlim3d() == pytest.approx(xlim0)
    assert ax.get_ylim3d() == pytest.approx(ylim0)
    assert ax.get_zlim3d() == pytest.approx(zlim0)

    tb.forward()
    assert ax.get_xlim3d() == pytest.approx(xlim, abs=0.01)
    assert ax.get_ylim3d() == pytest.approx(ylim, abs=0.01)
    assert ax.get_zlim3d() == pytest.approx(zlim, abs=0.01)

    tb.home()
    assert ax.get_xlim3d() == pytest.approx(xlim0)
    assert ax.get_ylim3d() == pytest.approx(ylim0)
    assert ax.get_zlim3d() == pytest.approx(zlim0)


@mpl.style.context('default')
@check_figures_equal(extensions=["png"])
def test_scalarmap_update(fig_test, fig_ref):

    x, y, z = np.array((list(itertools.product(*[np.arange(0, 5, 1),
                                                 np.arange(0, 5, 1),
                                                 np.arange(0, 5, 1)])))).T
    c = x + y

    # test
    ax_test = fig_test.add_subplot(111, projection='3d')
    sc_test = ax_test.scatter(x, y, z, c=c, s=40, cmap='viridis')
    # force a draw
    fig_test.canvas.draw()
    # mark it as "stale"
    sc_test.changed()

    # ref
    ax_ref = fig_ref.add_subplot(111, projection='3d')
    sc_ref = ax_ref.scatter(x, y, z, c=c, s=40, cmap='viridis')


def test_subfigure_simple():
    # smoketest that subfigures can work...
    fig = plt.figure()
    sf = fig.subfigures(1, 2)
    ax = sf[0].add_subplot(1, 1, 1, projection='3d')
    ax = sf[1].add_subplot(1, 1, 1, projection='3d', label='other')


# Update style when regenerating the test image
@image_comparison(baseline_images=['computed_zorder'], remove_text=True,
                  extensions=['png'], style=('classic', '_classic_test_patch'))
def test_computed_zorder():
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.computed_zorder = False

    # create a horizontal plane
    corners = ((0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0))
    for ax in (ax1, ax2):
        tri = art3d.Poly3DCollection([corners],
                                     facecolors='white',
                                     edgecolors='black',
                                     zorder=1)
        ax.add_collection3d(tri)

        # plot a vector
        ax.plot((2, 2), (2, 2), (0, 4), c='red', zorder=2)

        # plot some points
        ax.scatter((3, 3), (1, 3), (1, 3), c='red', zorder=10)

        ax.set_xlim((0, 5.0))
        ax.set_ylim((0, 5.0))
        ax.set_zlim((0, 2.5))

    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.computed_zorder = False

<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
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

        def calc_arrows(UVW

=======
    dim = 10
    X, Y = np.meshgrid((-dim, dim

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
), (-dim, dim))
    Z = np.zeros((2, 2))

    angle = 0.5
    X2, Y2 = np.meshgrid((-dim, dim), (0, dim))
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
            # compute the two arrowhead direction unit vectors
            rangle = math.radians(15

=======
    Z2 = Y2 * angle
    X3, Y3 = np.meshgrid((-dim, dim

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
            c = math.cos(rangle

=======
, (-dim, 0

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
            s = math.sin(rangle

=======

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)
    Z3 = Y3 * angle
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
)
            r13 = y_p * s
            r32 = x_p * s
            r12 = x_p * y_p * (1 - c)
            Rpos = np.array(

=======


>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
    r = 7
    M = 1000
    th = np.linspace(0, 2 * np.pi, M)
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
 * (1 - c), r12, r13],

=======

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
                 [r12, c + (y_p ** 2

=======

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
) * (1 - c), -r32],
                 [-r13, r32, np.full_like(x_p, c

=======
    x, y, z = r * np.cos(th

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
),  r * np.sin(th), angle * r * np.sin(th)
    for ax in (ax3, ax4):
        ax.plot_surface(X2, Y3, Z3,
                        color='blue',
                        alpha=0.5,
                        linewidth=0,
                        zorder=-1)
        ax.plot(x[y < 0], y[y < 0], z[y < 0],
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
) result.
            return np.stack([Rpos_vecs, Rneg_vecs], axis=1

=======
                lw=5,
                linestyle='--',
                color='green',
                zorder=0

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py

        had_data = self.has_data(

=======

        ax.plot_surface(X, Y, Z,
                        color='red',
                        alpha=0.5,
                        linewidth=0,
                        zorder=1

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)

        ax.plot(r * np.sin(th), r * np.cos(th), np.zeros(M),
                lw=5,
                linestyle='--',
                color='black',
                zorder=2)

        ax.plot_surface(X2, Y2, Z2,
                        color='blue',
                        alpha=0.5,
                        linewidth=0,
                        zorder=3)

        ax.plot(x[y > 0], y[y > 0], z[y > 0], lw=5,
                linestyle='--',
                color='green',
                zorder=4)
        ax.view_init(elev=20, azim=-20, roll=0)
        ax.axis('off')


def test_format_coord():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(10)
    ax.plot(x, np.sin(x))
    fig.canvas.draw()
    assert ax.format_coord(0, 0) == 'x=1.8066, y=1.0367, z=−0.0553'
    # Modify parameters
    ax.view_init(roll=30, vertical_axis="y")
    fig.canvas.draw()
    assert ax.format_coord(0, 0) == 'x=9.1651, y=−0.9215, z=−0.0359'
    # Reset parameters
    ax.view_init()
    fig.canvas.draw()
    assert ax.format_coord(0, 0) == 'x=1.8066, y=1.0367, z=−0.0553'


def test_get_axis_position():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(10)
    ax.plot(x, np.sin(x))
    fig.canvas.draw()
    assert ax.get_axis_position() == (False, True, False)


def test_margins():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.margins(0.2)
    assert ax.margins() == (0.2, 0.2, 0.2)
    ax.margins(0.1, 0.2, 0.3)
    assert ax.margins() == (0.1, 0.2, 0.3)
    ax.margins(x=0)
    assert ax.margins() == (0, 0.2, 0.3)
    ax.margins(y=0.1)
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
                        f"When multidimensional, {name} must match the shape "
                        "of filled"

=======
    assert ax.margins() == (0, 0.1, 0.3

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)
    ax.margins(z=0)
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
                raise ValueError(f"Invalid {name} argument"

=======
    assert ax.margins() == (0, 0.1, 0

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)


@pytest.mark.parametrize('err, args, kwargs, match', (
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
    ax = fig_ref.add_subplot(projection="3d"
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
:N]). e.g. *errorevery* =(6, 3

=======

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
)
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
<<<<<<< /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/left.py
    @_api.make_keyword_only("3.8", "call_axes_locator")
    def get_tightbbox(self, renderer=None, call_axes_locator=True,

=======

>>>>>>> /home/ze/miningframework/bug_results/matplotlib_results/matplotlib/9296df8c760461c97a59dff79689cb2490d2500c/lib/mpl_toolkits/mplot3d/tests/test_axes3d.py/right.py
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
