r"""
A module for parsing a subset of the TeX math syntax and rendering it to a
Matplotlib backend.

For a tutorial of its usage, see :doc:`/tutorials/text/mathtext`.  This
document is primarily concerned with implementation details.

The module uses pyparsing_ to parse the TeX expression.

.. _pyparsing: https://pypi.org/project/pyparsing/

The Bakoma distribution of the TeX Computer Modern fonts, and STIX
fonts are supported.  There is experimental support for using
arbitrary fonts, but results may vary without proper tweaking and
metrics for those fonts.
"""

from collections import namedtuple
import functools
from io import StringIO
import logging
import types

import numpy as np
from PIL import Image

from matplotlib import _api, colors as mcolors, rcParams, _mathtext
from matplotlib.ft2font import FT2Image, LOAD_NO_HINTING
from matplotlib.font_manager import FontProperties
# Backcompat imports, all are deprecated as of 3.4.
from matplotlib._mathtext import (  # noqa: F401
    SHRINK_FACTOR, GROW_FACTOR, NUM_SIZE_LEVELS)
from matplotlib._mathtext_data import (  # noqa: F401
    latex_to_bakoma, latex_to_cmex, latex_to_standard, stix_virtual_fonts,
    tex2uni)

_log = logging.getLogger(__name__)


get_unicode_index = _mathtext.get_unicode_index
get_unicode_index.__module__ = __name__


class MathtextBackend:
    """
    The base class for the mathtext backend-specific code.  `MathtextBackend`
    subclasses interface between mathtext and specific Matplotlib graphics
    backends.

    Subclasses need to override the following:

    - :meth:`render_glyph`
    - :meth:`render_rect_filled`
    - :meth:`get_results`

    And optionally, if you need to use a FreeType hinting style:

    - :meth:`get_hinting_type`
    """
    def __init__(self):
        self.width = 0
        self.height = 0
        self.depth = 0

    def set_canvas_size(self, w, h, d):
        """Set the dimension of the drawing canvas."""
        self.width  = w
        self.height = h
        self.depth  = d

    def render_glyph(self, ox, oy, info):
        """
        Draw a glyph described by *info* to the reference point (*ox*,
        *oy*).
        """
        raise NotImplementedError()

    def render_rect_filled(self, x1, y1, x2, y2):
        """
        Draw a filled black rectangle from (*x1*, *y1*) to (*x2*, *y2*).
        """
        raise NotImplementedError()

    def get_results(self, box):
        """
        Return a backend-specific tuple to return to the backend after
        all processing is done.
        """
        raise NotImplementedError()

    def get_hinting_type(self):
        """
        Get the FreeType hinting type to use with this particular
        backend.
        """
        return LOAD_NO_HINTING


class MathtextBackendAgg(MathtextBackend):
    """
    Render glyphs and rectangles to an FTImage buffer, which is later
    transferred to the Agg image by the Agg backend.
    """
    def __init__(self):
        self.ox = 0
        self.oy = 0
        self.image = None
        self.mode = 'bbox'
        self.bbox = [0, 0, 0, 0]
        super().__init__()

    def _update_bbox(self, x1, y1, x2, y2):
        self.bbox = [min(self.bbox[0], x1),
                     min(self.bbox[1], y1),
                     max(self.bbox[2], x2),
                     max(self.bbox[3], y2)]

    def set_canvas_size(self, w, h, d):
        super().set_canvas_size(w, h, d)
        if self.mode != 'bbox':
            self.image = FT2Image(np.ceil(w), np.ceil(h + max(d, 0)))

    def render_glyph(self, ox, oy, info):
        if self.mode == 'bbox':
            self._update_bbox(ox + info.metrics.xmin,
                              oy - info.metrics.ymax,
                              ox + info.metrics.xmax,
                              oy - info.metrics.ymin)
        else:
            info.font.draw_glyph_to_bitmap(
                self.image, ox, oy - info.metrics.iceberg, info.glyph,
                antialiased=rcParams['text.antialiased'])

    def render_rect_filled(self, x1, y1, x2, y2):
        if self.mode == 'bbox':
            self._update_bbox(x1, y1, x2, y2)
        else:
            height = max(int(y2 - y1) - 1, 0)
            if height == 0:
                center = (y2 + y1) / 2.0
                y = int(center - (height + 1) / 2.0)
            else:
                y = int(y1)
            self.image.draw_rect_filled(int(x1), y, np.ceil(x2), y + height)

    def get_results(self, box, used_characters):
        self.mode = 'bbox'
        orig_height = box.height
        orig_depth  = box.depth
        _mathtext.ship(0, 0, box)
        bbox = self.bbox
        bbox = [bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1]
        self.mode = 'render'
        self.set_canvas_size(
            bbox[2] - bbox[0],
            (bbox[3] - bbox[1]) - orig_depth,
            (bbox[3] - bbox[1]) - orig_height)
        _mathtext.ship(-bbox[0], -bbox[1], box)
        result = (self.ox,
                  self.oy,
                  self.width,
                  self.height + self.depth,
                  self.depth,
                  self.image,
                  used_characters)
        self.image = None
        return result

    def get_hinting_type(self):
        from matplotlib.backends import backend_agg
        return backend_agg.get_hinting_flag()


@_api.deprecated("3.4", alternative="mathtext.math_to_image")
class MathtextBackendBitmap(MathtextBackendAgg):
    def get_results(self, box, used_characters):
        ox, oy, width, height, depth, image, characters = \
            super().get_results(box, used_characters)
        return image, depth


@_api.deprecated("3.4", alternative="MathtextBackendPath")
class MathtextBackendPs(MathtextBackend):
    """
    Store information to write a mathtext rendering to the PostScript backend.
    """

    _PSResult = namedtuple(
        "_PSResult", "width height depth pswriter used_characters")

    def __init__(self):
        self.pswriter = StringIO()
        self.lastfont = None

    def render_glyph(self, ox, oy, info):
        oy = self.height - oy + info.offset
        postscript_name = info.postscript_name
        fontsize = info.fontsize

        if (postscript_name, fontsize) != self.lastfont:
            self.lastfont = postscript_name, fontsize
            self.pswriter.write(
                f"/{postscript_name} findfont\n"
                f"{fontsize} scalefont\n"
                f"setfont\n")

        self.pswriter.write(
            f"{ox:f} {oy:f} moveto\n"
            f"/{info.symbol_name} glyphshow\n")

    def render_rect_filled(self, x1, y1, x2, y2):
        ps = "%f %f %f %f rectfill\n" % (
            x1, self.height - y2, x2 - x1, y2 - y1)
        self.pswriter.write(ps)

    def get_results(self, box, used_characters):
        _mathtext.ship(0, 0, box)
        return self._PSResult(self.width,
                              self.height + self.depth,
                              self.depth,
                              self.pswriter,
                              used_characters)


@_api.deprecated("3.4", alternative="MathtextBackendPath")
class MathtextBackendPdf(MathtextBackend):
    """Store information to write a mathtext rendering to the PDF backend."""

    _PDFResult = namedtuple(
        "_PDFResult", "width height depth glyphs rects used_characters")

    def __init__(self):
        self.glyphs = []
        self.rects = []

    def render_glyph(self, ox, oy, info):
        filename = info.font.fname
        oy = self.height - oy + info.offset
        self.glyphs.append(
            (ox, oy, filename, info.fontsize,
             info.num, info.symbol_name))

    def render_rect_filled(self, x1, y1, x2, y2):
        self.rects.append((x1, self.height - y2, x2 - x1, y2 - y1))

    def get_results(self, box, used_characters):
        _mathtext.ship(0, 0, box)
        return self._PDFResult(self.width,
                               self.height + self.depth,
                               self.depth,
                               self.glyphs,
                               self.rects,
                               used_characters)


@_api.deprecated("3.4", alternative="MathtextBackendPath")
class MathtextBackendSvg(MathtextBackend):
    """
    Store information to write a mathtext rendering to the SVG
    backend.
    """
    def __init__(self):
        self.svg_glyphs = []
        self.svg_rects = []

    def render_glyph(self, ox, oy, info):
        oy = self.height - oy + info.offset

        self.svg_glyphs.append(
            (info.font, info.fontsize, info.num, ox, oy, info.metrics))

    def render_rect_filled(self, x1, y1, x2, y2):
        self.svg_rects.append(
            (x1, self.height - y1 + 1, x2 - x1, y2 - y1))

    def get_results(self, box, used_characters):
        _mathtext.ship(0, 0, box)
        svg_elements = types.SimpleNamespace(svg_glyphs=self.svg_glyphs,
                                             svg_rects=self.svg_rects)
        return (self.width,
                self.height + self.depth,
                self.depth,
                svg_elements,
                used_characters)


class MathtextBackendPath(MathtextBackend):
    """
    Store information to write a mathtext rendering to the text path
    machinery.
    """

    _Result = namedtuple("_Result", "width height depth glyphs rects")

    def __init__(self):
        self.glyphs = []
        self.rects = []

    def render_glyph(self, ox, oy, info):
        oy = self.height - oy + info.offset
        self.glyphs.append((info.font, info.fontsize, info.num, ox, oy))

    def render_rect_filled(self, x1, y1, x2, y2):
        self.rects.append((x1, self.height - y2, x2 - x1, y2 - y1))

    def get_results(self, box, used_characters):
        _mathtext.ship(0, 0, box)
        return self._Result(self.width,
                            self.height + self.depth,
                            self.depth,
                            self.glyphs,
                            self.rects)


@_api.deprecated("3.4", alternative="MathtextBackendPath")
class MathtextBackendCairo(MathtextBackend):
    """
    Store information to write a mathtext rendering to the Cairo
    backend.
    """

    def __init__(self):
        self.glyphs = []
        self.rects = []

    def render_glyph(self, ox, oy, info):
        oy = oy - info.offset - self.height
        thetext = chr(info.num)
        self.glyphs.append(
            (info.font, info.fontsize, thetext, ox, oy))

    def render_rect_filled(self, x1, y1, x2, y2):
        self.rects.append(
            (x1, y1 - self.height, x2 - x1, y2 - y1))

    def get_results(self, box, used_characters):
        _mathtext.ship(0, 0, box)
        return (self.width,
                self.height + self.depth,
                self.depth,
                self.glyphs,
                self.rects)


for _cls_name in [
        "Fonts",
        *[c.__name__ for c in _mathtext.Fonts.__subclasses__()],
        "FontConstantsBase",
        *[c.__name__ for c in _mathtext.FontConstantsBase.__subclasses__()],
        "Node",
        *[c.__name__ for c in _mathtext.Node.__subclasses__()],
        "Ship", "Parser",
]:
    globals()[_cls_name] = _api.deprecated("3.4")(
        type(_cls_name, (getattr(_mathtext, _cls_name),), {}))


class MathTextWarning(Warning):
    pass


@_api.deprecated("3.3")
class GlueSpec:
    """See `Glue`."""

    def __init__(self, width=0., stretch=0., stretch_order=0,
                 shrink=0., shrink_order=0):
        self.width         = width
        self.stretch       = stretch
        self.stretch_order = stretch_order
        self.shrink        = shrink
        self.shrink_order  = shrink_order

    def copy(self):
        return GlueSpec(
            self.width,
            self.stretch,
            self.stretch_order,
            self.shrink,
            self.shrink_order)

    @classmethod
    def factory(cls, glue_type):
        return cls._types[glue_type]


with _api.suppress_matplotlib_deprecation_warning():
    GlueSpec._types = {k: GlueSpec(**v._asdict())
                       for k, v in _mathtext._GlueSpec._named.items()}


<<<<<<< /home/ze/miningframework/mining_results/matplotlib_results/matplotlib/96274e6315ced46aa72f4f12993166847c26e4b0/lib/matplotlib/mathtext.py/left.py
@_api.deprecated("3.4")
def ship(ox, oy, box):
    _mathtext.ship(ox, oy, box)
||||||| /home/ze/miningframework/mining_results/matplotlib_results/matplotlib/96274e6315ced46aa72f4f12993166847c26e4b0/lib/matplotlib/mathtext.py/base.py
# Some convenient ways to get common kinds of glue


@cbook.deprecated("3.3", alternative="Glue('fil')")
class Fil(Glue):
    def __init__(self):
        Glue.__init__(self, 'fil')


@cbook.deprecated("3.3", alternative="Glue('fill')")
class Fill(Glue):
    def __init__(self):
        Glue.__init__(self, 'fill')


@cbook.deprecated("3.3", alternative="Glue('filll')")
class Filll(Glue):
    def __init__(self):
        Glue.__init__(self, 'filll')


@cbook.deprecated("3.3", alternative="Glue('neg_fil')")
class NegFil(Glue):
    def __init__(self):
        Glue.__init__(self, 'neg_fil')


@cbook.deprecated("3.3", alternative="Glue('neg_fill')")
class NegFill(Glue):
    def __init__(self):
        Glue.__init__(self, 'neg_fill')


@cbook.deprecated("3.3", alternative="Glue('neg_filll')")
class NegFilll(Glue):
    def __init__(self):
        Glue.__init__(self, 'neg_filll')


@cbook.deprecated("3.3", alternative="Glue('ss')")
class SsGlue(Glue):
    def __init__(self):
        Glue.__init__(self, 'ss')


class HCentered(Hlist):
    """
    A convenience class to create an `Hlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements):
        super().__init__([Glue('ss'), *elements, Glue('ss')], do_kern=False)


class VCentered(Vlist):
    """
    A convenience class to create a `Vlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements):
        super().__init__([Glue('ss'), *elements, Glue('ss')])


class Kern(Node):
    """
    A `Kern` node has a width field to specify a (normally
    negative) amount of spacing. This spacing correction appears in
    horizontal lists between letters like A and V when the font
    designer said that it looks better to move them closer together or
    further apart. A kern node can also appear in a vertical list,
    when its *width* denotes additional spacing in the vertical
    direction.
    """

    height = 0
    depth = 0

    def __init__(self, width):
        Node.__init__(self)
        self.width = width

    def __repr__(self):
        return "k%.02f" % self.width

    def shrink(self):
        Node.shrink(self)
        if self.size < NUM_SIZE_LEVELS:
            self.width *= SHRINK_FACTOR

    def grow(self):
        Node.grow(self)
        self.width *= GROW_FACTOR


class SubSuperCluster(Hlist):
    """
    A hack to get around that fact that this code does a two-pass parse like
    TeX.  This lets us store enough information in the hlist itself, namely the
    nucleus, sub- and super-script, such that if another script follows that
    needs to be attached, it can be reconfigured on the fly.
    """

    def __init__(self):
        self.nucleus = None
        self.sub = None
        self.super = None
        Hlist.__init__(self, [])


class AutoHeightChar(Hlist):
    """
    A character as close to the given height and depth as possible.

    When using a font with multiple height versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c, height, depth, state, always=False, factor=None):
        alternatives = state.font_output.get_sized_alternatives_for_symbol(
            state.font, c)

        xHeight = state.font_output.get_xheight(
            state.font, state.fontsize, state.dpi)

        state = state.copy()
        target_total = height + depth
        for fontname, sym in alternatives:
            state.font = fontname
            char = Char(sym, state)
            # Ensure that size 0 is chosen when the text is regular sized but
            # with descender glyphs by subtracting 0.2 * xHeight
            if char.height + char.depth >= target_total - 0.2 * xHeight:
                break

        shift = 0
        if state.font != 0:
            if factor is None:
                factor = target_total / (char.height + char.depth)
            state.fontsize *= factor
            char = Char(sym, state)

            shift = (depth - char.depth)

        Hlist.__init__(self, [char])
        self.shift_amount = shift


class AutoWidthChar(Hlist):
    """
    A character as close to the given width as possible.

    When using a font with multiple width versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c, width, state, always=False, char_class=Char):
        alternatives = state.font_output.get_sized_alternatives_for_symbol(
            state.font, c)

        state = state.copy()
        for fontname, sym in alternatives:
            state.font = fontname
            char = char_class(sym, state)
            if char.width >= width:
                break

        factor = width / char.width
        state.fontsize *= factor
        char = char_class(sym, state)

        Hlist.__init__(self, [char])
        self.width = char.width


class Ship:
    """
    Ship boxes to output once they have been set up, this sends them to output.

    Since boxes can be inside of boxes inside of boxes, the main work of `Ship`
    is done by two mutually recursive routines, `hlist_out` and `vlist_out`,
    which traverse the `Hlist` nodes and `Vlist` nodes inside of horizontal
    and vertical boxes.  The global variables used in TeX to store state as it
    processes have become member variables here.
    """

    def __call__(self, ox, oy, box):
        self.max_push    = 0  # Deepest nesting of push commands so far
        self.cur_s       = 0
        self.cur_v       = 0.
        self.cur_h       = 0.
        self.off_h       = ox
        self.off_v       = oy + box.height
        self.hlist_out(box)

    @staticmethod
    def clamp(value):
        if value < -1000000000.:
            return -1000000000.
        if value > 1000000000.:
            return 1000000000.
        return value

    def hlist_out(self, box):
        cur_g         = 0
        cur_glue      = 0.
        glue_order    = box.glue_order
        glue_sign     = box.glue_sign
        base_line     = self.cur_v
        left_edge     = self.cur_h
        self.cur_s    += 1
        self.max_push = max(self.cur_s, self.max_push)
        clamp         = self.clamp

        for p in box.children:
            if isinstance(p, Char):
                p.render(self.cur_h + self.off_h, self.cur_v + self.off_v)
                self.cur_h += p.width
            elif isinstance(p, Kern):
                self.cur_h += p.width
            elif isinstance(p, List):
                # node623
                if len(p.children) == 0:
                    self.cur_h += p.width
                else:
                    edge = self.cur_h
                    self.cur_v = base_line + p.shift_amount
                    if isinstance(p, Hlist):
                        self.hlist_out(p)
                    else:
                        # p.vpack(box.height + box.depth, 'exactly')
                        self.vlist_out(p)
                    self.cur_h = edge + p.width
                    self.cur_v = base_line
            elif isinstance(p, Box):
                # node624
                rule_height = p.height
                rule_depth  = p.depth
                rule_width  = p.width
                if np.isinf(rule_height):
                    rule_height = box.height
                if np.isinf(rule_depth):
                    rule_depth = box.depth
                if rule_height > 0 and rule_width > 0:
                    self.cur_v = base_line + rule_depth
                    p.render(self.cur_h + self.off_h,
                             self.cur_v + self.off_v,
                             rule_width, rule_height)
                    self.cur_v = base_line
                self.cur_h += rule_width
            elif isinstance(p, Glue):
                # node625
                glue_spec = p.glue_spec
                rule_width = glue_spec.width - cur_g
                if glue_sign != 0:  # normal
                    if glue_sign == 1:  # stretching
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch
                            cur_g = round(clamp(box.glue_set * cur_glue))
                    elif glue_spec.shrink_order == glue_order:
                        cur_glue += glue_spec.shrink
                        cur_g = round(clamp(box.glue_set * cur_glue))
                rule_width += cur_g
                self.cur_h += rule_width
        self.cur_s -= 1

    def vlist_out(self, box):
        cur_g         = 0
        cur_glue      = 0.
        glue_order    = box.glue_order
        glue_sign     = box.glue_sign
        self.cur_s    += 1
        self.max_push = max(self.max_push, self.cur_s)
        left_edge     = self.cur_h
        self.cur_v    -= box.height
        top_edge      = self.cur_v
        clamp         = self.clamp

        for p in box.children:
            if isinstance(p, Kern):
                self.cur_v += p.width
            elif isinstance(p, List):
                if len(p.children) == 0:
                    self.cur_v += p.height + p.depth
                else:
                    self.cur_v += p.height
                    self.cur_h = left_edge + p.shift_amount
                    save_v = self.cur_v
                    p.width = box.width
                    if isinstance(p, Hlist):
                        self.hlist_out(p)
                    else:
                        self.vlist_out(p)
                    self.cur_v = save_v + p.depth
                    self.cur_h = left_edge
            elif isinstance(p, Box):
                rule_height = p.height
                rule_depth = p.depth
                rule_width = p.width
                if np.isinf(rule_width):
                    rule_width = box.width
                rule_height += rule_depth
                if rule_height > 0 and rule_depth > 0:
                    self.cur_v += rule_height
                    p.render(self.cur_h + self.off_h,
                             self.cur_v + self.off_v,
                             rule_width, rule_height)
            elif isinstance(p, Glue):
                glue_spec = p.glue_spec
                rule_height = glue_spec.width - cur_g
                if glue_sign != 0:  # normal
                    if glue_sign == 1:  # stretching
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch
                            cur_g = round(clamp(box.glue_set * cur_glue))
                    elif glue_spec.shrink_order == glue_order:  # shrinking
                        cur_glue += glue_spec.shrink
                        cur_g = round(clamp(box.glue_set * cur_glue))
                rule_height += cur_g
                self.cur_v += rule_height
            elif isinstance(p, Char):
                raise RuntimeError(
                    "Internal mathtext error: Char node found in vlist")
        self.cur_s -= 1


ship = Ship()


##############################################################################
# PARSER


def Error(msg):
    """Helper class to raise parser errors."""
    def raise_error(s, loc, toks):
        raise ParseFatalException(s, loc, msg)

    empty = Empty()
    empty.setParseAction(raise_error)
    return empty


class Parser:
    """
    A pyparsing-based parser for strings containing math expressions.

    Raw text may also appear outside of pairs of ``$``.

    The grammar is based directly on that in TeX, though it cuts a few corners.
    """

    _math_style_dict = dict(displaystyle=0, textstyle=1,
                            scriptstyle=2, scriptscriptstyle=3)

    _binary_operators = set('''
      + * -
      \\pm             \\sqcap                   \\rhd
      \\mp             \\sqcup                   \\unlhd
      \\times          \\vee                     \\unrhd
      \\div            \\wedge                   \\oplus
      \\ast            \\setminus                \\ominus
      \\star           \\wr                      \\otimes
      \\circ           \\diamond                 \\oslash
      \\bullet         \\bigtriangleup           \\odot
      \\cdot           \\bigtriangledown         \\bigcirc
      \\cap            \\triangleleft            \\dagger
      \\cup            \\triangleright           \\ddagger
      \\uplus          \\lhd                     \\amalg'''.split())

    _relation_symbols = set('''
      = < > :
      \\leq        \\geq        \\equiv   \\models
      \\prec       \\succ       \\sim     \\perp
      \\preceq     \\succeq     \\simeq   \\mid
      \\ll         \\gg         \\asymp   \\parallel
      \\subset     \\supset     \\approx  \\bowtie
      \\subseteq   \\supseteq   \\cong    \\Join
      \\sqsubset   \\sqsupset   \\neq     \\smile
      \\sqsubseteq \\sqsupseteq \\doteq   \\frown
      \\in         \\ni         \\propto  \\vdash
      \\dashv      \\dots       \\dotplus \\doteqdot'''.split())

    _arrow_symbols = set('''
      \\leftarrow              \\longleftarrow           \\uparrow
      \\Leftarrow              \\Longleftarrow           \\Uparrow
      \\rightarrow             \\longrightarrow          \\downarrow
      \\Rightarrow             \\Longrightarrow          \\Downarrow
      \\leftrightarrow         \\longleftrightarrow      \\updownarrow
      \\Leftrightarrow         \\Longleftrightarrow      \\Updownarrow
      \\mapsto                 \\longmapsto              \\nearrow
      \\hookleftarrow          \\hookrightarrow          \\searrow
      \\leftharpoonup          \\rightharpoonup          \\swarrow
      \\leftharpoondown        \\rightharpoondown        \\nwarrow
      \\rightleftharpoons      \\leadsto'''.split())

    _spaced_symbols = _binary_operators | _relation_symbols | _arrow_symbols

    _punctuation_symbols = set(r', ; . ! \ldotp \cdotp'.split())

    _overunder_symbols = set(r'''
       \sum \prod \coprod \bigcap \bigcup \bigsqcup \bigvee
       \bigwedge \bigodot \bigotimes \bigoplus \biguplus
       '''.split())

    _overunder_functions = set(
        "lim liminf limsup sup max min".split())

    _dropsub_symbols = set(r'''\int \oint'''.split())

    _fontnames = set("rm cal it tt sf bf default bb frak scr regular".split())

    _function_names = set("""
      arccos csc ker min arcsin deg lg Pr arctan det lim sec arg dim
      liminf sin cos exp limsup sinh cosh gcd ln sup cot hom log tan
      coth inf max tanh""".split())

    _ambi_delim = set("""
      | \\| / \\backslash \\uparrow \\downarrow \\updownarrow \\Uparrow
      \\Downarrow \\Updownarrow . \\vert \\Vert \\\\|""".split())

    _left_delim = set(r"( [ \{ < \lfloor \langle \lceil".split())

    _right_delim = set(r") ] \} > \rfloor \rangle \rceil".split())

    def __init__(self):
        p = types.SimpleNamespace()
        # All forward declarations are here
        p.accent           = Forward()
        p.ambi_delim       = Forward()
        p.apostrophe       = Forward()
        p.auto_delim       = Forward()
        p.binom            = Forward()
        p.bslash           = Forward()
        p.c_over_c         = Forward()
        p.customspace      = Forward()
        p.end_group        = Forward()
        p.float_literal    = Forward()
        p.font             = Forward()
        p.frac             = Forward()
        p.dfrac            = Forward()
        p.function         = Forward()
        p.genfrac          = Forward()
        p.group            = Forward()
        p.int_literal      = Forward()
        p.latexfont        = Forward()
        p.lbracket         = Forward()
        p.left_delim       = Forward()
        p.lbrace           = Forward()
        p.main             = Forward()
        p.math             = Forward()
        p.math_string      = Forward()
        p.non_math         = Forward()
        p.operatorname     = Forward()
        p.overline         = Forward()
        p.placeable        = Forward()
        p.rbrace           = Forward()
        p.rbracket         = Forward()
        p.required_group   = Forward()
        p.right_delim      = Forward()
        p.right_delim_safe = Forward()
        p.simple           = Forward()
        p.simple_group     = Forward()
        p.single_symbol    = Forward()
        p.accentprefixed   = Forward()
        p.space            = Forward()
        p.sqrt             = Forward()
        p.stackrel         = Forward()
        p.start_group      = Forward()
        p.subsuper         = Forward()
        p.subsuperop       = Forward()
        p.symbol           = Forward()
        p.symbol_name      = Forward()
        p.token            = Forward()
        p.unknown_symbol   = Forward()

        # Set names on everything -- very useful for debugging
        for key, val in vars(p).items():
            if not key.startswith('_'):
                val.setName(key)

        p.float_literal <<= Regex(r"[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)")
        p.int_literal   <<= Regex("[-+]?[0-9]+")

        p.lbrace        <<= Literal('{').suppress()
        p.rbrace        <<= Literal('}').suppress()
        p.lbracket      <<= Literal('[').suppress()
        p.rbracket      <<= Literal(']').suppress()
        p.bslash        <<= Literal('\\')

        p.space         <<= oneOf(list(self._space_widths))
        p.customspace   <<= (
            Suppress(Literal(r'\hspace'))
            - ((p.lbrace + p.float_literal + p.rbrace)
               | Error(r"Expected \hspace{n}"))
        )

        unicode_range = "\U00000080-\U0001ffff"
        p.single_symbol <<= Regex(
            r"([a-zA-Z0-9 +\-*/<>=:,.;!\?&'@()\[\]|%s])|(\\[%%${}\[\]_|])" %
            unicode_range)
        p.accentprefixed <<= Suppress(p.bslash) + oneOf(self._accentprefixed)
        p.symbol_name   <<= (
            Combine(p.bslash + oneOf(list(tex2uni)))
            + FollowedBy(Regex("[^A-Za-z]").leaveWhitespace() | StringEnd())
        )
        p.symbol        <<= (p.single_symbol | p.symbol_name).leaveWhitespace()

        p.apostrophe    <<= Regex("'+")

        p.c_over_c      <<= (
            Suppress(p.bslash)
            + oneOf(list(self._char_over_chars))
        )

        p.accent        <<= Group(
            Suppress(p.bslash)
            + oneOf([*self._accent_map, *self._wide_accents])
            - p.placeable
        )

        p.function      <<= (
            Suppress(p.bslash)
            + oneOf(list(self._function_names))
        )

        p.start_group    <<= Optional(p.latexfont) + p.lbrace
        p.end_group      <<= p.rbrace.copy()
        p.simple_group   <<= Group(p.lbrace + ZeroOrMore(p.token) + p.rbrace)
        p.required_group <<= Group(p.lbrace + OneOrMore(p.token) + p.rbrace)
        p.group          <<= Group(
            p.start_group + ZeroOrMore(p.token) + p.end_group
        )

        p.font          <<= Suppress(p.bslash) + oneOf(list(self._fontnames))
        p.latexfont     <<= (
            Suppress(p.bslash)
            + oneOf(['math' + x for x in self._fontnames])
        )

        p.frac          <<= Group(
            Suppress(Literal(r"\frac"))
            - ((p.required_group + p.required_group)
               | Error(r"Expected \frac{num}{den}"))
        )

        p.dfrac         <<= Group(
            Suppress(Literal(r"\dfrac"))
            - ((p.required_group + p.required_group)
               | Error(r"Expected \dfrac{num}{den}"))
        )

        p.stackrel      <<= Group(
            Suppress(Literal(r"\stackrel"))
            - ((p.required_group + p.required_group)
               | Error(r"Expected \stackrel{num}{den}"))
        )

        p.binom         <<= Group(
            Suppress(Literal(r"\binom"))
            - ((p.required_group + p.required_group)
               | Error(r"Expected \binom{num}{den}"))
        )

        p.ambi_delim    <<= oneOf(list(self._ambi_delim))
        p.left_delim    <<= oneOf(list(self._left_delim))
        p.right_delim   <<= oneOf(list(self._right_delim))
        p.right_delim_safe <<= oneOf([*(self._right_delim - {'}'}), r'\}'])

        p.genfrac <<= Group(
            Suppress(Literal(r"\genfrac"))
            - (((p.lbrace
                 + Optional(p.ambi_delim | p.left_delim, default='')
                 + p.rbrace)
                + (p.lbrace
                   + Optional(p.ambi_delim | p.right_delim_safe, default='')
                   + p.rbrace)
                + (p.lbrace + p.float_literal + p.rbrace)
                + p.simple_group + p.required_group + p.required_group)
               | Error("Expected "
                       r"\genfrac{ldelim}{rdelim}{rulesize}{style}{num}{den}"))
        )

        p.sqrt <<= Group(
            Suppress(Literal(r"\sqrt"))
            - ((Optional(p.lbracket + p.int_literal + p.rbracket, default=None)
                + p.required_group)
               | Error("Expected \\sqrt{value}"))
        )

        p.overline <<= Group(
            Suppress(Literal(r"\overline"))
            - (p.required_group | Error("Expected \\overline{value}"))
        )

        p.unknown_symbol <<= Combine(p.bslash + Regex("[A-Za-z]*"))

        p.operatorname <<= Group(
            Suppress(Literal(r"\operatorname"))
            - ((p.lbrace + ZeroOrMore(p.simple | p.unknown_symbol) + p.rbrace)
               | Error("Expected \\operatorname{value}"))
        )

        p.placeable     <<= (
            p.accentprefixed  # Must be before accent so named symbols that are
                              # prefixed with an accent name work
            | p.accent   # Must be before symbol as all accents are symbols
            | p.symbol   # Must be third to catch all named symbols and single
                         # chars not in a group
            | p.c_over_c
            | p.function
            | p.group
            | p.frac
            | p.dfrac
            | p.stackrel
            | p.binom
            | p.genfrac
            | p.sqrt
            | p.overline
            | p.operatorname
        )

        p.simple        <<= (
            p.space
            | p.customspace
            | p.font
            | p.subsuper
        )

        p.subsuperop    <<= oneOf(["_", "^"])

        p.subsuper      <<= Group(
            (Optional(p.placeable)
             + OneOrMore(p.subsuperop - p.placeable)
             + Optional(p.apostrophe))
            | (p.placeable + Optional(p.apostrophe))
            | p.apostrophe
        )

        p.token         <<= (
            p.simple
            | p.auto_delim
            | p.unknown_symbol  # Must be last
        )

        p.auto_delim    <<= (
            Suppress(Literal(r"\left"))
            - ((p.left_delim | p.ambi_delim)
               | Error("Expected a delimiter"))
            + Group(ZeroOrMore(p.simple | p.auto_delim))
            + Suppress(Literal(r"\right"))
            - ((p.right_delim | p.ambi_delim)
               | Error("Expected a delimiter"))
        )

        p.math          <<= OneOrMore(p.token)

        p.math_string   <<= QuotedString('$', '\\', unquoteResults=False)

        p.non_math      <<= Regex(r"(?:(?:\\[$])|[^$])*").leaveWhitespace()

        p.main          <<= (
            p.non_math + ZeroOrMore(p.math_string + p.non_math) + StringEnd()
        )

        # Set actions
        for key, val in vars(p).items():
            if not key.startswith('_'):
                if hasattr(self, key):
                    val.setParseAction(getattr(self, key))

        self._expression = p.main
        self._math_expression = p.math

    def parse(self, s, fonts_object, fontsize, dpi):
        """
        Parse expression *s* using the given *fonts_object* for
        output, at the given *fontsize* and *dpi*.

        Returns the parse tree of `Node` instances.
        """
        self._state_stack = [
            self.State(fonts_object, 'default', 'rm', fontsize, dpi)]
        self._em_width_cache = {}
        try:
            result = self._expression.parseString(s)
        except ParseBaseException as err:
            raise ValueError("\n".join(["",
                                        err.line,
                                        " " * (err.column - 1) + "^",
                                        str(err)])) from err
        self._state_stack = None
        self._em_width_cache = {}
        self._expression.resetCache()
        return result[0]

    # The state of the parser is maintained in a stack.  Upon
    # entering and leaving a group { } or math/non-math, the stack
    # is pushed and popped accordingly.  The current state always
    # exists in the top element of the stack.
    class State:
        """
        Stores the state of the parser.

        States are pushed and popped from a stack as necessary, and
        the "current" state is always at the top of the stack.
        """
        def __init__(self, font_output, font, font_class, fontsize, dpi):
            self.font_output = font_output
            self._font = font
            self.font_class = font_class
            self.fontsize = fontsize
            self.dpi = dpi

        def copy(self):
            return Parser.State(
                self.font_output,
                self.font,
                self.font_class,
                self.fontsize,
                self.dpi)

        @property
        def font(self):
            return self._font

        @font.setter
        def font(self, name):
            if name in ('rm', 'it', 'bf'):
                self.font_class = name
            self._font = name

    def get_state(self):
        """Get the current `State` of the parser."""
        return self._state_stack[-1]

    def pop_state(self):
        """Pop a `State` off of the stack."""
        self._state_stack.pop()

    def push_state(self):
        """Push a new `State` onto the stack, copying the current state."""
        self._state_stack.append(self.get_state().copy())

    def main(self, s, loc, toks):
        return [Hlist(toks)]

    def math_string(self, s, loc, toks):
        return self._math_expression.parseString(toks[0][1:-1])

    def math(self, s, loc, toks):
        hlist = Hlist(toks)
        self.pop_state()
        return [hlist]

    def non_math(self, s, loc, toks):
        s = toks[0].replace(r'\$', '$')
        symbols = [Char(c, self.get_state(), math=False) for c in s]
        hlist = Hlist(symbols)
        # We're going into math now, so set font to 'it'
        self.push_state()
        self.get_state().font = rcParams['mathtext.default']
        return [hlist]

    def _make_space(self, percentage):
        # All spaces are relative to em width
        state = self.get_state()
        key = (state.font, state.fontsize, state.dpi)
        width = self._em_width_cache.get(key)
        if width is None:
            metrics = state.font_output.get_metrics(
                state.font, rcParams['mathtext.default'], 'm', state.fontsize,
                state.dpi)
            width = metrics.advance
            self._em_width_cache[key] = width
        return Kern(width * percentage)

    _space_widths = {
        r'\,':         0.16667,   # 3/18 em = 3 mu
        r'\thinspace': 0.16667,   # 3/18 em = 3 mu
        r'\/':         0.16667,   # 3/18 em = 3 mu
        r'\>':         0.22222,   # 4/18 em = 4 mu
        r'\:':         0.22222,   # 4/18 em = 4 mu
        r'\;':         0.27778,   # 5/18 em = 5 mu
        r'\ ':         0.33333,   # 6/18 em = 6 mu
        r'~':          0.33333,   # 6/18 em = 6 mu, nonbreakable
        r'\enspace':   0.5,       # 9/18 em = 9 mu
        r'\quad':      1,         # 1 em = 18 mu
        r'\qquad':     2,         # 2 em = 36 mu
        r'\!':         -0.16667,  # -3/18 em = -3 mu
    }

    def space(self, s, loc, toks):
        assert len(toks) == 1
        num = self._space_widths[toks[0]]
        box = self._make_space(num)
        return [box]

    def customspace(self, s, loc, toks):
        return [self._make_space(float(toks[0]))]

    def symbol(self, s, loc, toks):
        c = toks[0]
        try:
            char = Char(c, self.get_state())
        except ValueError as err:
            raise ParseFatalException(s, loc,
                                      "Unknown symbol: %s" % c) from err

        if c in self._spaced_symbols:
            # iterate until we find previous character, needed for cases
            # such as ${ -2}$, $ -2$, or $   -2$.
            prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
            # Binary operators at start of string should not be spaced
            if (c in self._binary_operators and
                    (len(s[:loc].split()) == 0 or prev_char == '{' or
                     prev_char in self._left_delim)):
                return [char]
            else:
                return [Hlist([self._make_space(0.2),
                               char,
                               self._make_space(0.2)],
                              do_kern=True)]
        elif c in self._punctuation_symbols:

            # Do not space commas between brackets
            if c == ',':
                prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
                next_char = next((c for c in s[loc + 1:] if c != ' '), '')
                if prev_char == '{' and next_char == '}':
                    return [char]

            # Do not space dots as decimal separators
            if c == '.' and s[loc - 1].isdigit() and s[loc + 1].isdigit():
                return [char]
            else:
                return [Hlist([char, self._make_space(0.2)], do_kern=True)]
        return [char]

    accentprefixed = symbol

    def unknown_symbol(self, s, loc, toks):
        c = toks[0]
        raise ParseFatalException(s, loc, "Unknown symbol: %s" % c)

    _char_over_chars = {
        # The first 2 entries in the tuple are (font, char, sizescale) for
        # the two symbols under and over.  The third element is the space
        # (in multiples of underline height)
        r'AA': (('it', 'A', 1.0), (None, '\\circ', 0.5), 0.0),
    }

    def c_over_c(self, s, loc, toks):
        sym = toks[0]
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        under_desc, over_desc, space = \
            self._char_over_chars.get(sym, (None, None, 0.0))
        if under_desc is None:
            raise ParseFatalException("Error parsing symbol")

        over_state = state.copy()
        if over_desc[0] is not None:
            over_state.font = over_desc[0]
        over_state.fontsize *= over_desc[2]
        over = Accent(over_desc[1], over_state)

        under_state = state.copy()
        if under_desc[0] is not None:
            under_state.font = under_desc[0]
        under_state.fontsize *= under_desc[2]
        under = Char(under_desc[1], under_state)

        width = max(over.width, under.width)

        over_centered = HCentered([over])
        over_centered.hpack(width, 'exactly')

        under_centered = HCentered([under])
        under_centered.hpack(width, 'exactly')

        return Vlist([
                over_centered,
                Vbox(0., thickness * space),
                under_centered
                ])

    _accent_map = {
        r'hat':            r'\circumflexaccent',
        r'breve':          r'\combiningbreve',
        r'bar':            r'\combiningoverline',
        r'grave':          r'\combininggraveaccent',
        r'acute':          r'\combiningacuteaccent',
        r'tilde':          r'\combiningtilde',
        r'dot':            r'\combiningdotabove',
        r'ddot':           r'\combiningdiaeresis',
        r'vec':            r'\combiningrightarrowabove',
        r'"':              r'\combiningdiaeresis',
        r"`":              r'\combininggraveaccent',
        r"'":              r'\combiningacuteaccent',
        r'~':              r'\combiningtilde',
        r'.':              r'\combiningdotabove',
        r'^':              r'\circumflexaccent',
        r'overrightarrow': r'\rightarrow',
        r'overleftarrow':  r'\leftarrow',
        r'mathring':       r'\circ',
    }

    _wide_accents = set(r"widehat widetilde widebar".split())

    # make a lambda and call it to get the namespace right
    _accentprefixed = (lambda am: [
        p for p in tex2uni
        if any(p.startswith(a) and a != p for a in am)
    ])(set(_accent_map))

    def accent(self, s, loc, toks):
        assert len(toks) == 1
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        if len(toks[0]) != 2:
            raise ParseFatalException("Error parsing accent")
        accent, sym = toks[0]
        if accent in self._wide_accents:
            accent_box = AutoWidthChar(
                '\\' + accent, sym.width, state, char_class=Accent)
        else:
            accent_box = Accent(self._accent_map[accent], state)
        if accent == 'mathring':
            accent_box.shrink()
            accent_box.shrink()
        centered = HCentered([Hbox(sym.width / 4.0), accent_box])
        centered.hpack(sym.width, 'exactly')
        return Vlist([
                centered,
                Vbox(0., thickness * 2.0),
                Hlist([sym])
                ])

    def function(self, s, loc, toks):
        self.push_state()
        state = self.get_state()
        state.font = 'rm'
        hlist = Hlist([Char(c, state) for c in toks[0]])
        self.pop_state()
        hlist.function_name = toks[0]
        return hlist

    def operatorname(self, s, loc, toks):
        self.push_state()
        state = self.get_state()
        state.font = 'rm'
        # Change the font of Chars, but leave Kerns alone
        for c in toks[0]:
            if isinstance(c, Char):
                c.font = 'rm'
                c._update_metrics()
        self.pop_state()
        return Hlist(toks[0])

    def start_group(self, s, loc, toks):
        self.push_state()
        # Deal with LaTeX-style font tokens
        if len(toks):
            self.get_state().font = toks[0][4:]
        return []

    def group(self, s, loc, toks):
        grp = Hlist(toks[0])
        return [grp]
    required_group = simple_group = group

    def end_group(self, s, loc, toks):
        self.pop_state()
        return []

    def font(self, s, loc, toks):
        assert len(toks) == 1
        name = toks[0]
        self.get_state().font = name
        return []

    def is_overunder(self, nucleus):
        if isinstance(nucleus, Char):
            return nucleus.c in self._overunder_symbols
        elif isinstance(nucleus, Hlist) and hasattr(nucleus, 'function_name'):
            return nucleus.function_name in self._overunder_functions
        return False

    def is_dropsub(self, nucleus):
        if isinstance(nucleus, Char):
            return nucleus.c in self._dropsub_symbols
        return False

    def is_slanted(self, nucleus):
        if isinstance(nucleus, Char):
            return nucleus.is_slanted()
        return False

    def is_between_brackets(self, s, loc):
        return False

    def subsuper(self, s, loc, toks):
        assert len(toks) == 1

        nucleus = None
        sub = None
        super = None

        # Pick all of the apostrophes out, including first apostrophes that
        # have been parsed as characters
        napostrophes = 0
        new_toks = []
        for tok in toks[0]:
            if isinstance(tok, str) and tok not in ('^', '_'):
                napostrophes += len(tok)
            elif isinstance(tok, Char) and tok.c == "'":
                napostrophes += 1
            else:
                new_toks.append(tok)
        toks = new_toks

        if len(toks) == 0:
            assert napostrophes
            nucleus = Hbox(0.0)
        elif len(toks) == 1:
            if not napostrophes:
                return toks[0]  # .asList()
            else:
                nucleus = toks[0]
        elif len(toks) in (2, 3):
            # single subscript or superscript
            nucleus = toks[0] if len(toks) == 3 else Hbox(0.0)
            op, next = toks[-2:]
            if op == '_':
                sub = next
            else:
                super = next
        elif len(toks) in (4, 5):
            # subscript and superscript
            nucleus = toks[0] if len(toks) == 5 else Hbox(0.0)
            op1, next1, op2, next2 = toks[-4:]
            if op1 == op2:
                if op1 == '_':
                    raise ParseFatalException("Double subscript")
                else:
                    raise ParseFatalException("Double superscript")
            if op1 == '_':
                sub = next1
                super = next2
            else:
                super = next1
                sub = next2
        else:
            raise ParseFatalException(
                "Subscript/superscript sequence is too long. "
                "Use braces { } to remove ambiguity.")

        state = self.get_state()
        rule_thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        xHeight = state.font_output.get_xheight(
            state.font, state.fontsize, state.dpi)

        if napostrophes:
            if super is None:
                super = Hlist([])
            for i in range(napostrophes):
                super.children.extend(self.symbol(s, loc, ['\\prime']))
            # kern() and hpack() needed to get the metrics right after
            # extending
            super.kern()
            super.hpack()

        # Handle over/under symbols, such as sum or integral
        if self.is_overunder(nucleus):
            vlist = []
            shift = 0.
            width = nucleus.width
            if super is not None:
                super.shrink()
                width = max(width, super.width)
            if sub is not None:
                sub.shrink()
                width = max(width, sub.width)

            if super is not None:
                hlist = HCentered([super])
                hlist.hpack(width, 'exactly')
                vlist.extend([hlist, Kern(rule_thickness * 3.0)])
            hlist = HCentered([nucleus])
            hlist.hpack(width, 'exactly')
            vlist.append(hlist)
            if sub is not None:
                hlist = HCentered([sub])
                hlist.hpack(width, 'exactly')
                vlist.extend([Kern(rule_thickness * 3.0), hlist])
                shift = hlist.height
            vlist = Vlist(vlist)
            vlist.shift_amount = shift + nucleus.depth
            result = Hlist([vlist])
            return [result]

        # We remove kerning on the last character for consistency (otherwise
        # it will compute kerning based on non-shrunk characters and may put
        # them too close together when superscripted)
        # We change the width of the last character to match the advance to
        # consider some fonts with weird metrics: e.g. stix's f has a width of
        # 7.75 and a kerning of -4.0 for an advance of 3.72, and we want to put
        # the superscript at the advance
        last_char = nucleus
        if isinstance(nucleus, Hlist):
            new_children = nucleus.children
            if len(new_children):
                # remove last kern
                if (isinstance(new_children[-1], Kern) and
                        hasattr(new_children[-2], '_metrics')):
                    new_children = new_children[:-1]
                last_char = new_children[-1]
                if hasattr(last_char, '_metrics'):
                    last_char.width = last_char._metrics.advance
            # create new Hlist without kerning
            nucleus = Hlist(new_children, do_kern=False)
        else:
            if isinstance(nucleus, Char):
                last_char.width = last_char._metrics.advance
            nucleus = Hlist([nucleus])

        # Handle regular sub/superscripts
        constants = _get_font_constant_set(state)
        lc_height   = last_char.height
        lc_baseline = 0
        if self.is_dropsub(last_char):
            lc_baseline = last_char.depth

        # Compute kerning for sub and super
        superkern = constants.delta * xHeight
        subkern = constants.delta * xHeight
        if self.is_slanted(last_char):
            superkern += constants.delta * xHeight
            superkern += (constants.delta_slanted *
                          (lc_height - xHeight * 2. / 3.))
            if self.is_dropsub(last_char):
                subkern = (3 * constants.delta -
                           constants.delta_integral) * lc_height
                superkern = (3 * constants.delta +
                             constants.delta_integral) * lc_height
            else:
                subkern = 0

        if super is None:
            # node757
            x = Hlist([Kern(subkern), sub])
            x.shrink()
            if self.is_dropsub(last_char):
                shift_down = lc_baseline + constants.subdrop * xHeight
            else:
                shift_down = constants.sub1 * xHeight
            x.shift_amount = shift_down
        else:
            x = Hlist([Kern(superkern), super])
            x.shrink()
            if self.is_dropsub(last_char):
                shift_up = lc_height - constants.subdrop * xHeight
            else:
                shift_up = constants.sup1 * xHeight
            if sub is None:
                x.shift_amount = -shift_up
            else:  # Both sub and superscript
                y = Hlist([Kern(subkern), sub])
                y.shrink()
                if self.is_dropsub(last_char):
                    shift_down = lc_baseline + constants.subdrop * xHeight
                else:
                    shift_down = constants.sub2 * xHeight
                # If sub and superscript collide, move super up
                clr = (2.0 * rule_thickness -
                       ((shift_up - x.depth) - (y.height - shift_down)))
                if clr > 0.:
                    shift_up += clr
                x = Vlist([
                    x,
                    Kern((shift_up - x.depth) - (y.height - shift_down)),
                    y])
                x.shift_amount = shift_down

        if not self.is_dropsub(last_char):
            x.width += constants.script_space * xHeight
        result = Hlist([nucleus, x])

        return [result]

    def _genfrac(self, ldelim, rdelim, rule, style, num, den):
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        rule = float(rule)

        # If style != displaystyle == 0, shrink the num and den
        if style != self._math_style_dict['displaystyle']:
            num.shrink()
            den.shrink()
        cnum = HCentered([num])
        cden = HCentered([den])
        width = max(num.width, den.width)
        cnum.hpack(width, 'exactly')
        cden.hpack(width, 'exactly')
        vlist = Vlist([cnum,                      # numerator
                       Vbox(0, thickness * 2.0),  # space
                       Hrule(state, rule),        # rule
                       Vbox(0, thickness * 2.0),  # space
                       cden                       # denominator
                       ])

        # Shift so the fraction line sits in the middle of the
        # equals sign
        metrics = state.font_output.get_metrics(
            state.font, rcParams['mathtext.default'],
            '=', state.fontsize, state.dpi)
        shift = (cden.height -
                 ((metrics.ymax + metrics.ymin) / 2 -
                  thickness * 3.0))
        vlist.shift_amount = shift

        result = [Hlist([vlist, Hbox(thickness * 2.)])]
        if ldelim or rdelim:
            if ldelim == '':
                ldelim = '.'
            if rdelim == '':
                rdelim = '.'
            return self._auto_sized_delimiter(ldelim, result, rdelim)
        return result

    def genfrac(self, s, loc, toks):
        assert len(toks) == 1
        assert len(toks[0]) == 6

        return self._genfrac(*tuple(toks[0]))

    def frac(self, s, loc, toks):
        assert len(toks) == 1
        assert len(toks[0]) == 2
        state = self.get_state()

        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        num, den = toks[0]

        return self._genfrac('', '', thickness,
                             self._math_style_dict['textstyle'], num, den)

    def dfrac(self, s, loc, toks):
        assert len(toks) == 1
        assert len(toks[0]) == 2
        state = self.get_state()

        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        num, den = toks[0]

        return self._genfrac('', '', thickness,
                             self._math_style_dict['displaystyle'], num, den)

    def binom(self, s, loc, toks):
        assert len(toks) == 1
        assert len(toks[0]) == 2
        num, den = toks[0]

        return self._genfrac('(', ')', 0.0,
                             self._math_style_dict['textstyle'], num, den)

    def sqrt(self, s, loc, toks):
        root, body = toks[0]
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        # Determine the height of the body, and add a little extra to
        # the height so it doesn't seem cramped
        height = body.height - body.shift_amount + thickness * 5.0
        depth = body.depth + body.shift_amount
        check = AutoHeightChar(r'\__sqrt__', height, depth, state, always=True)
        height = check.height - check.shift_amount
        depth = check.depth + check.shift_amount

        # Put a little extra space to the left and right of the body
        padded_body = Hlist([Hbox(2 * thickness), body, Hbox(2 * thickness)])
        rightside = Vlist([Hrule(state), Glue('fill'), padded_body])
        # Stretch the glue between the hrule and the body
        rightside.vpack(height + (state.fontsize * state.dpi) / (100.0 * 12.0),
                        'exactly', depth)

        # Add the root and shift it upward so it is above the tick.
        # The value of 0.6 is a hard-coded hack ;)
        if root is None:
            root = Box(check.width * 0.5, 0., 0.)
        else:
            root = Hlist([Char(x, state) for x in root])
            root.shrink()
            root.shrink()

        root_vlist = Vlist([Hlist([root])])
        root_vlist.shift_amount = -height * 0.6

        hlist = Hlist([root_vlist,               # Root
                       # Negative kerning to put root over tick
                       Kern(-check.width * 0.5),
                       check,                    # Check
                       rightside])               # Body
        return [hlist]

    def overline(self, s, loc, toks):
        assert len(toks) == 1
        assert len(toks[0]) == 1

        body = toks[0][0]

        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        height = body.height - body.shift_amount + thickness * 3.0
        depth = body.depth + body.shift_amount

        # Place overline above body
        rightside = Vlist([Hrule(state), Glue('fill'), Hlist([body])])

        # Stretch the glue between the hrule and the body
        rightside.vpack(height + (state.fontsize * state.dpi) / (100.0 * 12.0),
                        'exactly', depth)

        hlist = Hlist([rightside])
        return [hlist]

    def _auto_sized_delimiter(self, front, middle, back):
        state = self.get_state()
        if len(middle):
            height = max(x.height for x in middle)
            depth = max(x.depth for x in middle)
            factor = None
        else:
            height = 0
            depth = 0
            factor = 1.0
        parts = []
        # \left. and \right. aren't supposed to produce any symbols
        if front != '.':
            parts.append(
                AutoHeightChar(front, height, depth, state, factor=factor))
        parts.extend(middle)
        if back != '.':
            parts.append(
                AutoHeightChar(back, height, depth, state, factor=factor))
        hlist = Hlist(parts)
        return hlist

    def auto_delim(self, s, loc, toks):
        front, middle, back = toks

        return self._auto_sized_delimiter(front, middle.asList(), back)
=======
# Some convenient ways to get common kinds of glue


@cbook.deprecated("3.3", alternative="Glue('fil')")
class Fil(Glue):
    def __init__(self):
        Glue.__init__(self, 'fil')


@cbook.deprecated("3.3", alternative="Glue('fill')")
class Fill(Glue):
    def __init__(self):
        Glue.__init__(self, 'fill')


@cbook.deprecated("3.3", alternative="Glue('filll')")
class Filll(Glue):
    def __init__(self):
        Glue.__init__(self, 'filll')


@cbook.deprecated("3.3", alternative="Glue('neg_fil')")
class NegFil(Glue):
    def __init__(self):
        Glue.__init__(self, 'neg_fil')


@cbook.deprecated("3.3", alternative="Glue('neg_fill')")
class NegFill(Glue):
    def __init__(self):
        Glue.__init__(self, 'neg_fill')


@cbook.deprecated("3.3", alternative="Glue('neg_filll')")
class NegFilll(Glue):
    def __init__(self):
        Glue.__init__(self, 'neg_filll')


@cbook.deprecated("3.3", alternative="Glue('ss')")
class SsGlue(Glue):
    def __init__(self):
        Glue.__init__(self, 'ss')


class HCentered(Hlist):
    """
    A convenience class to create an `Hlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements):
        super().__init__([Glue('ss'), *elements, Glue('ss')], do_kern=False)


class VCentered(Vlist):
    """
    A convenience class to create a `Vlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements):
        super().__init__([Glue('ss'), *elements, Glue('ss')])


class Kern(Node):
    """
    A `Kern` node has a width field to specify a (normally
    negative) amount of spacing. This spacing correction appears in
    horizontal lists between letters like A and V when the font
    designer said that it looks better to move them closer together or
    further apart. A kern node can also appear in a vertical list,
    when its *width* denotes additional spacing in the vertical
    direction.
    """

    height = 0
    depth = 0

    def __init__(self, width):
        Node.__init__(self)
        self.width = width

    def __repr__(self):
        return "k%.02f" % self.width

    def shrink(self):
        Node.shrink(self)
        if self.size < NUM_SIZE_LEVELS:
            self.width *= SHRINK_FACTOR

    def grow(self):
        Node.grow(self)
        self.width *= GROW_FACTOR


class SubSuperCluster(Hlist):
    """
    A hack to get around that fact that this code does a two-pass parse like
    TeX.  This lets us store enough information in the hlist itself, namely the
    nucleus, sub- and super-script, such that if another script follows that
    needs to be attached, it can be reconfigured on the fly.
    """

    def __init__(self):
        self.nucleus = None
        self.sub = None
        self.super = None
        Hlist.__init__(self, [])


class AutoHeightChar(Hlist):
    """
    A character as close to the given height and depth as possible.

    When using a font with multiple height versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c, height, depth, state, always=False, factor=None):
        alternatives = state.font_output.get_sized_alternatives_for_symbol(
            state.font, c)

        xHeight = state.font_output.get_xheight(
            state.font, state.fontsize, state.dpi)

        state = state.copy()
        target_total = height + depth
        for fontname, sym in alternatives:
            state.font = fontname
            char = Char(sym, state)
            # Ensure that size 0 is chosen when the text is regular sized but
            # with descender glyphs by subtracting 0.2 * xHeight
            if char.height + char.depth >= target_total - 0.2 * xHeight:
                break

        shift = 0
        if state.font != 0:
            if factor is None:
                factor = target_total / (char.height + char.depth)
            state.fontsize *= factor
            char = Char(sym, state)

            shift = (depth - char.depth)

        Hlist.__init__(self, [char])
        self.shift_amount = shift


class AutoWidthChar(Hlist):
    """
    A character as close to the given width as possible.

    When using a font with multiple width versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c, width, state, always=False, char_class=Char):
        alternatives = state.font_output.get_sized_alternatives_for_symbol(
            state.font, c)

        state = state.copy()
        for fontname, sym in alternatives:
            state.font = fontname
            char = char_class(sym, state)
            if char.width >= width:
                break

        factor = width / char.width
        state.fontsize *= factor
        char = char_class(sym, state)

        Hlist.__init__(self, [char])
        self.width = char.width


class Ship:
    """
    Ship boxes to output once they have been set up, this sends them to output.

    Since boxes can be inside of boxes inside of boxes, the main work of `Ship`
    is done by two mutually recursive routines, `hlist_out` and `vlist_out`,
    which traverse the `Hlist` nodes and `Vlist` nodes inside of horizontal
    and vertical boxes.  The global variables used in TeX to store state as it
    processes have become member variables here.
    """

    def __call__(self, ox, oy, box):
        self.max_push    = 0  # Deepest nesting of push commands so far
        self.cur_s       = 0
        self.cur_v       = 0.
        self.cur_h       = 0.
        self.off_h       = ox
        self.off_v       = oy + box.height
        self.hlist_out(box)

    @staticmethod
    def clamp(value):
        if value < -1000000000.:
            return -1000000000.
        if value > 1000000000.:
            return 1000000000.
        return value

    def hlist_out(self, box):
        cur_g         = 0
        cur_glue      = 0.
        glue_order    = box.glue_order
        glue_sign     = box.glue_sign
        base_line     = self.cur_v
        left_edge     = self.cur_h
        self.cur_s    += 1
        self.max_push = max(self.cur_s, self.max_push)
        clamp         = self.clamp

        for p in box.children:
            if isinstance(p, Char):
                p.render(self.cur_h + self.off_h, self.cur_v + self.off_v)
                self.cur_h += p.width
            elif isinstance(p, Kern):
                self.cur_h += p.width
            elif isinstance(p, List):
                # node623
                if len(p.children) == 0:
                    self.cur_h += p.width
                else:
                    edge = self.cur_h
                    self.cur_v = base_line + p.shift_amount
                    if isinstance(p, Hlist):
                        self.hlist_out(p)
                    else:
                        # p.vpack(box.height + box.depth, 'exactly')
                        self.vlist_out(p)
                    self.cur_h = edge + p.width
                    self.cur_v = base_line
            elif isinstance(p, Box):
                # node624
                rule_height = p.height
                rule_depth  = p.depth
                rule_width  = p.width
                if np.isinf(rule_height):
                    rule_height = box.height
                if np.isinf(rule_depth):
                    rule_depth = box.depth
                if rule_height > 0 and rule_width > 0:
                    self.cur_v = base_line + rule_depth
                    p.render(self.cur_h + self.off_h,
                             self.cur_v + self.off_v,
                             rule_width, rule_height)
                    self.cur_v = base_line
                self.cur_h += rule_width
            elif isinstance(p, Glue):
                # node625
                glue_spec = p.glue_spec
                rule_width = glue_spec.width - cur_g
                if glue_sign != 0:  # normal
                    if glue_sign == 1:  # stretching
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch
                            cur_g = round(clamp(box.glue_set * cur_glue))
                    elif glue_spec.shrink_order == glue_order:
                        cur_glue += glue_spec.shrink
                        cur_g = round(clamp(box.glue_set * cur_glue))
                rule_width += cur_g
                self.cur_h += rule_width
        self.cur_s -= 1

    def vlist_out(self, box):
        cur_g         = 0
        cur_glue      = 0.
        glue_order    = box.glue_order
        glue_sign     = box.glue_sign
        self.cur_s    += 1
        self.max_push = max(self.max_push, self.cur_s)
        left_edge     = self.cur_h
        self.cur_v    -= box.height
        top_edge      = self.cur_v
        clamp         = self.clamp

        for p in box.children:
            if isinstance(p, Kern):
                self.cur_v += p.width
            elif isinstance(p, List):
                if len(p.children) == 0:
                    self.cur_v += p.height + p.depth
                else:
                    self.cur_v += p.height
                    self.cur_h = left_edge + p.shift_amount
                    save_v = self.cur_v
                    p.width = box.width
                    if isinstance(p, Hlist):
                        self.hlist_out(p)
                    else:
                        self.vlist_out(p)
                    self.cur_v = save_v + p.depth
                    self.cur_h = left_edge
            elif isinstance(p, Box):
                rule_height = p.height
                rule_depth = p.depth
                rule_width = p.width
                if np.isinf(rule_width):
                    rule_width = box.width
                rule_height += rule_depth
                if rule_height > 0 and rule_depth > 0:
                    self.cur_v += rule_height
                    p.render(self.cur_h + self.off_h,
                             self.cur_v + self.off_v,
                             rule_width, rule_height)
            elif isinstance(p, Glue):
                glue_spec = p.glue_spec
                rule_height = glue_spec.width - cur_g
                if glue_sign != 0:  # normal
                    if glue_sign == 1:  # stretching
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch
                            cur_g = round(clamp(box.glue_set * cur_glue))
                    elif glue_spec.shrink_order == glue_order:  # shrinking
                        cur_glue += glue_spec.shrink
                        cur_g = round(clamp(box.glue_set * cur_glue))
                rule_height += cur_g
                self.cur_v += rule_height
            elif isinstance(p, Char):
                raise RuntimeError(
                    "Internal mathtext error: Char node found in vlist")
        self.cur_s -= 1


ship = Ship()


##############################################################################
# PARSER


def Error(msg):
    """Helper class to raise parser errors."""
    def raise_error(s, loc, toks):
        raise ParseFatalException(s, loc, msg)

    empty = Empty()
    empty.setParseAction(raise_error)
    return empty


class Parser:
    """
    A pyparsing-based parser for strings containing math expressions.

    Raw text may also appear outside of pairs of ``$``.

    The grammar is based directly on that in TeX, though it cuts a few corners.
    """

    _math_style_dict = dict(displaystyle=0, textstyle=1,
                            scriptstyle=2, scriptscriptstyle=3)

    _binary_operators = set('''
      + * -
      \\pm             \\sqcap                   \\rhd
      \\mp             \\sqcup                   \\unlhd
      \\times          \\vee                     \\unrhd
      \\div            \\wedge                   \\oplus
      \\ast            \\setminus                \\ominus
      \\star           \\wr                      \\otimes
      \\circ           \\diamond                 \\oslash
      \\bullet         \\bigtriangleup           \\odot
      \\cdot           \\bigtriangledown         \\bigcirc
      \\cap            \\triangleleft            \\dagger
      \\cup            \\triangleright           \\ddagger
      \\uplus          \\lhd                     \\amalg'''.split())

    _relation_symbols = set('''
      = < > :
      \\leq        \\geq        \\equiv   \\models
      \\prec       \\succ       \\sim     \\perp
      \\preceq     \\succeq     \\simeq   \\mid
      \\ll         \\gg         \\asymp   \\parallel
      \\subset     \\supset     \\approx  \\bowtie
      \\subseteq   \\supseteq   \\cong    \\Join
      \\sqsubset   \\sqsupset   \\neq     \\smile
      \\sqsubseteq \\sqsupseteq \\doteq   \\frown
      \\in         \\ni         \\propto  \\vdash
      \\dashv      \\dots       \\dotplus \\doteqdot'''.split())

    _arrow_symbols = set('''
      \\leftarrow              \\longleftarrow           \\uparrow
      \\Leftarrow              \\Longleftarrow           \\Uparrow
      \\rightarrow             \\longrightarrow          \\downarrow
      \\Rightarrow             \\Longrightarrow          \\Downarrow
      \\leftrightarrow         \\longleftrightarrow      \\updownarrow
      \\Leftrightarrow         \\Longleftrightarrow      \\Updownarrow
      \\mapsto                 \\longmapsto              \\nearrow
      \\hookleftarrow          \\hookrightarrow          \\searrow
      \\leftharpoonup          \\rightharpoonup          \\swarrow
      \\leftharpoondown        \\rightharpoondown        \\nwarrow
      \\rightleftharpoons      \\leadsto'''.split())

    _spaced_symbols = _binary_operators | _relation_symbols | _arrow_symbols

    _punctuation_symbols = set(r', ; . ! \ldotp \cdotp'.split())

    _overunder_symbols = set(r'''
       \sum \prod \coprod \bigcap \bigcup \bigsqcup \bigvee
       \bigwedge \bigodot \bigotimes \bigoplus \biguplus
       '''.split())

    _overunder_functions = set(
        "lim liminf limsup sup max min".split())

    _dropsub_symbols = set(r'''\int \oint'''.split())

    _fontnames = set("rm cal it tt sf bf default bb frak scr regular".split())

    _function_names = set("""
      arccos csc ker min arcsin deg lg Pr arctan det lim sec arg dim
      liminf sin cos exp limsup sinh cosh gcd ln sup cot hom log tan
      coth inf max tanh""".split())

    _ambi_delim = set("""
      | \\| / \\backslash \\uparrow \\downarrow \\updownarrow \\Uparrow
      \\Downarrow \\Updownarrow . \\vert \\Vert \\\\|""".split())

    _left_delim = set(r"( [ \{ < \lfloor \langle \lceil".split())

    _right_delim = set(r") ] \} > \rfloor \rangle \rceil".split())

    def __init__(self):
        p = types.SimpleNamespace()
        # All forward declarations are here
        p.accent           = Forward()
        p.ambi_delim       = Forward()
        p.apostrophe       = Forward()
        p.auto_delim       = Forward()
        p.binom            = Forward()
        p.bslash           = Forward()
        p.c_over_c         = Forward()
        p.customspace      = Forward()
        p.end_group        = Forward()
        p.float_literal    = Forward()
        p.font             = Forward()
        p.frac             = Forward()
        p.dfrac            = Forward()
        p.function         = Forward()
        p.genfrac          = Forward()
        p.group            = Forward()
        p.int_literal      = Forward()
        p.latexfont        = Forward()
        p.lbracket         = Forward()
        p.left_delim       = Forward()
        p.lbrace           = Forward()
        p.main             = Forward()
        p.math             = Forward()
        p.math_string      = Forward()
        p.non_math         = Forward()
        p.operatorname     = Forward()
        p.overline         = Forward()
        p.placeable        = Forward()
        p.rbrace           = Forward()
        p.rbracket         = Forward()
        p.required_group   = Forward()
        p.right_delim      = Forward()
        p.right_delim_safe = Forward()
        p.simple           = Forward()
        p.simple_group     = Forward()
        p.single_symbol    = Forward()
        p.accentprefixed   = Forward()
        p.space            = Forward()
        p.sqrt             = Forward()
        p.start_group      = Forward()
        p.subsuper         = Forward()
        p.subsuperop       = Forward()
        p.symbol           = Forward()
        p.symbol_name      = Forward()
        p.token            = Forward()
        p.unknown_symbol   = Forward()

        # Set names on everything -- very useful for debugging
        for key, val in vars(p).items():
            if not key.startswith('_'):
                val.setName(key)

        p.float_literal <<= Regex(r"[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)")
        p.int_literal   <<= Regex("[-+]?[0-9]+")

        p.lbrace        <<= Literal('{').suppress()
        p.rbrace        <<= Literal('}').suppress()
        p.lbracket      <<= Literal('[').suppress()
        p.rbracket      <<= Literal(']').suppress()
        p.bslash        <<= Literal('\\')

        p.space         <<= oneOf(list(self._space_widths))
        p.customspace   <<= (
            Suppress(Literal(r'\hspace'))
            - ((p.lbrace + p.float_literal + p.rbrace)
               | Error(r"Expected \hspace{n}"))
        )

        unicode_range = "\U00000080-\U0001ffff"
        p.single_symbol <<= Regex(
            r"([a-zA-Z0-9 +\-*/<>=:,.;!\?&'@()\[\]|%s])|(\\[%%${}\[\]_|])" %
            unicode_range)
        p.accentprefixed <<= Suppress(p.bslash) + oneOf(self._accentprefixed)
        p.symbol_name   <<= (
            Combine(p.bslash + oneOf(list(tex2uni)))
            + FollowedBy(Regex("[^A-Za-z]").leaveWhitespace() | StringEnd())
        )
        p.symbol        <<= (p.single_symbol | p.symbol_name).leaveWhitespace()

        p.apostrophe    <<= Regex("'+")

        p.c_over_c      <<= (
            Suppress(p.bslash)
            + oneOf(list(self._char_over_chars))
        )

        p.accent        <<= Group(
            Suppress(p.bslash)
            + oneOf([*self._accent_map, *self._wide_accents])
            - p.placeable
        )

        p.function      <<= (
            Suppress(p.bslash)
            + oneOf(list(self._function_names))
        )

        p.start_group    <<= Optional(p.latexfont) + p.lbrace
        p.end_group      <<= p.rbrace.copy()
        p.simple_group   <<= Group(p.lbrace + ZeroOrMore(p.token) + p.rbrace)
        p.required_group <<= Group(p.lbrace + OneOrMore(p.token) + p.rbrace)
        p.group          <<= Group(
            p.start_group + ZeroOrMore(p.token) + p.end_group
        )

        p.font          <<= Suppress(p.bslash) + oneOf(list(self._fontnames))
        p.latexfont     <<= (
            Suppress(p.bslash)
            + oneOf(['math' + x for x in self._fontnames])
        )

        p.frac          <<= Group(
            Suppress(Literal(r"\frac"))
            - ((p.required_group + p.required_group)
               | Error(r"Expected \frac{num}{den}"))
        )

        p.dfrac         <<= Group(
            Suppress(Literal(r"\dfrac"))
            - ((p.required_group + p.required_group)
               | Error(r"Expected \dfrac{num}{den}"))
        )

        p.binom         <<= Group(
            Suppress(Literal(r"\binom"))
            - ((p.required_group + p.required_group)
               | Error(r"Expected \binom{num}{den}"))
        )

        p.ambi_delim    <<= oneOf(list(self._ambi_delim))
        p.left_delim    <<= oneOf(list(self._left_delim))
        p.right_delim   <<= oneOf(list(self._right_delim))
        p.right_delim_safe <<= oneOf([*(self._right_delim - {'}'}), r'\}'])

        p.genfrac <<= Group(
            Suppress(Literal(r"\genfrac"))
            - (((p.lbrace
                 + Optional(p.ambi_delim | p.left_delim, default='')
                 + p.rbrace)
                + (p.lbrace
                   + Optional(p.ambi_delim | p.right_delim_safe, default='')
                   + p.rbrace)
                + (p.lbrace + p.float_literal + p.rbrace)
                + p.simple_group + p.required_group + p.required_group)
               | Error("Expected "
                       r"\genfrac{ldelim}{rdelim}{rulesize}{style}{num}{den}"))
        )

        p.sqrt <<= Group(
            Suppress(Literal(r"\sqrt"))
            - ((Optional(p.lbracket + p.int_literal + p.rbracket, default=None)
                + p.required_group)
               | Error("Expected \\sqrt{value}"))
        )

        p.overline <<= Group(
            Suppress(Literal(r"\overline"))
            - (p.required_group | Error("Expected \\overline{value}"))
        )

        p.unknown_symbol <<= Combine(p.bslash + Regex("[A-Za-z]*"))

        p.operatorname <<= Group(
            Suppress(Literal(r"\operatorname"))
            - ((p.lbrace + ZeroOrMore(p.simple | p.unknown_symbol) + p.rbrace)
               | Error("Expected \\operatorname{value}"))
        )

        p.placeable     <<= (
            p.accentprefixed  # Must be before accent so named symbols that are
                              # prefixed with an accent name work
            | p.accent   # Must be before symbol as all accents are symbols
            | p.symbol   # Must be third to catch all named symbols and single
                         # chars not in a group
            | p.c_over_c
            | p.function
            | p.group
            | p.frac
            | p.dfrac
            | p.binom
            | p.genfrac
            | p.sqrt
            | p.overline
            | p.operatorname
        )

        p.simple        <<= (
            p.space
            | p.customspace
            | p.font
            | p.subsuper
        )

        p.subsuperop    <<= oneOf(["_", "^"])

        p.subsuper      <<= Group(
            (Optional(p.placeable)
             + OneOrMore(p.subsuperop - p.placeable)
             + Optional(p.apostrophe))
            | (p.placeable + Optional(p.apostrophe))
            | p.apostrophe
        )

        p.token         <<= (
            p.simple
            | p.auto_delim
            | p.unknown_symbol  # Must be last
        )

        p.auto_delim    <<= (
            Suppress(Literal(r"\left"))
            - ((p.left_delim | p.ambi_delim)
               | Error("Expected a delimiter"))
            + Group(ZeroOrMore(p.simple | p.auto_delim))
            + Suppress(Literal(r"\right"))
            - ((p.right_delim | p.ambi_delim)
               | Error("Expected a delimiter"))
        )

        p.math          <<= OneOrMore(p.token)

        p.math_string   <<= QuotedString('$', '\\', unquoteResults=False)

        p.non_math      <<= Regex(r"(?:(?:\\[$])|[^$])*").leaveWhitespace()

        p.main          <<= (
            p.non_math + ZeroOrMore(p.math_string + p.non_math) + StringEnd()
        )

        # Set actions
        for key, val in vars(p).items():
            if not key.startswith('_'):
                if hasattr(self, key):
                    val.setParseAction(getattr(self, key))

        self._expression = p.main
        self._math_expression = p.math

    def parse(self, s, fonts_object, fontsize, dpi):
        """
        Parse expression *s* using the given *fonts_object* for
        output, at the given *fontsize* and *dpi*.

        Returns the parse tree of `Node` instances.
        """
        self._state_stack = [
            self.State(fonts_object, 'default', 'rm', fontsize, dpi)]
        self._em_width_cache = {}
        try:
            result = self._expression.parseString(s)
        except ParseBaseException as err:
            raise ValueError("\n".join(["",
                                        err.line,
                                        " " * (err.column - 1) + "^",
                                        str(err)])) from err
        self._state_stack = None
        self._em_width_cache = {}
        self._expression.resetCache()
        return result[0]

    # The state of the parser is maintained in a stack.  Upon
    # entering and leaving a group { } or math/non-math, the stack
    # is pushed and popped accordingly.  The current state always
    # exists in the top element of the stack.
    class State:
        """
        Stores the state of the parser.

        States are pushed and popped from a stack as necessary, and
        the "current" state is always at the top of the stack.
        """
        def __init__(self, font_output, font, font_class, fontsize, dpi):
            self.font_output = font_output
            self._font = font
            self.font_class = font_class
            self.fontsize = fontsize
            self.dpi = dpi

        def copy(self):
            return Parser.State(
                self.font_output,
                self.font,
                self.font_class,
                self.fontsize,
                self.dpi)

        @property
        def font(self):
            return self._font

        @font.setter
        def font(self, name):
            if name in ('rm', 'it', 'bf'):
                self.font_class = name
            self._font = name

    def get_state(self):
        """Get the current `State` of the parser."""
        return self._state_stack[-1]

    def pop_state(self):
        """Pop a `State` off of the stack."""
        self._state_stack.pop()

    def push_state(self):
        """Push a new `State` onto the stack, copying the current state."""
        self._state_stack.append(self.get_state().copy())

    def main(self, s, loc, toks):
        return [Hlist(toks)]

    def math_string(self, s, loc, toks):
        return self._math_expression.parseString(toks[0][1:-1])

    def math(self, s, loc, toks):
        hlist = Hlist(toks)
        self.pop_state()
        return [hlist]

    def non_math(self, s, loc, toks):
        s = toks[0].replace(r'\$', '$')
        symbols = [Char(c, self.get_state(), math=False) for c in s]
        hlist = Hlist(symbols)
        # We're going into math now, so set font to 'it'
        self.push_state()
        self.get_state().font = rcParams['mathtext.default']
        return [hlist]

    def _make_space(self, percentage):
        # All spaces are relative to em width
        state = self.get_state()
        key = (state.font, state.fontsize, state.dpi)
        width = self._em_width_cache.get(key)
        if width is None:
            metrics = state.font_output.get_metrics(
                state.font, rcParams['mathtext.default'], 'm', state.fontsize,
                state.dpi)
            width = metrics.advance
            self._em_width_cache[key] = width
        return Kern(width * percentage)

    _space_widths = {
        r'\,':         0.16667,   # 3/18 em = 3 mu
        r'\thinspace': 0.16667,   # 3/18 em = 3 mu
        r'\/':         0.16667,   # 3/18 em = 3 mu
        r'\>':         0.22222,   # 4/18 em = 4 mu
        r'\:':         0.22222,   # 4/18 em = 4 mu
        r'\;':         0.27778,   # 5/18 em = 5 mu
        r'\ ':         0.33333,   # 6/18 em = 6 mu
        r'~':          0.33333,   # 6/18 em = 6 mu, nonbreakable
        r'\enspace':   0.5,       # 9/18 em = 9 mu
        r'\quad':      1,         # 1 em = 18 mu
        r'\qquad':     2,         # 2 em = 36 mu
        r'\!':         -0.16667,  # -3/18 em = -3 mu
    }

    def space(self, s, loc, toks):
        assert len(toks) == 1
        num = self._space_widths[toks[0]]
        box = self._make_space(num)
        return [box]

    def customspace(self, s, loc, toks):
        return [self._make_space(float(toks[0]))]

    def symbol(self, s, loc, toks):
        c = toks[0]
        try:
            char = Char(c, self.get_state())
        except ValueError as err:
            raise ParseFatalException(s, loc,
                                      "Unknown symbol: %s" % c) from err

        if c in self._spaced_symbols:
            # iterate until we find previous character, needed for cases
            # such as ${ -2}$, $ -2$, or $   -2$.
            prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
            # Binary operators at start of string should not be spaced
            if (c in self._binary_operators and
                    (len(s[:loc].split()) == 0 or prev_char == '{' or
                     prev_char in self._left_delim)):
                return [char]
            else:
                return [Hlist([self._make_space(0.2),
                               char,
                               self._make_space(0.2)],
                              do_kern=True)]
        elif c in self._punctuation_symbols:

            # Do not space commas between brackets
            if c == ',':
                prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
                next_char = next((c for c in s[loc + 1:] if c != ' '), '')
                if prev_char == '{' and next_char == '}':
                    return [char]

            # Do not space dots as decimal separators
            if c == '.' and s[loc - 1].isdigit() and s[loc + 1].isdigit():
                return [char]
            else:
                return [Hlist([char, self._make_space(0.2)], do_kern=True)]
        return [char]

    accentprefixed = symbol

    def unknown_symbol(self, s, loc, toks):
        c = toks[0]
        raise ParseFatalException(s, loc, "Unknown symbol: %s" % c)

    _char_over_chars = {
        # The first 2 entries in the tuple are (font, char, sizescale) for
        # the two symbols under and over.  The third element is the space
        # (in multiples of underline height)
        r'AA': (('it', 'A', 1.0), (None, '\\circ', 0.5), 0.0),
    }

    def c_over_c(self, s, loc, toks):
        sym = toks[0]
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        under_desc, over_desc, space = \
            self._char_over_chars.get(sym, (None, None, 0.0))
        if under_desc is None:
            raise ParseFatalException("Error parsing symbol")

        over_state = state.copy()
        if over_desc[0] is not None:
            over_state.font = over_desc[0]
        over_state.fontsize *= over_desc[2]
        over = Accent(over_desc[1], over_state)

        under_state = state.copy()
        if under_desc[0] is not None:
            under_state.font = under_desc[0]
        under_state.fontsize *= under_desc[2]
        under = Char(under_desc[1], under_state)

        width = max(over.width, under.width)

        over_centered = HCentered([over])
        over_centered.hpack(width, 'exactly')

        under_centered = HCentered([under])
        under_centered.hpack(width, 'exactly')

        return Vlist([
                over_centered,
                Vbox(0., thickness * space),
                under_centered
                ])

    _accent_map = {
        r'hat':            r'\circumflexaccent',
        r'breve':          r'\combiningbreve',
        r'bar':            r'\combiningoverline',
        r'grave':          r'\combininggraveaccent',
        r'acute':          r'\combiningacuteaccent',
        r'tilde':          r'\combiningtilde',
        r'dot':            r'\combiningdotabove',
        r'ddot':           r'\combiningdiaeresis',
        r'vec':            r'\combiningrightarrowabove',
        r'"':              r'\combiningdiaeresis',
        r"`":              r'\combininggraveaccent',
        r"'":              r'\combiningacuteaccent',
        r'~':              r'\combiningtilde',
        r'.':              r'\combiningdotabove',
        r'^':              r'\circumflexaccent',
        r'overrightarrow': r'\rightarrow',
        r'overleftarrow':  r'\leftarrow',
        r'mathring':       r'\circ',
    }

    _wide_accents = set(r"widehat widetilde widebar".split())

    # make a lambda and call it to get the namespace right
    _accentprefixed = (lambda am: [
        p for p in tex2uni
        if any(p.startswith(a) and a != p for a in am)
    ])(set(_accent_map))

    def accent(self, s, loc, toks):
        assert len(toks) == 1
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        if len(toks[0]) != 2:
            raise ParseFatalException("Error parsing accent")
        accent, sym = toks[0]
        if accent in self._wide_accents:
            accent_box = AutoWidthChar(
                '\\' + accent, sym.width, state, char_class=Accent)
        else:
            accent_box = Accent(self._accent_map[accent], state)
        if accent == 'mathring':
            accent_box.shrink()
            accent_box.shrink()
        centered = HCentered([Hbox(sym.width / 4.0), accent_box])
        centered.hpack(sym.width, 'exactly')
        return Vlist([
                centered,
                Vbox(0., thickness * 2.0),
                Hlist([sym])
                ])

    def function(self, s, loc, toks):
        self.push_state()
        state = self.get_state()
        state.font = 'rm'
        hlist = Hlist([Char(c, state) for c in toks[0]])
        self.pop_state()
        hlist.function_name = toks[0]
        return hlist

    def operatorname(self, s, loc, toks):
        self.push_state()
        state = self.get_state()
        state.font = 'rm'
        # Change the font of Chars, but leave Kerns alone
        for c in toks[0]:
            if isinstance(c, Char):
                c.font = 'rm'
                c._update_metrics()
        self.pop_state()
        return Hlist(toks[0])

    def start_group(self, s, loc, toks):
        self.push_state()
        # Deal with LaTeX-style font tokens
        if len(toks):
            self.get_state().font = toks[0][4:]
        return []

    def group(self, s, loc, toks):
        grp = Hlist(toks[0])
        return [grp]
    required_group = simple_group = group

    def end_group(self, s, loc, toks):
        self.pop_state()
        return []

    def font(self, s, loc, toks):
        assert len(toks) == 1
        name = toks[0]
        self.get_state().font = name
        return []

    def is_overunder(self, nucleus):
        if isinstance(nucleus, Char):
            return nucleus.c in self._overunder_symbols
        elif isinstance(nucleus, Hlist) and hasattr(nucleus, 'function_name'):
            return nucleus.function_name in self._overunder_functions
        return False

    def is_dropsub(self, nucleus):
        if isinstance(nucleus, Char):
            return nucleus.c in self._dropsub_symbols
        return False

    def is_slanted(self, nucleus):
        if isinstance(nucleus, Char):
            return nucleus.is_slanted()
        return False

    def is_between_brackets(self, s, loc):
        return False

    def subsuper(self, s, loc, toks):
        assert len(toks) == 1

        nucleus = None
        sub = None
        super = None

        # Pick all of the apostrophes out, including first apostrophes that
        # have been parsed as characters
        napostrophes = 0
        new_toks = []
        for tok in toks[0]:
            if isinstance(tok, str) and tok not in ('^', '_'):
                napostrophes += len(tok)
            elif isinstance(tok, Char) and tok.c == "'":
                napostrophes += 1
            else:
                new_toks.append(tok)
        toks = new_toks

        if len(toks) == 0:
            assert napostrophes
            nucleus = Hbox(0.0)
        elif len(toks) == 1:
            if not napostrophes:
                return toks[0]  # .asList()
            else:
                nucleus = toks[0]
        elif len(toks) in (2, 3):
            # single subscript or superscript
            nucleus = toks[0] if len(toks) == 3 else Hbox(0.0)
            op, next = toks[-2:]
            if op == '_':
                sub = next
            else:
                super = next
        elif len(toks) in (4, 5):
            # subscript and superscript
            nucleus = toks[0] if len(toks) == 5 else Hbox(0.0)
            op1, next1, op2, next2 = toks[-4:]
            if op1 == op2:
                if op1 == '_':
                    raise ParseFatalException("Double subscript")
                else:
                    raise ParseFatalException("Double superscript")
            if op1 == '_':
                sub = next1
                super = next2
            else:
                super = next1
                sub = next2
        else:
            raise ParseFatalException(
                "Subscript/superscript sequence is too long. "
                "Use braces { } to remove ambiguity.")

        state = self.get_state()
        rule_thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        xHeight = state.font_output.get_xheight(
            state.font, state.fontsize, state.dpi)

        if napostrophes:
            if super is None:
                super = Hlist([])
            for i in range(napostrophes):
                super.children.extend(self.symbol(s, loc, ['\\prime']))
            # kern() and hpack() needed to get the metrics right after
            # extending
            super.kern()
            super.hpack()

        # Handle over/under symbols, such as sum or integral
        if self.is_overunder(nucleus):
            vlist = []
            shift = 0.
            width = nucleus.width
            if super is not None:
                super.shrink()
                width = max(width, super.width)
            if sub is not None:
                sub.shrink()
                width = max(width, sub.width)

            if super is not None:
                hlist = HCentered([super])
                hlist.hpack(width, 'exactly')
                vlist.extend([hlist, Kern(rule_thickness * 3.0)])
            hlist = HCentered([nucleus])
            hlist.hpack(width, 'exactly')
            vlist.append(hlist)
            if sub is not None:
                hlist = HCentered([sub])
                hlist.hpack(width, 'exactly')
                vlist.extend([Kern(rule_thickness * 3.0), hlist])
                shift = hlist.height
            vlist = Vlist(vlist)
            vlist.shift_amount = shift + nucleus.depth
            result = Hlist([vlist])
            return [result]

        # We remove kerning on the last character for consistency (otherwise
        # it will compute kerning based on non-shrunk characters and may put
        # them too close together when superscripted)
        # We change the width of the last character to match the advance to
        # consider some fonts with weird metrics: e.g. stix's f has a width of
        # 7.75 and a kerning of -4.0 for an advance of 3.72, and we want to put
        # the superscript at the advance
        last_char = nucleus
        if isinstance(nucleus, Hlist):
            new_children = nucleus.children
            if len(new_children):
                # remove last kern
                if (isinstance(new_children[-1], Kern) and
                        hasattr(new_children[-2], '_metrics')):
                    new_children = new_children[:-1]
                last_char = new_children[-1]
                if hasattr(last_char, '_metrics'):
                    last_char.width = last_char._metrics.advance
            # create new Hlist without kerning
            nucleus = Hlist(new_children, do_kern=False)
        else:
            if isinstance(nucleus, Char):
                last_char.width = last_char._metrics.advance
            nucleus = Hlist([nucleus])

        # Handle regular sub/superscripts
        constants = _get_font_constant_set(state)
        lc_height   = last_char.height
        lc_baseline = 0
        if self.is_dropsub(last_char):
            lc_baseline = last_char.depth

        # Compute kerning for sub and super
        superkern = constants.delta * xHeight
        subkern = constants.delta * xHeight
        if self.is_slanted(last_char):
            superkern += constants.delta * xHeight
            superkern += (constants.delta_slanted *
                          (lc_height - xHeight * 2. / 3.))
            if self.is_dropsub(last_char):
                subkern = (3 * constants.delta -
                           constants.delta_integral) * lc_height
                superkern = (3 * constants.delta +
                             constants.delta_integral) * lc_height
            else:
                subkern = 0

        if super is None:
            # node757
            x = Hlist([Kern(subkern), sub])
            x.shrink()
            if self.is_dropsub(last_char):
                shift_down = lc_baseline + constants.subdrop * xHeight
            else:
                shift_down = constants.sub1 * xHeight
            x.shift_amount = shift_down
        else:
            x = Hlist([Kern(superkern), super])
            x.shrink()
            if self.is_dropsub(last_char):
                shift_up = lc_height - constants.subdrop * xHeight
            else:
                shift_up = constants.sup1 * xHeight
            if sub is None:
                x.shift_amount = -shift_up
            else:  # Both sub and superscript
                y = Hlist([Kern(subkern), sub])
                y.shrink()
                if self.is_dropsub(last_char):
                    shift_down = lc_baseline + constants.subdrop * xHeight
                else:
                    shift_down = constants.sub2 * xHeight
                # If sub and superscript collide, move super up
                clr = (2.0 * rule_thickness -
                       ((shift_up - x.depth) - (y.height - shift_down)))
                if clr > 0.:
                    shift_up += clr
                x = Vlist([
                    x,
                    Kern((shift_up - x.depth) - (y.height - shift_down)),
                    y])
                x.shift_amount = shift_down

        if not self.is_dropsub(last_char):
            x.width += constants.script_space * xHeight
        result = Hlist([nucleus, x])

        return [result]

    def _genfrac(self, ldelim, rdelim, rule, style, num, den):
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        rule = float(rule)

        # If style != displaystyle == 0, shrink the num and den
        if style != self._math_style_dict['displaystyle']:
            num.shrink()
            den.shrink()
        cnum = HCentered([num])
        cden = HCentered([den])
        width = max(num.width, den.width)
        cnum.hpack(width, 'exactly')
        cden.hpack(width, 'exactly')
        vlist = Vlist([cnum,                      # numerator
                       Vbox(0, thickness * 2.0),  # space
                       Hrule(state, rule),        # rule
                       Vbox(0, thickness * 2.0),  # space
                       cden                       # denominator
                       ])

        # Shift so the fraction line sits in the middle of the
        # equals sign
        metrics = state.font_output.get_metrics(
            state.font, rcParams['mathtext.default'],
            '=', state.fontsize, state.dpi)
        shift = (cden.height -
                 ((metrics.ymax + metrics.ymin) / 2 -
                  thickness * 3.0))
        vlist.shift_amount = shift

        result = [Hlist([vlist, Hbox(thickness * 2.)])]
        if ldelim or rdelim:
            if ldelim == '':
                ldelim = '.'
            if rdelim == '':
                rdelim = '.'
            return self._auto_sized_delimiter(ldelim, result, rdelim)
        return result

    def genfrac(self, s, loc, toks):
        assert len(toks) == 1
        assert len(toks[0]) == 6

        return self._genfrac(*tuple(toks[0]))

    def frac(self, s, loc, toks):
        assert len(toks) == 1
        assert len(toks[0]) == 2
        state = self.get_state()

        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        num, den = toks[0]

        return self._genfrac('', '', thickness,
                             self._math_style_dict['textstyle'], num, den)

    def dfrac(self, s, loc, toks):
        assert len(toks) == 1
        assert len(toks[0]) == 2
        state = self.get_state()

        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)
        num, den = toks[0]

        return self._genfrac('', '', thickness,
                             self._math_style_dict['displaystyle'], num, den)

    def binom(self, s, loc, toks):
        assert len(toks) == 1
        assert len(toks[0]) == 2
        num, den = toks[0]

        return self._genfrac('(', ')', 0.0,
                             self._math_style_dict['textstyle'], num, den)

    def sqrt(self, s, loc, toks):
        root, body = toks[0]
        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        # Determine the height of the body, and add a little extra to
        # the height so it doesn't seem cramped
        height = body.height - body.shift_amount + thickness * 5.0
        depth = body.depth + body.shift_amount
        check = AutoHeightChar(r'\__sqrt__', height, depth, state, always=True)
        height = check.height - check.shift_amount
        depth = check.depth + check.shift_amount

        # Put a little extra space to the left and right of the body
        padded_body = Hlist([Hbox(2 * thickness), body, Hbox(2 * thickness)])
        rightside = Vlist([Hrule(state), Glue('fill'), padded_body])
        # Stretch the glue between the hrule and the body
        rightside.vpack(height + (state.fontsize * state.dpi) / (100.0 * 12.0),
                        'exactly', depth)

        # Add the root and shift it upward so it is above the tick.
        # The value of 0.6 is a hard-coded hack ;)
        if root is None:
            root = Box(check.width * 0.5, 0., 0.)
        else:
            root = Hlist([Char(x, state) for x in root])
            root.shrink()
            root.shrink()

        root_vlist = Vlist([Hlist([root])])
        root_vlist.shift_amount = -height * 0.6

        hlist = Hlist([root_vlist,               # Root
                       # Negative kerning to put root over tick
                       Kern(-check.width * 0.5),
                       check,                    # Check
                       rightside])               # Body
        return [hlist]

    def overline(self, s, loc, toks):
        assert len(toks) == 1
        assert len(toks[0]) == 1

        body = toks[0][0]

        state = self.get_state()
        thickness = state.font_output.get_underline_thickness(
            state.font, state.fontsize, state.dpi)

        height = body.height - body.shift_amount + thickness * 3.0
        depth = body.depth + body.shift_amount

        # Place overline above body
        rightside = Vlist([Hrule(state), Glue('fill'), Hlist([body])])

        # Stretch the glue between the hrule and the body
        rightside.vpack(height + (state.fontsize * state.dpi) / (100.0 * 12.0),
                        'exactly', depth)

        hlist = Hlist([rightside])
        return [hlist]

    def _auto_sized_delimiter(self, front, middle, back):
        state = self.get_state()
        if len(middle):
            height = max(x.height for x in middle)
            depth = max(x.depth for x in middle)
            factor = None
        else:
            height = 0
            depth = 0
            factor = 1.0
        parts = []
        # \left. and \right. aren't supposed to produce any symbols
        if front != '.':
            parts.append(
                AutoHeightChar(front, height, depth, state, factor=factor))
        parts.extend(middle)
        if back != '.':
            parts.append(
                AutoHeightChar(back, height, depth, state, factor=factor))
        hlist = Hlist(parts)
        return hlist

    def auto_delim(self, s, loc, toks):
        front, middle, back = toks

        return self._auto_sized_delimiter(front, middle.asList(), back)
>>>>>>> /home/ze/miningframework/mining_results/matplotlib_results/matplotlib/96274e6315ced46aa72f4f12993166847c26e4b0/lib/matplotlib/mathtext.py/right.py


##############################################################################
# MAIN


class MathTextParser:
    _parser = None

    _backend_mapping = {
        'bitmap': MathtextBackendBitmap,
        'agg':    MathtextBackendAgg,
        'ps':     MathtextBackendPs,
        'pdf':    MathtextBackendPdf,
        'svg':    MathtextBackendSvg,
        'path':   MathtextBackendPath,
        'cairo':  MathtextBackendCairo,
        'macosx': MathtextBackendAgg,
    }
    _font_type_mapping = {
        'cm':          _mathtext.BakomaFonts,
        'dejavuserif': _mathtext.DejaVuSerifFonts,
        'dejavusans':  _mathtext.DejaVuSansFonts,
        'stix':        _mathtext.StixFonts,
        'stixsans':    _mathtext.StixSansFonts,
        'custom':      _mathtext.UnicodeFonts,
    }

    def __init__(self, output):
        """Create a MathTextParser for the given backend *output*."""
        self._output = output.lower()

    def parse(self, s, dpi=72, prop=None, *, _force_standard_ps_fonts=False):
        """
        Parse the given math expression *s* at the given *dpi*.  If *prop* is
        provided, it is a `.FontProperties` object specifying the "default"
        font to use in the math expression, used for all non-math text.

        The results are cached, so multiple calls to `parse`
        with the same expression should be fast.
        """
        if _force_standard_ps_fonts:
            _api.warn_deprecated(
                "3.4",
                removal="3.5",
                message=(
                    "Mathtext using only standard PostScript fonts has "
                    "been likely to produce wrong output for a while, "
                    "has been deprecated in %(since)s and will be removed "
                    "in %(removal)s, after which ps.useafm will have no "
                    "effect on mathtext."
                )
            )

        # lru_cache can't decorate parse() directly because the ps.useafm and
        # mathtext.fontset rcParams also affect the parse (e.g. by affecting
        # the glyph metrics).
        return self._parse_cached(s, dpi, prop, _force_standard_ps_fonts)

    @functools.lru_cache(50)
    def _parse_cached(self, s, dpi, prop, force_standard_ps_fonts):
        if prop is None:
            prop = FontProperties()

        fontset_class = (
            _mathtext.StandardPsFonts if force_standard_ps_fonts
            else _api.check_getitem(
                self._font_type_mapping, fontset=prop.get_math_fontfamily()))
        backend = self._backend_mapping[self._output]()
        font_output = fontset_class(prop, backend)

        fontsize = prop.get_size_in_points()

        # This is a class variable so we don't rebuild the parser
        # with each request.
        if self._parser is None:
            self.__class__._parser = _mathtext.Parser()

        box = self._parser.parse(s, font_output, fontsize, dpi)
        font_output.set_canvas_size(box.width, box.height, box.depth)
        return font_output.get_results(box)

    @_api.deprecated("3.4", alternative="mathtext.math_to_image")
    def to_mask(self, texstr, dpi=120, fontsize=14):
        r"""
        Convert a mathtext string to a grayscale array and depth.

        Parameters
        ----------
        texstr : str
            A valid mathtext string, e.g., r'IQ: $\sigma_i=15$'.
        dpi : float
            The dots-per-inch setting used to render the text.
        fontsize : int
            The font size in points

        Returns
        -------
        array : 2D uint8 alpha
            Mask array of rasterized tex.
        depth : int
            Offset of the baseline from the bottom of the image, in pixels.
        """
        assert self._output == "bitmap"
        prop = FontProperties(size=fontsize)
        ftimage, depth = self.parse(texstr, dpi=dpi, prop=prop)
        return np.asarray(ftimage), depth

    @_api.deprecated("3.4", alternative="mathtext.math_to_image")
    def to_rgba(self, texstr, color='black', dpi=120, fontsize=14):
        r"""
        Convert a mathtext string to an RGBA array and depth.

        Parameters
        ----------
        texstr : str
            A valid mathtext string, e.g., r'IQ: $\sigma_i=15$'.
        color : color
            The text color.
        dpi : float
            The dots-per-inch setting used to render the text.
        fontsize : int
            The font size in points.

        Returns
        -------
        array : (M, N, 4) array
            RGBA color values of rasterized tex, colorized with *color*.
        depth : int
            Offset of the baseline from the bottom of the image, in pixels.
        """
        x, depth = self.to_mask(texstr, dpi=dpi, fontsize=fontsize)

        r, g, b, a = mcolors.to_rgba(color)
        RGBA = np.zeros((x.shape[0], x.shape[1], 4), dtype=np.uint8)
        RGBA[:, :, 0] = 255 * r
        RGBA[:, :, 1] = 255 * g
        RGBA[:, :, 2] = 255 * b
        RGBA[:, :, 3] = x
        return RGBA, depth

    @_api.deprecated("3.4", alternative="mathtext.math_to_image")
    def to_png(self, filename, texstr, color='black', dpi=120, fontsize=14):
        r"""
        Render a tex expression to a PNG file.

        Parameters
        ----------
        filename
            A writable filename or fileobject.
        texstr : str
            A valid mathtext string, e.g., r'IQ: $\sigma_i=15$'.
        color : color
            The text color.
        dpi : float
            The dots-per-inch setting used to render the text.
        fontsize : int
            The font size in points.

        Returns
        -------
        int
            Offset of the baseline from the bottom of the image, in pixels.
        """
        rgba, depth = self.to_rgba(
            texstr, color=color, dpi=dpi, fontsize=fontsize)
        Image.fromarray(rgba).save(filename, format="png")
        return depth

    @_api.deprecated("3.4", alternative="mathtext.math_to_image")
    def get_depth(self, texstr, dpi=120, fontsize=14):
        r"""
        Get the depth of a mathtext string.

        Parameters
        ----------
        texstr : str
            A valid mathtext string, e.g., r'IQ: $\sigma_i=15$'.
        dpi : float
            The dots-per-inch setting used to render the text.

        Returns
        -------
        int
            Offset of the baseline from the bottom of the image, in pixels.
        """
        assert self._output == "bitmap"
        prop = FontProperties(size=fontsize)
        ftimage, depth = self.parse(texstr, dpi=dpi, prop=prop)
        return depth


def math_to_image(s, filename_or_obj, prop=None, dpi=None, format=None):
    """
    Given a math expression, renders it in a closely-clipped bounding
    box to an image file.

    Parameters
    ----------
    s : str
        A math expression.  The math portion must be enclosed in dollar signs.
    filename_or_obj : str or path-like or file-like
        Where to write the image data.
    prop : `.FontProperties`, optional
        The size and style of the text.
    dpi : float, optional
        The output dpi.  If not set, the dpi is determined as for
        `.Figure.savefig`.
    format : str, optional
        The output format, e.g., 'svg', 'pdf', 'ps' or 'png'.  If not set, the
        format is determined as for `.Figure.savefig`.
    """
    from matplotlib import figure
    # backend_agg supports all of the core output formats
    from matplotlib.backends import backend_agg

    if prop is None:
        prop = FontProperties()

    parser = MathTextParser('path')
    width, height, depth, _, _ = parser.parse(s, dpi=72, prop=prop)

    fig = figure.Figure(figsize=(width / 72.0, height / 72.0))
    fig.text(0, depth/height, s, fontproperties=prop)
    backend_agg.FigureCanvasAgg(fig)
    fig.savefig(filename_or_obj, dpi=dpi, format=format)

    return depth
