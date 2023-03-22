from contextlib import ExitStack
from copy import copy
import io
import os
from pathlib import Path
import platform
import sys
import urllib.request

import numpy as np
from numpy.testing import assert_array_equal
from PIL import Image

from matplotlib import (
    _api, colors, image as mimage, patches, pyplot as plt, style, rcParams)
from matplotlib.image import (AxesImage, BboxImage, FigureImage,
                              NonUniformImage, PcolorImage)
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.transforms import Bbox, Affine2D, TransformedBbox
import matplotlib.ticker as mticker

import pytest


@image_comparison(['image_interps'], style='mpl20')
def test_image_interps():
    """Make the basic nearest, bilinear and bicubic interps."""
    # Remove this line when this test image is regenerated.
    plt.rcParams['text.kerning_factor'] = 6

    X = np.arange(100).reshape(5, 20)

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.imshow(X, interpolation='nearest')
    ax1.set_title('three interpolations')
    ax1.set_ylabel('nearest')

    ax2.imshow(X, interpolation='bilinear')
    ax2.set_ylabel('bilinear')

    ax3.imshow(X, interpolation='bicubic')
    ax3.set_ylabel('bicubic')


@image_comparison(['interp_alpha.png'], remove_text=True)
def test_alpha_interp():
    """Test the interpolation of the alpha channel on RGBA images"""
    fig, (axl, axr) = plt.subplots(1, 2)
    # full green image
    img = np.zeros((5, 5, 4))
    img[..., 1] = np.ones((5, 5))
    # transparent under main diagonal
    img[..., 3] = np.tril(np.ones((5, 5), dtype=np.uint8))
    axl.imshow(img, interpolation="none")
    axr.imshow(img, interpolation="bilinear")


@image_comparison(['interp_nearest_vs_none'],
                  extensions=['pdf', 'svg'], remove_text=True)
def test_interp_nearest_vs_none():
    """Test the effect of "nearest" and "none" interpolation"""
    # Setting dpi to something really small makes the difference very
    # visible. This works fine with pdf, since the dpi setting doesn't
    # affect anything but images, but the agg output becomes unusably
    # small.
    rcParams['savefig.dpi'] = 3
    X = np.array([[[218, 165, 32], [122, 103, 238]],
                  [[127, 255, 0], [255, 99, 71]]], dtype=np.uint8)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(X, interpolation='none')
    ax1.set_title('interpolation none')
    ax2.imshow(X, interpolation='nearest')
    ax2.set_title('interpolation nearest')


@pytest.mark.parametrize('suppressComposite', [False, True])
@image_comparison(['figimage'], extensions=['png', 'pdf'])
def test_figimage(suppressComposite):
    fig = plt.figure(figsize=(2, 2), dpi=100)
    fig.suppressComposite = suppressComposite
    x, y = np.ix_(np.arange(100) / 100.0, np.arange(100) / 100)
    z = np.sin(x**2 + y**2 - x*y)
    c = np.sin(20*x**2 + 50*y**2)
    img = z + c/5

    fig.figimage(img, xo=0, yo=0, origin='lower')
    fig.figimage(img[::-1, :], xo=0, yo=100, origin='lower')
    fig.figimage(img[:, ::-1], xo=100, yo=0, origin='lower')
    fig.figimage(img[::-1, ::-1], xo=100, yo=100, origin='lower')


def test_image_python_io():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    plt.imread(buffer)


@pytest.mark.parametrize(
    "img_size, fig_size, interpolation",
    [(5, 2, "hanning"),  # data larger than figure.
     (5, 5, "nearest"),  # exact resample.
     (5, 10, "nearest"),  # double sample.
     (3, 2.9, "hanning"),  # <3 upsample.
     (3, 9.1, "nearest"),  # >3 upsample.
     ])
@check_figures_equal(extensions=['png'])
def test_imshow_antialiased(fig_test, fig_ref,
                            img_size, fig_size, interpolation):
    np.random.seed(19680801)
    dpi = plt.rcParams["savefig.dpi"]
    A = np.random.rand(int(dpi * img_size), int(dpi * img_size))
    for fig in [fig_test, fig_ref]:
        fig.set_size_inches(fig_size, fig_size)
    axs = fig_test.subplots()
    axs.set_position([0, 0, 1, 1])
    axs.imshow(A, interpolation='antialiased')
    axs = fig_ref.subplots()
    axs.set_position([0, 0, 1, 1])
    axs.imshow(A, interpolation=interpolation)


@check_figures_equal(extensions=['png'])
def test_imshow_zoom(fig_test, fig_ref):
    # should be less than 3 upsample, so should be nearest...
    np.random.seed(19680801)
    dpi = plt.rcParams["savefig.dpi"]
    A = np.random.rand(int(dpi * 3), int(dpi * 3))
    for fig in [fig_test, fig_ref]:
        fig.set_size_inches(2.9, 2.9)
    axs = fig_test.subplots()
    axs.imshow(A, interpolation='antialiased')
    axs.set_xlim([10, 20])
    axs.set_ylim([10, 20])
    axs = fig_ref.subplots()
    axs.imshow(A, interpolation='nearest')
    axs.set_xlim([10, 20])
    axs.set_ylim([10, 20])


@check_figures_equal()
def test_imshow_pil(fig_test, fig_ref):
    style.use("default")
    png_path = Path(__file__).parent / "baseline_images/pngsuite/basn3p04.png"
    tiff_path = Path(__file__).parent / "baseline_images/test_image/uint16.tif"
    axs = fig_test.subplots(2)
    axs[0].imshow(Image.open(png_path))
    axs[1].imshow(Image.open(tiff_path))
    axs = fig_ref.subplots(2)
    axs[0].imshow(plt.imread(png_path))
    axs[1].imshow(plt.imread(tiff_path))


def test_imread_pil_uint16():
    img = plt.imread(os.path.join(os.path.dirname(__file__),
                     'baseline_images', 'test_image', 'uint16.tif'))
    assert img.dtype == np.uint16
    assert np.sum(img) == 134184960


def test_imread_fspath():
    img = plt.imread(
        Path(__file__).parent / 'baseline_images/test_image/uint16.tif')
    assert img.dtype == np.uint16
    assert np.sum(img) == 134184960


@pytest.mark.parametrize("fmt", ["png", "jpg", "jpeg", "tiff"])
def test_imsave(fmt):
    has_alpha = fmt not in ["jpg", "jpeg"]

    # The goal here is that the user can specify an output logical DPI
    # for the image, but this will not actually add any extra pixels
    # to the image, it will merely be used for metadata purposes.

    # So we do the traditional case (dpi == 1), and the new case (dpi
    # == 100) and read the resulting PNG files back in and make sure
    # the data is 100% identical.
    np.random.seed(1)
    # The height of 1856 pixels was selected because going through creating an
    # actual dpi=100 figure to save the image to a Pillow-provided format would
    # cause a rounding error resulting in a final image of shape 1855.
    data = np.random.rand(1856, 2)

    buff_dpi1 = io.BytesIO()
    plt.imsave(buff_dpi1, data, format=fmt, dpi=1)

    buff_dpi100 = io.BytesIO()
    plt.imsave(buff_dpi100, data, format=fmt, dpi=100)

    buff_dpi1.seek(0)
    arr_dpi1 = plt.imread(buff_dpi1, format=fmt)

    buff_dpi100.seek(0)
    arr_dpi100 = plt.imread(buff_dpi100, format=fmt)

    assert arr_dpi1.shape == (1856, 2, 3 + has_alpha)
    assert arr_dpi100.shape == (1856, 2, 3 + has_alpha)

    assert_array_equal(arr_dpi1, arr_dpi100)


@pytest.mark.parametrize("fmt", ["png", "pdf", "ps", "eps", "svg"])
def test_imsave_fspath(fmt):
    plt.imsave(Path(os.devnull), np.array([[0, 1]]), format=fmt)


def test_imsave_color_alpha():
    # Test that imsave accept arrays with ndim=3 where the third dimension is
    # color and alpha without raising any exceptions, and that the data is
    # acceptably preserved through a save/read roundtrip.
    np.random.seed(1)

    for origin in ['lower', 'upper']:
        data = np.random.rand(16, 16, 4)
        buff = io.BytesIO()
        plt.imsave(buff, data, origin=origin, format="png")

        buff.seek(0)
        arr_buf = plt.imread(buff)

        # Recreate the float -> uint8 conversion of the data
        # We can only expect to be the same with 8 bits of precision,
        # since that's what the PNG file used.
        data = (255*data).astype('uint8')
        if origin == 'lower':
            data = data[::-1]
        arr_buf = (255*arr_buf).astype('uint8')

        assert_array_equal(data, arr_buf)


def test_imsave_pil_kwargs_png():
    from PIL.PngImagePlugin import PngInfo
    buf = io.BytesIO()
    pnginfo = PngInfo()
    pnginfo.add_text("Software", "test")
    plt.imsave(buf, [[0, 1], [2, 3]],
               format="png", pil_kwargs={"pnginfo": pnginfo})
    im = Image.open(buf)
    assert im.info["Software"] == "test"


def test_imsave_pil_kwargs_tiff():
    from PIL.TiffTags import TAGS_V2 as TAGS
    buf = io.BytesIO()
    pil_kwargs = {"description": "test image"}
    plt.imsave(buf, [[0, 1], [2, 3]], format="tiff", pil_kwargs=pil_kwargs)
    im = Image.open(buf)
    tags = {TAGS[k].name: v for k, v in im.tag_v2.items()}
    assert tags["ImageDescription"] == "test image"


@image_comparison(['image_alpha'], remove_text=True)
def test_image_alpha():
    np.random.seed(0)
    Z = np.random.rand(6, 6)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(Z, alpha=1.0, interpolation='none')
    ax2.imshow(Z, alpha=0.5, interpolation='none')
    ax3.imshow(Z, alpha=0.5, interpolation='nearest')


def test_cursor_data():
    from matplotlib.backend_bases import MouseEvent

    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(100).reshape(10, 10), origin='upper')

    x, y = 4, 4
    xdisp, ydisp = ax.transData.transform([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) == 44

    # Now try for a point outside the image
    # Tests issue #4957
    x, y = 10.1, 4
    xdisp, ydisp = ax.transData.transform([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) is None

    # Hmm, something is wrong here... I get 0, not None...
    # But, this works further down in the tests with extents flipped
    #x, y = 0.1, -0.1
    #xdisp, ydisp = ax.transData.transform([x, y])
    #event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    #z = im.get_cursor_data(event)
    #assert z is None, "Did not get None, got %d" % z

    ax.clear()
    # Now try with the extents flipped.
    im = ax.imshow(np.arange(100).reshape(10, 10), origin='lower')

    x, y = 4, 4
    xdisp, ydisp = ax.transData.transform([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) == 44

    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(100).reshape(10, 10), extent=[0, 0.5, 0, 0.5])

    x, y = 0.25, 0.25
    xdisp, ydisp = ax.transData.transform([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) == 55

    # Now try for a point outside the image
    # Tests issue #4957
    x, y = 0.75, 0.25
    xdisp, ydisp = ax.transData.transform([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) is None

    x, y = 0.01, -0.01
    xdisp, ydisp = ax.transData.transform([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) is None

    # Now try with additional transform applied to the image artist
    trans = Affine2D().scale(2).rotate(0.5)
    im = ax.imshow(np.arange(100).reshape(10, 10),
                   transform=trans + ax.transData)
    x, y = 3, 10
    xdisp, ydisp = ax.transData.transform([x, y])
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) == 44


@pytest.mark.parametrize(
    "data, text_without_colorbar, text_with_colorbar", [
        ([[10001, 10000]], "[1e+04]", "[10001]"),
        ([[.123, .987]], "[0.123]", "[0.123]"),
])
def test_format_cursor_data(data, text_without_colorbar, text_with_colorbar):
    from matplotlib.backend_bases import MouseEvent

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    xdisp, ydisp = ax.transData.transform([0, 0])
    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    assert im.get_cursor_data(event) == data[0][0]
    assert im.format_cursor_data(im.get_cursor_data(event)) \
        == text_without_colorbar

    fig.colorbar(im)
    fig.canvas.draw()  # This is necessary to set up the colorbar formatter.

    assert im.get_cursor_data(event) == data[0][0]
    assert im.format_cursor_data(im.get_cursor_data(event)) \
        == text_with_colorbar


@image_comparison(['image_clip'], style='mpl20')
def test_image_clip():
    d = [[1, 2], [3, 4]]

    fig, ax = plt.subplots()
    im = ax.imshow(d)
    patch = patches.Circle((0, 0), radius=1, transform=ax.transData)
    im.set_clip_path(patch)


@image_comparison(['image_cliprect'], style='mpl20')
def test_image_cliprect():
    fig, ax = plt.subplots()
    d = [[1, 2], [3, 4]]

    im = ax.imshow(d, extent=(0, 5, 0, 5))

    rect = patches.Rectangle(
        xy=(1, 1), width=2, height=2, transform=im.axes.transData)
    im.set_clip_path(rect)


@image_comparison(['imshow'], remove_text=True, style='mpl20')
def test_imshow():
    fig, ax = plt.subplots()
    arr = np.arange(100).reshape((10, 10))
    ax.imshow(arr, interpolation="bilinear", extent=(1, 2, 1, 2))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)


@check_figures_equal(extensions=['png'])
def test_imshow_10_10_1(fig_test, fig_ref):
    # 10x10x1 should be the same as 10x10
    arr = np.arange(100).reshape((10, 10, 1))
    ax = fig_ref.subplots()
    ax.imshow(arr[:, :, 0], interpolation="bilinear", extent=(1, 2, 1, 2))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    ax = fig_test.subplots()
    ax.imshow(arr, interpolation="bilinear", extent=(1, 2, 1, 2))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)


def test_imshow_10_10_2():
    fig, ax = plt.subplots()
    arr = np.arange(200).reshape((10, 10, 2))
    with pytest.raises(TypeError):
        ax.imshow(arr)


def test_imshow_10_10_5():
    fig, ax = plt.subplots()
    arr = np.arange(500).reshape((10, 10, 5))
    with pytest.raises(TypeError):
        ax.imshow(arr)


@image_comparison(['no_interpolation_origin'], remove_text=True)
def test_no_interpolation_origin():
    fig, axs = plt.subplots(2)
    axs[0].imshow(np.arange(100).reshape((2, 50)), origin="lower",
                  interpolation='none')
    axs[1].imshow(np.arange(100).reshape((2, 50)), interpolation='none')


@image_comparison(['image_shift'], remove_text=True, extensions=['pdf', 'svg'])
def test_image_shift():
    imgData = [[1 / x + 1 / y for x in range(1, 100)] for y in range(1, 100)]
    tMin = 734717.945208
    tMax = 734717.946366

    fig, ax = plt.subplots()
    ax.imshow(imgData, norm=colors.LogNorm(), interpolation='none',
              extent=(tMin, tMax, 1, 100))
    ax.set_aspect('auto')


def test_image_edges():
    fig = plt.figure(figsize=[1, 1])
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)

    data = np.tile(np.arange(12), 15).reshape(20, 9)

    im = ax.imshow(data, origin='upper', extent=[-10, 10, -10, 10],
                   interpolation='none', cmap='gray')

    x = y = 2
    ax.set_xlim([-x, x])
    ax.set_ylim([-y, y])

    ax.set_xticks([])
    ax.set_yticks([])

    buf = io.BytesIO()
    fig.savefig(buf, facecolor=(0, 1, 0))

    buf.seek(0)

    im = plt.imread(buf)
    r, g, b, a = sum(im[:, 0])
    r, g, b, a = sum(im[:, -1])

    assert g != 100, 'Expected a non-green edge - but sadly, it was.'


@image_comparison(['image_composite_background'],
                  remove_text=True, style='mpl20')
def test_image_composite_background():
    fig, ax = plt.subplots()
    arr = np.arange(12).reshape(4, 3)
    ax.imshow(arr, extent=[0, 2, 15, 0])
    ax.imshow(arr, extent=[4, 6, 15, 0])
    ax.set_facecolor((1, 0, 0, 0.5))
    ax.set_xlim([0, 12])


@image_comparison(['image_composite_alpha'], remove_text=True)
def test_image_composite_alpha():
    """
    Tests that the alpha value is recognized and correctly applied in the
    process of compositing images together.
    """
    fig, ax = plt.subplots()
    arr = np.zeros((11, 21, 4))
    arr[:, :, 0] = 1
    arr[:, :, 3] = np.concatenate(
        (np.arange(0, 1.1, 0.1), np.arange(0, 1, 0.1)[::-1]))
    arr2 = np.zeros((21, 11, 4))
    arr2[:, :, 0] = 1
    arr2[:, :, 1] = 1
    arr2[:, :, 3] = np.concatenate(
        (np.arange(0, 1.1, 0.1), np.arange(0, 1, 0.1)[::-1]))[:, np.newaxis]
    ax.imshow(arr, extent=[1, 2, 5, 0], alpha=0.3)
    ax.imshow(arr, extent=[2, 3, 5, 0], alpha=0.6)
    ax.imshow(arr, extent=[3, 4, 5, 0])
    ax.imshow(arr2, extent=[0, 5, 1, 2])
    ax.imshow(arr2, extent=[0, 5, 2, 3], alpha=0.6)
    ax.imshow(arr2, extent=[0, 5, 3, 4], alpha=0.3)
    ax.set_facecolor((0, 0.5, 0, 1))
    ax.set_xlim([0, 5])
    ax.set_ylim([5, 0])


@image_comparison(['rasterize_10dpi'],
                  extensions=['pdf', 'svg'], remove_text=True, style='mpl20')
def test_rasterize_dpi():
    # This test should check rasterized rendering with high output resolution.
    # It plots a rasterized line and a normal image with imshow.  So it will
    # catch when images end up in the wrong place in case of non-standard dpi
    # setting.  Instead of high-res rasterization I use low-res.  Therefore
    # the fact that the resolution is non-standard is easily checked by
    # image_comparison.
    img = np.asarray([[1, 2], [3, 4]])

    fig, axs = plt.subplots(1, 3, figsize=(3, 1))

    axs[0].imshow(img)

    axs[1].plot([0, 1], [0, 1], linewidth=20., rasterized=True)
    axs[1].set(xlim=(0, 1), ylim=(-1, 2))

    axs[2].plot([0, 1], [0, 1], linewidth=20.)
    axs[2].set(xlim=(0, 1), ylim=(-1, 2))

    # Low-dpi PDF rasterization errors prevent proper image comparison tests.
    # Hide detailed structures like the axes spines.
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

    rcParams['savefig.dpi'] = 10


@image_comparison(['bbox_image_inverted'], remove_text=True, style='mpl20')
def test_bbox_image_inverted():
    # This is just used to produce an image to feed to BboxImage
    image = np.arange(100).reshape((10, 10))

    fig, ax = plt.subplots()
    bbox_im = BboxImage(
        TransformedBbox(Bbox([[100, 100], [0, 0]]), ax.transData),
        interpolation='nearest')
    bbox_im.set_data(image)
    bbox_im.set_clip_on(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.add_artist(bbox_im)

    image = np.identity(10)

    bbox_im = BboxImage(TransformedBbox(Bbox([[0.1, 0.2], [0.3, 0.25]]),
                                        ax.figure.transFigure),
                        interpolation='nearest')
    bbox_im.set_data(image)
    bbox_im.set_clip_on(False)
    ax.add_artist(bbox_im)


def test_get_window_extent_for_AxisImage():
    # Create a figure of known size (1000x1000 pixels), place an image
    # object at a given location and check that get_window_extent()
    # returns the correct bounding box values (in pixels).

    im = np.array([[0.25, 0.75, 1.0, 0.75], [0.1, 0.65, 0.5, 0.4],
                   [0.6, 0.3, 0.0, 0.2], [0.7, 0.9, 0.4, 0.6]])
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    im_obj = ax.imshow(
        im, extent=[0.4, 0.7, 0.2, 0.9], interpolation='nearest')

    fig.canvas.draw()
    renderer = fig.canvas.renderer
    im_bbox = im_obj.get_window_extent(renderer)

    assert_array_equal(im_bbox.get_points(), [[400, 200], [700, 900]])


@image_comparison(['zoom_and_clip_upper_origin.png'],
                  remove_text=True, style='mpl20')
def test_zoom_and_clip_upper_origin():
    image = np.arange(100)
    image = image.reshape((10, 10))

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_ylim(2.0, -0.5)
    ax.set_xlim(-0.5, 2.0)


def test_nonuniformimage_setcmap():
    ax = plt.gca()
    im = NonUniformImage(ax)
    im.set_cmap('Blues')


def test_nonuniformimage_setnorm():
    ax = plt.gca()
    im = NonUniformImage(ax)
    im.set_norm(plt.Normalize())


def test_jpeg_2d():
    # smoke test that mode-L pillow images work.
    imd = np.ones((10, 10), dtype='uint8')
    for i in range(10):
        imd[i, :] = np.linspace(0.0, 1.0, 10) * 255
    im = Image.new('L', (10, 10))
    im.putdata(imd.flatten())
    fig, ax = plt.subplots()
    ax.imshow(im)


def test_jpeg_alpha():
    plt.figure(figsize=(1, 1), dpi=300)
    # Create an image that is all black, with a gradient from 0-1 in
    # the alpha channel from left to right.
    im = np.zeros((300, 300, 4), dtype=float)
    im[..., 3] = np.linspace(0.0, 1.0, 300)

    plt.figimage(im)

    buff = io.BytesIO()
    plt.savefig(buff, facecolor="red", format='jpg', dpi=300)

    buff.seek(0)
    image = Image.open(buff)

    # If this fails, there will be only one color (all black). If this
    # is working, we should have all 256 shades of grey represented.
    num_colors = len(image.getcolors(256))
    assert 175 <= num_colors <= 185
    # The fully transparent part should be red.
    corner_pixel = image.getpixel((0, 0))
    assert corner_pixel == (254, 0, 0)


def test_axesimage_setdata():
    ax = plt.gca()
    im = AxesImage(ax)
    z = np.arange(12, dtype=float).reshape((4, 3))
    im.set_data(z)
    z[0, 0] = 9.9
    assert im._A[0, 0] == 0, 'value changed'


def test_figureimage_setdata():
    fig = plt.gcf()
    im = FigureImage(fig)
    z = np.arange(12, dtype=float).reshape((4, 3))
    im.set_data(z)
    z[0, 0] = 9.9
    assert im._A[0, 0] == 0, 'value changed'


@pytest.mark.parametrize(
    "image_cls,x,y,a", [
        (NonUniformImage,
         np.arange(3.), np.arange(4.), np.arange(12.).reshape((4, 3))),
        (PcolorImage,
         np.arange(3.), np.arange(4.), np.arange(6.).reshape((3, 2))),
    ])
def test_setdata_xya(image_cls, x, y, a):
    ax = plt.gca()
    im = image_cls(ax)
    im.set_data(x, y, a)
    x[0] = y[0] = a[0, 0] = 9.9
    assert im._A[0, 0] == im._Ax[0] == im._Ay[0] == 0, 'value changed'
    im.set_data(x, y, a.reshape((*a.shape, -1)))  # Just a smoketest.


def test_minimized_rasterized():
    # This ensures that the rasterized content in the colorbars is
    # only as thick as the colorbar, and doesn't extend to other parts
    # of the image.  See #5814.  While the original bug exists only
    # in Postscript, the best way to detect it is to generate SVG
    # and then parse the output to make sure the two colorbar images
    # are the same size.
    from xml.etree import ElementTree

    np.random.seed(0)
    data = np.random.rand(10, 10)

    fig, ax = plt.subplots(1, 2)
    p1 = ax[0].pcolormesh(data)
    p2 = ax[1].pcolormesh(data)

    plt.colorbar(p1, ax=ax[0])
    plt.colorbar(p2, ax=ax[1])

    buff = io.BytesIO()
    plt.savefig(buff, format='svg')

    buff = io.BytesIO(buff.getvalue())
    tree = ElementTree.parse(buff)
    width = None
    for image in tree.iter('image'):
        if width is None:
            width = image['width']
        else:
            if image['width'] != width:
                assert False


def test_load_from_url():
    path = Path(__file__).parent / "baseline_images/pngsuite/basn3p04.png"
    url = ('file:'
           + ('///' if sys.platform == 'win32' else '')
           + path.resolve().as_posix())
<<<<<<< ./test_image.py/left.py
    with _api.suppress_matplotlib_deprecation_warning():
        plt.imread(url)
    with urllib.request.urlopen(url) as file:
        plt.imread(file)
||||||| ./test_image.py/base.py
    plt.imread(url)
    plt.imread(urllib.request.urlopen(url))
=======
    plt.imread(url)
    with urllib.request.urlopen(url) as file:
        plt.imread(file)
>>>>>>> ./test_image.py/right.py


@image_comparison(['log_scale_image'], remove_text=True)
def test_log_scale_image():
    Z = np.zeros((10, 10))
    Z[::2] = 1

    fig, ax = plt.subplots()
    ax.imshow(Z, extent=[1, 100, 1, 100], cmap='viridis', vmax=1, vmin=-1,
              aspect='auto')
    ax.set(yscale='log')


# Increased tolerance is needed for PDF test to avoid failure. After the PDF
# backend was modified to use indexed color, there are ten pixels that differ
# due to how the subpixel calculation is done when converting the PDF files to
# PNG images.
@image_comparison(['rotate_image'], remove_text=True, tol=0.35)
def test_rotate_image():
    delta = 0.25
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    Z2 = (np.exp(-(((X - 1) / 1.5)**2 + ((Y - 1) / 0.5)**2) / 2) /
          (2 * np.pi * 0.5 * 1.5))
    Z = Z2 - Z1  # difference of Gaussians

    fig, ax1 = plt.subplots(1, 1)
    im1 = ax1.imshow(Z, interpolation='none', cmap='viridis',
                     origin='lower',
                     extent=[-2, 4, -3, 2], clip_on=True)

    trans_data2 = Affine2D().rotate_deg(30) + ax1.transData
    im1.set_transform(trans_data2)

    # display intended extent of the image
    x1, x2, y1, y2 = im1.get_extent()

    ax1.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "r--", lw=3,
             transform=trans_data2)

    ax1.set_xlim(2, 5)
    ax1.set_ylim(0, 4)


def test_image_preserve_size():
    buff = io.BytesIO()

    im = np.zeros((481, 321))
    plt.imsave(buff, im, format="png")

    buff.seek(0)
    img = plt.imread(buff)

    assert img.shape[:2] == im.shape


def test_image_preserve_size2():
    n = 7
    data = np.identity(n, float)

    fig = plt.figure(figsize=(n, n), frameon=False)

    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, interpolation='nearest', origin='lower', aspect='auto')
    buff = io.BytesIO()
    fig.savefig(buff, dpi=1)

    buff.seek(0)
    img = plt.imread(buff)

    assert img.shape == (7, 7, 4)

    assert_array_equal(np.asarray(img[:, :, 0], bool),
                       np.identity(n, bool)[::-1])


@image_comparison(['mask_image_over_under.png'], remove_text=True, tol=1.0)
def test_mask_image_over_under():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    Z2 = (np.exp(-(((X - 1) / 1.5)**2 + ((Y - 1) / 0.5)**2) / 2) /
          (2 * np.pi * 0.5 * 1.5))
    Z = 10*(Z2 - Z1)  # difference of Gaussians

    palette = plt.cm.gray.with_extremes(over='r', under='g', bad='b')
    Zm = np.ma.masked_where(Z > 1.2, Z)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im = ax1.imshow(Zm, interpolation='bilinear',
                    cmap=palette,
                    norm=colors.Normalize(vmin=-1.0, vmax=1.0, clip=False),
                    origin='lower', extent=[-3, 3, -3, 3])
    ax1.set_title('Green=low, Red=high, Blue=bad')
    fig.colorbar(im, extend='both', orientation='horizontal',
                 ax=ax1, aspect=10)

    im = ax2.imshow(Zm, interpolation='nearest',
                    cmap=palette,
                    norm=colors.BoundaryNorm([-1, -0.5, -0.2, 0, 0.2, 0.5, 1],
                                             ncolors=256, clip=False),
                    origin='lower', extent=[-3, 3, -3, 3])
    ax2.set_title('With BoundaryNorm')
    fig.colorbar(im, extend='both', spacing='proportional',
                 orientation='horizontal', ax=ax2, aspect=10)


@image_comparison(['mask_image'], remove_text=True)
def test_mask_image():
    # Test mask image two ways: Using nans and using a masked array.

    fig, (ax1, ax2) = plt.subplots(1, 2)

    A = np.ones((5, 5))
    A[1:2, 1:2] = np.nan

    ax1.imshow(A, interpolation='nearest')

    A = np.zeros((5, 5), dtype=bool)
    A[1:2, 1:2] = True
    A = np.ma.masked_array(np.ones((5, 5), dtype=np.uint16), A)

    ax2.imshow(A, interpolation='nearest')


def test_mask_image_all():
    # Test behavior with an image that is entirely masked does not warn
    data = np.full((2, 2), np.nan)
    fig, ax = plt.subplots()
    ax.imshow(data)
    fig.canvas.draw_idle()  # would emit a warning


@image_comparison(['imshow_endianess.png'], remove_text=True)
def test_imshow_endianess():
    x = np.arange(10)
    X, Y = np.meshgrid(x, x)
    Z = np.hypot(X - 5, Y - 5)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    kwargs = dict(origin="lower", interpolation='nearest', cmap='viridis')

    ax1.imshow(Z.astype('<f8'), **kwargs)
    ax2.imshow(Z.astype('>f8'), **kwargs)


@image_comparison(['imshow_masked_interpolation'],
                  tol=0 if platform.machine() == 'x86_64' else 0.01,
                  remove_text=True, style='mpl20')
def test_imshow_masked_interpolation():

    cmap = plt.get_cmap('viridis').with_extremes(over='r', under='b', bad='k')

    N = 20
    n = colors.Normalize(vmin=0, vmax=N*N-1)

    data = np.arange(N*N, dtype=float).reshape(N, N)

    data[5, 5] = -1
    # This will cause crazy ringing for the higher-order
    # interpolations
    data[15, 5] = 1e5

    # data[3, 3] = np.nan

    data[15, 15] = np.inf

    mask = np.zeros_like(data).astype('bool')
    mask[5, 15] = True

    data = np.ma.masked_array(data, mask)

    fig, ax_grid = plt.subplots(3, 6)
    interps = sorted(mimage._interpd_)
    interps.remove('antialiased')

    for interp, ax in zip(interps, ax_grid.ravel()):
        ax.set_title(interp)
        ax.imshow(data, norm=n, cmap=cmap, interpolation=interp)
        ax.axis('off')


def test_imshow_no_warn_invalid():
    plt.imshow([[1, 2], [3, np.nan]])  # Check that no warning is emitted.


@pytest.mark.parametrize(
    'dtype', [np.dtype(s) for s in 'u2 u4 i2 i4 i8 f4 f8'.split()])
def test_imshow_clips_rgb_to_valid_range(dtype):
    arr = np.arange(300, dtype=dtype).reshape((10, 10, 3))
    if dtype.kind != 'u':
        arr -= 10
    too_low = arr < 0
    too_high = arr > 255
    if dtype.kind == 'f':
        arr = arr / 255
    _, ax = plt.subplots()
    out = ax.imshow(arr).get_array()
    assert (out[too_low] == 0).all()
    if dtype.kind == 'f':
        assert (out[too_high] == 1).all()
        assert out.dtype.kind == 'f'
    else:
        assert (out[too_high] == 255).all()
        assert out.dtype == np.uint8


@image_comparison(['imshow_flatfield.png'], remove_text=True, style='mpl20')
def test_imshow_flatfield():
    fig, ax = plt.subplots()
    im = ax.imshow(np.ones((5, 5)), interpolation='nearest')
    im.set_clim(.5, 1.5)


@image_comparison(['imshow_bignumbers.png'], remove_text=True, style='mpl20')
def test_imshow_bignumbers():
    rcParams['image.interpolation'] = 'nearest'
    # putting a big number in an array of integers shouldn't
    # ruin the dynamic range of the resolved bits.
    fig, ax = plt.subplots()
    img = np.array([[1, 2, 1e12], [3, 1, 4]], dtype=np.uint64)
    pc = ax.imshow(img)
    pc.set_clim(0, 5)


@image_comparison(['imshow_bignumbers_real.png'],
                  remove_text=True, style='mpl20')
def test_imshow_bignumbers_real():
    rcParams['image.interpolation'] = 'nearest'
    # putting a big number in an array of integers shouldn't
    # ruin the dynamic range of the resolved bits.
    fig, ax = plt.subplots()
    img = np.array([[2., 1., 1.e22], [4., 1., 3.]])
    pc = ax.imshow(img)
    pc.set_clim(0, 5)


@pytest.mark.parametrize(
    "make_norm",
    [colors.Normalize,
     colors.LogNorm,
     lambda: colors.SymLogNorm(1),
     lambda: colors.PowerNorm(1)])
def test_empty_imshow(make_norm):
    fig, ax = plt.subplots()
    with pytest.warns(UserWarning,
                      match="Attempting to set identical left == right"):
        im = ax.imshow([[]], norm=make_norm())
    im.set_extent([-5, 5, -5, 5])
    fig.canvas.draw()

    with pytest.raises(RuntimeError):
        im.make_image(fig._cachedRenderer)


def test_imshow_float128():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((3, 3), dtype=np.longdouble))
    with (ExitStack() if np.can_cast(np.longdouble, np.float64, "equiv")
          else pytest.warns(UserWarning)):
        # Ensure that drawing doesn't cause crash.
        fig.canvas.draw()


def test_imshow_bool():
    fig, ax = plt.subplots()
    ax.imshow(np.array([[True, False], [False, True]], dtype=bool))


def test_full_invalid():
    fig, ax = plt.subplots()
    ax.imshow(np.full((10, 10), np.nan))
    with pytest.warns(UserWarning):
        fig.canvas.draw()


@pytest.mark.parametrize("fmt,counted",
                         [("ps", b" colorimage"), ("svg", b"<image")])
@pytest.mark.parametrize("composite_image,count", [(True, 1), (False, 2)])
def test_composite(fmt, counted, composite_image, count):
    # Test that figures can be saved with and without combining multiple images
    # (on a single set of axes) into a single composite image.
    X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
    Z = np.sin(Y ** 2)

    fig, ax = plt.subplots()
    ax.set_xlim(0, 3)
    ax.imshow(Z, extent=[0, 1, 0, 1])
    ax.imshow(Z[::-1], extent=[2, 3, 0, 1])
    plt.rcParams['image.composite_image'] = composite_image
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt)
    assert buf.getvalue().count(counted) == count


def test_relim():
    fig, ax = plt.subplots()
    ax.imshow([[0]], extent=(0, 1, 0, 1))
    ax.relim()
    ax.autoscale()
    assert ax.get_xlim() == ax.get_ylim() == (0, 1)


def test_unclipped():
    fig, ax = plt.subplots()
    ax.set_axis_off()
    im = ax.imshow([[0, 0], [0, 0]], aspect="auto", extent=(-10, 10, -10, 10),
                   cmap='gray', clip_on=False)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.canvas.draw()
    # The unclipped image should fill the *entire* figure and be black.
    # Ignore alpha for this comparison.
    assert (np.array(fig.canvas.buffer_rgba())[..., :3] == 0).all()


def test_respects_bbox():
    fig, axs = plt.subplots(2)
    for ax in axs:
        ax.set_axis_off()
    im = axs[1].imshow([[0, 1], [2, 3]], aspect="auto", extent=(0, 1, 0, 1))
    im.set_clip_path(None)
    # Make the image invisible in axs[1], but visible in axs[0] if we pan
    # axs[1] up.
    im.set_clip_box(axs[0].bbox)
    buf_before = io.BytesIO()
    fig.savefig(buf_before, format="rgba")
    assert {*buf_before.getvalue()} == {0xff}  # All white.
    axs[1].set(ylim=(-1, 0))
    buf_after = io.BytesIO()
    fig.savefig(buf_after, format="rgba")
    assert buf_before.getvalue() != buf_after.getvalue()  # Not all white.


def test_image_cursor_formatting():
    fig, ax = plt.subplots()
    # Create a dummy image to be able to call format_cursor_data
    im = ax.imshow(np.zeros((4, 4)))

    data = np.ma.masked_array([0], mask=[True])
    assert im.format_cursor_data(data) == '[]'

    data = np.ma.masked_array([0], mask=[False])
    assert im.format_cursor_data(data) == '[0]'

    data = np.nan
    assert im.format_cursor_data(data) == '[nan]'


@check_figures_equal()
def test_image_array_alpha(fig_test, fig_ref):
    """Per-pixel alpha channel test."""
    x = np.linspace(0, 1)
    xx, yy = np.meshgrid(x, x)

    zz = np.exp(- 3 * ((xx - 0.5) ** 2) + (yy - 0.7 ** 2))
    alpha = zz / zz.max()

    cmap = plt.get_cmap('viridis')
    ax = fig_test.add_subplot()
    ax.imshow(zz, alpha=alpha, cmap=cmap, interpolation='nearest')

    ax = fig_ref.add_subplot()
    rgba = cmap(colors.Normalize()(zz))
    rgba[..., -1] = alpha
    ax.imshow(rgba, interpolation='nearest')


def test_image_array_alpha_validation():
    with pytest.raises(TypeError, match="alpha must be a float, two-d"):
        plt.imshow(np.zeros((2, 2)), alpha=[1, 1])


@pytest.mark.style('mpl20')
def test_exact_vmin():
    cmap = copy(plt.cm.get_cmap("autumn_r"))
    cmap.set_under(color="lightgrey")

    # make the image exactly 190 pixels wide
    fig = plt.figure(figsize=(1.9, 0.1), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    data = np.array(
        [[-1, -1, -1, 0, 0, 0, 0, 43, 79, 95, 66, 1, -1, -1, -1, 0, 0, 0, 34]],
        dtype=float,
    )

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.axis("off")
    fig.canvas.draw()

    # get the RGBA slice from the image
    from_image = im.make_image(fig.canvas.renderer)[0][0]
    # expand the input to be 190 long and run through norm / cmap
    direct_computation = (
        im.cmap(im.norm((data * ([[1]] * 10)).T.ravel())) * 255
    ).astype(int)

    # check than the RBGA values are the same
    assert np.all(from_image == direct_computation)


@pytest.mark.network
@pytest.mark.flaky
def test_https_imread_smoketest():
    with _api.suppress_matplotlib_deprecation_warning():
        v = mimage.imread('https://matplotlib.org/1.5.0/_static/logo2.png')


# A basic ndarray subclass that implements a quantity
# It does not implement an entire unit system or all quantity math.
# There is just enough implemented to test handling of ndarray
# subclasses.
class QuantityND(np.ndarray):
    def __new__(cls, input_array, units):
        obj = np.asarray(input_array).view(cls)
        obj.units = units
        return obj

    def __array_finalize__(self, obj):
        self.units = getattr(obj, "units", None)

    def __getitem__(self, item):
        units = getattr(self, "units", None)
        ret = super(QuantityND, self).__getitem__(item)
        if isinstance(ret, QuantityND) or units is not None:
            ret = QuantityND(ret, units)
        return ret

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        func = getattr(ufunc, method)
        if "out" in kwargs:
            raise NotImplementedError
        if len(inputs) == 1:
            i0 = inputs[0]
            unit = getattr(i0, "units", "dimensionless")
            out_arr = func(np.asarray(i0), **kwargs)
        elif len(inputs) == 2:
            i0 = inputs[0]
            i1 = inputs[1]
            u0 = getattr(i0, "units", "dimensionless")
            u1 = getattr(i1, "units", "dimensionless")
            u0 = u1 if u0 is None else u0
            u1 = u0 if u1 is None else u1
            if ufunc in [np.add, np.subtract]:
                if u0 != u1:
                    raise ValueError
                unit = u0
            elif ufunc == np.multiply:
                unit = f"{u0}*{u1}"
            elif ufunc == np.divide:
                unit = f"{u0}/({u1})"
            else:
                raise NotImplementedError
            out_arr = func(i0.view(np.ndarray), i1.view(np.ndarray), **kwargs)
        else:
            raise NotImplementedError
        if unit is None:
            out_arr = np.array(out_arr)
        else:
            out_arr = QuantityND(out_arr, unit)
        return out_arr

    @property
    def v(self):
        return self.view(np.ndarray)


def test_quantitynd():
    q = QuantityND([1, 2], "m")
    q0, q1 = q[:]
    assert np.all(q.v == np.asarray([1, 2]))
    assert q.units == "m"
    assert np.all((q0 + q1).v == np.asarray([3]))
    assert (q0 * q1).units == "m*m"
    assert (q1 / q0).units == "m/(m)"
    with pytest.raises(ValueError):
        q0 + QuantityND(1, "s")


def test_imshow_quantitynd():
    # generate a dummy ndarray subclass
    arr = QuantityND(np.ones((2, 2)), "m")
    fig, ax = plt.subplots()
    ax.imshow(arr)
    # executing the draw should not raise an exception
    fig.canvas.draw()


@check_figures_equal(extensions=['png'])
def test_huge_range_log(fig_test, fig_ref):
    data = np.full((5, 5), -1, dtype=np.float64)
    data[0:2, :] = 1E20

    ax = fig_test.subplots()
    im = ax.imshow(data, norm=colors.LogNorm(vmin=100, vmax=data.max()),
                   interpolation='nearest', cmap='viridis')

    data = np.full((5, 5), -1, dtype=np.float64)
    data[0:2, :] = 1000

    cmap = copy(plt.get_cmap('viridis'))
    cmap.set_under('w')
    ax = fig_ref.subplots()
    im = ax.imshow(data, norm=colors.Normalize(vmin=100, vmax=data.max()),
                   interpolation='nearest', cmap=cmap)


@check_figures_equal()
def test_spy_box(fig_test, fig_ref):
    # setting up reference and test
    ax_test = fig_test.subplots(1, 3)
    ax_ref = fig_ref.subplots(1, 3)

    plot_data = (
        [[1, 1], [1, 1]],
        [[0, 0], [0, 0]],
        [[0, 1], [1, 0]],
    )
    plot_titles = ["ones", "zeros", "mixed"]

    for i, (z, title) in enumerate(zip(plot_data, plot_titles)):
        ax_test[i].set_title(title)
        ax_test[i].spy(z)
        ax_ref[i].set_title(title)
        ax_ref[i].imshow(z, interpolation='nearest',
                            aspect='equal', origin='upper', cmap='Greys',
                            vmin=0, vmax=1)
        ax_ref[i].set_xlim(-0.5, 1.5)
        ax_ref[i].set_ylim(1.5, -0.5)
        ax_ref[i].xaxis.tick_top()
        ax_ref[i].title.set_y(1.05)
        ax_ref[i].xaxis.set_ticks_position('both')
        ax_ref[i].xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True)
        )
        ax_ref[i].yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True)
        )
