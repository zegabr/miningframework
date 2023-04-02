2737a
>>>>>>> right.py_temp.py
.
2735a
<<<<<<< left.py_temp.py
$$$$$$$ / fig_h
=======
.
2730a
>>>>>>> right.py_temp.py
.
2729a
<<<<<<< left.py_temp.py
$$$$$$$ h1 $= y0 + oy[ny] / fig_h
=======
.
2722a
>>>>>>> right.py_temp.py
.
2721a
<<<<<<< left.py_temp.py
=======
.
2718,2720d
2715a
||||||| left.py_temp.py
$$$$$$$ self.get_anchor
$$$$$$$(
$$$$$$$
$$$$$$$)
$$$$$$$
=======
$$$$$$$ message$="Support for passing ny1$=None to mean ny+1 is "
                "deprecated since %
$$$$$$$(
$$$$$$$since
$$$$$$$)
$$$$$$$s; in a future version
$$$$$$$,
$$$$$$$ ny1$=None "
                "will mean 'up to the last cell'."
>>>>>>> right.py_temp.py
.
2714a
<<<<<<< base.py_temp.py
.
2713a
||||||| left.py_temp.py
$$$$$$$ fig_h
$$$$$$$,
$$$$$$$ fig_w
=======
$$$$$$$
                                      figH
$$$$$$$,
$$$$$$$ figW
$$$$$$$)
$$$$$$$
        if ny1 is None
$$$$$$$:
$$$$$$$
            _api.warn_deprecated
$$$$$$$(
$$$$$$$
                "3.5"
>>>>>>> right.py_temp.py
.
2711a
<<<<<<< base.py_temp.py
.
2710a
>>>>>>> right.py_temp.py
.
2709a
<<<<<<< left.py_temp.py
$$$$$$$ equal_ws
=======
.
2708a
>>>>>>> right.py_temp.py
.
2706a
<<<<<<< left.py_temp.py
$$$$$$$ summed_hs
=======
.
2699a
>>>>>>> right.py_temp.py
.
2698a
<<<<<<< left.py_temp.py
$$$$$$$
            y
=======
.
2697a
>>>>>>> right.py_temp.py
.
2696a
<<<<<<< left.py_temp.py
$$$$$$$ ww $= _locate
=======
.
2686a
>>>>>>> right.py_temp.py
.
2685a
<<<<<<< left.py_temp.py
        equal_ws $= self.get_horizontal_sizes
=======
.
2681a
>>>>>>> right.py_temp.py
.
2679a
<<<<<<< left.py_temp.py
        summed_hs $= self.get_vertical_sizes
=======
.
2666a
>>>>>>> right.py_temp.py
.
2661a
<<<<<<< left.py_temp.py
$$$$$$$ fig_h $= self._fig.bbox.size / self._fig.dpi
=======
.
2660a
>>>>>>> right.py_temp.py
.
2659a
<<<<<<< left.py_temp.py
        fig_w
=======
.
2636c
$$$$$$$ ny1 if ny1 is not None else ny + 1
.
2634c
$$$$$$$ 1
.
2591a
>>>>>>> right.py_temp.py
.
2590a
<<<<<<< left.py_temp.py
    A `SubplotDivider` for laying out axes vertically
$$$$$$$,
$$$$$$$ while ensuring that they
    have equal widths.
=======
.
2585a
>>>>>>> right.py_temp.py
.
2584a
<<<<<<< left.py_temp.py
$$$$$$$SubplotDivider
=======
.
2569a
>>>>>>> right.py_temp.py
.
2568a
<<<<<<< left.py_temp.py
=======
.
2563a
>>>>>>> right.py_temp.py
.
2562a
<<<<<<< left.py_temp.py
$$$$$$$ / fig_w
=======
.
2557a
>>>>>>> right.py_temp.py
.
2556a
<<<<<<< left.py_temp.py
$$$$$$$ w1 $= x0 + ox[nx] / fig_w
=======
.
2554a
>>>>>>> right.py_temp.py
.
2553a
<<<<<<< left.py_temp.py
=======
.
2550,2552d
2547a
||||||| left.py_temp.py
$$$$$$$ self.get_anchor
$$$$$$$(
$$$$$$$
$$$$$$$)
$$$$$$$
=======
$$$$$$$ message$="Support for passing nx1$=None to mean nx+1 is "
                "deprecated since %
$$$$$$$(
$$$$$$$since
$$$$$$$)
$$$$$$$s; in a future version
$$$$$$$,
$$$$$$$ nx1$=None "
                "will mean 'up to the last cell'."
>>>>>>> right.py_temp.py
.
2546a
<<<<<<< base.py_temp.py
.
2545a
||||||| left.py_temp.py
$$$$$$$ fig_w
$$$$$$$,
$$$$$$$ fig_h
=======
$$$$$$$
                                      figW
$$$$$$$,
$$$$$$$ figH
$$$$$$$)
$$$$$$$
        if nx1 is None
$$$$$$$:
$$$$$$$
            _api.warn_deprecated
$$$$$$$(
$$$$$$$
                "3.5"
>>>>>>> right.py_temp.py
.
2543a
<<<<<<< base.py_temp.py
.
2542a
>>>>>>> right.py_temp.py
.
2541a
<<<<<<< left.py_temp.py
$$$$$$$ equal_hs
=======
.
2540a
>>>>>>> right.py_temp.py
.
2538a
<<<<<<< left.py_temp.py
$$$$$$$ summed_ws
=======
.
2531a
>>>>>>> right.py_temp.py
.
2530a
<<<<<<< left.py_temp.py
$$$$$$$
            x
=======
.
2529a
>>>>>>> right.py_temp.py
.
2528a
<<<<<<< left.py_temp.py
$$$$$$$ hh $= _locate
=======
.
2518a
>>>>>>> right.py_temp.py
.
2517a
<<<<<<< left.py_temp.py
        equal_hs $= self.get_vertical_sizes
=======
.
2513a
>>>>>>> right.py_temp.py
.
2511a
<<<<<<< left.py_temp.py
        summed_ws $= self.get_horizontal_sizes
=======
.
2498a
>>>>>>> right.py_temp.py
.
2493a
<<<<<<< left.py_temp.py
$$$$$$$ fig_h $= self._fig.bbox.size / self._fig.dpi
=======
.
2492a
>>>>>>> right.py_temp.py
.
2491a
<<<<<<< left.py_temp.py
        fig_w
=======
.
2471a
>>>>>>> right.py_temp.py
.
2471a
<<<<<<< left.py_temp.py

class HBoxDivider
$$$$$$$(
$$$$$$$SubplotDivider
$$$$$$$)
$$$$$$$
$$$$$$$:
$$$$$$$
    """
    A `SubplotDivider` for laying out axes horizontally
$$$$$$$,
$$$$$$$ while ensuring that
    they have equal heights.

    Examples
    --------
    .. plot
$$$$$$$:
$$$$$$$
$$$$$$$:
$$$$$$$ gallery/axes_grid1/demo_axes_hbox_divider.py
    """

    def new_locator
$$$$$$$(
$$$$$$$self
$$$$$$$,
$$$$$$$ nx
$$$$$$$,
$$$$$$$ nx1$=None
$$$$$$$)
$$$$$$$
$$$$$$$:
$$$$$$$
        """
        Create a new `AxesLocator` for the specified cell.

        Parameters
        ----------
        nx
$$$$$$$,
$$$$$$$ nx1 
$$$$$$$:
$$$$$$$ int
            Integers specifying the column-position of the
            cell. When *nx1* is None
$$$$$$$,
$$$$$$$ a single *nx*-th column is
            specified. Otherwise location of columns spanning between *nx*
            to *nx1* 
$$$$$$$(
$$$$$$$but excluding *nx1*-th column
$$$$$$$)
$$$$$$$ is specified.
        """
        return AxesLocator
$$$$$$$(
$$$$$$$self
$$$$$$$,
$$$$$$$ nx
$$$$$$$,
$$$$$$$ 0
$$$$$$$,
$$$$$$$ nx1
$$$$$$$,
$$$$$$$ None
$$$$$$$)
$$$$$$$

=======
.
2464a
>>>>>>> right.py_temp.py
.
2463a
<<<<<<< left.py_temp.py
    return x0
=======
.
2458a
>>>>>>> right.py_temp.py
.
2457a
<<<<<<< left.py_temp.py
    x0
=======
.
2453a
>>>>>>> right.py_temp.py
.
2450a
<<<<<<< left.py_temp.py
$$$$$$$anchor
=======
.
2449a
>>>>>>> right.py_temp.py
.
2446a
<<<<<<< left.py_temp.py
    pb1_anchored $= pb1.anchored
=======
.
2436a
>>>>>>> right.py_temp.py
.
2435a
<<<<<<< left.py_temp.py
    pb1 $= mtransforms.Bbox.from_bounds
=======
.
2425a
>>>>>>> right.py_temp.py
.
2423a
<<<<<<< left.py_temp.py
$$$$$$$ / fig_h
    pb $= mtransforms.Bbox.from_bounds
=======
.
2422a
>>>>>>> right.py_temp.py
.
2421a
<<<<<<< left.py_temp.py
$$$$$$$karray[0]*h0_r + h0_a
=======
.
2420a
>>>>>>> right.py_temp.py
.
2417a
<<<<<<< left.py_temp.py
$$$$$$$ / fig_w
    h0_r
$$$$$$$,
$$$$$$$ h0_a $= equal_heights[0]
    hh $= 
=======
.
2414a
>>>>>>> right.py_temp.py
.
2413a
<<<<<<< left.py_temp.py
    ww $= 
=======
.
2408a
>>>>>>> right.py_temp.py
.
2407a
<<<<<<< left.py_temp.py
$$$$$$$summed_widths
=======
.
2406a
>>>>>>> right.py_temp.py
.
2404a
<<<<<<< left.py_temp.py
    ox $= _calc_offsets
=======
.
2402a
>>>>>>> right.py_temp.py
.
2400a
<<<<<<< left.py_temp.py
$$$$$$$ max_height$=fig_h * h
=======
.
2399a
>>>>>>> right.py_temp.py
.
2398a
<<<<<<< left.py_temp.py
        total_width$=fig_w * w
=======
.
2396a
>>>>>>> right.py_temp.py
.
2395a
<<<<<<< left.py_temp.py
$$$$$$$ equal_heights
=======
.
2394a
>>>>>>> right.py_temp.py
.
2393a
<<<<<<< left.py_temp.py
$$$$$$$
        summed_widths
=======
.
2392a
>>>>>>> right.py_temp.py
.
2386a
<<<<<<< left.py_temp.py
    karray $= _determine_karray
=======
.
2382a
>>>>>>> right.py_temp.py
.
2378a
<<<<<<< left.py_temp.py
$$$$$$$ fig_w
$$$$$$$,
$$$$$$$ fig_h
$$$$$$$,
$$$$$$$ anchor
=======
.
2377a
>>>>>>> right.py_temp.py
.
2376a
<<<<<<< left.py_temp.py
$$$$$$$ equal_heights
=======
.
2375a
>>>>>>> right.py_temp.py
.
2373a
<<<<<<< left.py_temp.py
$$$$$$$ summed_widths
=======
.
2366a
>>>>>>> right.py_temp.py
.
2363a
<<<<<<< left.py_temp.py
$$$$$$$x
=======
.
2362a
>>>>>>> right.py_temp.py
.
2359a
<<<<<<< left.py_temp.py
$$$$$$$.
def _locate
=======
.
2358a
||||||| left.py_temp.py
$$$$$$$see above re
$$$$$$$:
$$$$$$$ variable naming
=======
$$$$$$$self
$$$$$$$,
$$$$$$$ nx
$$$$$$$,
$$$$$$$ 0
$$$$$$$,
$$$$$$$ nx1 if nx1 is not None else nx + 1
$$$$$$$,
$$$$$$$ 1
>>>>>>> right.py_temp.py
.
2349a
<<<<<<< base.py_temp.py
.
2348a
>>>>>>> right.py_temp.py
.
2329a
<<<<<<< left.py_temp.py
# Helper for HBoxDivider/VBoxDivider 
=======
.
2328a
>>>>>>> right.py_temp.py
.
2315a
<<<<<<< left.py_temp.py
=======
.
2314a
>>>>>>> right.py_temp.py
.
2313a
<<<<<<< left.py_temp.py
    return offsets
=======
.
2309a
>>>>>>> right.py_temp.py
.
2308a
<<<<<<< left.py_temp.py
        offsets.append
=======
.
2302a
>>>>>>> right.py_temp.py
.
2301a
<<<<<<< left.py_temp.py
$$$$$$$summed_sizes
=======
.
2292a
>>>>>>> right.py_temp.py
.
2290a
<<<<<<< left.py_temp.py
    offsets $= [0.]
    for 
=======
.
2284a
>>>>>>> right.py_temp.py
.
2283a
<<<<<<< left.py_temp.py
$$$$$$$summed_sizes
=======
.
2282a
>>>>>>> right.py_temp.py
.
2280a
<<<<<<< left.py_temp.py

# Helper for HBoxDivider/VBoxDivider 
$$$$$$$(
$$$$$$$see above re
$$$$$$$:
$$$$$$$ variable naming
$$$$$$$)
$$$$$$$.
def _calc_offsets
=======
.
2279a
>>>>>>> right.py_temp.py
.
2278a
<<<<<<< left.py_temp.py
    return karray
=======
.
2276a
>>>>>>> right.py_temp.py
.
2275a
<<<<<<< left.py_temp.py
$$$$$$$max_height - eq_as
=======
.
2274a
>>>>>>> right.py_temp.py
.
2273a
<<<<<<< left.py_temp.py
        karray $= 
=======
.
2269a
>>>>>>> right.py_temp.py
.
2267a
<<<<<<< left.py_temp.py
    height $= karray_and_height[-1]
    if height > max_height
=======
.
2265a
>>>>>>> right.py_temp.py
.
2264a
<<<<<<< left.py_temp.py
    karray $= karray_and_height[
=======
.
2258a
>>>>>>> right.py_temp.py
.
2257a
<<<<<<< left.py_temp.py
    karray_and_height $= np.linalg.solve
=======
.
2253a
>>>>>>> right.py_temp.py
.
2252a
<<<<<<< left.py_temp.py
    # 
=======
.
2250a
>>>>>>> right.py_temp.py
.
2249a
<<<<<<< left.py_temp.py
$$$$$$$ $= total_summed_width
=======
.
2248a
>>>>>>> right.py_temp.py
.
2247a
<<<<<<< left.py_temp.py
$$$$$$$sm_r_i * k_i + sm_a_i
=======
.
2246a
>>>>>>> right.py_temp.py
.
2245a
<<<<<<< left.py_temp.py
    #   sum
=======
.
2243a
>>>>>>> right.py_temp.py
.
2242a
<<<<<<< left.py_temp.py
    #   eq_r_i * k_i + eq_a_i $= H for all i
=======
.
2234a
>>>>>>> right.py_temp.py
.
2233a
<<<<<<< left.py_temp.py
    # A @ K $= B
=======
.
2231a
>>>>>>> right.py_temp.py
.
2230a
<<<<<<< left.py_temp.py
$$$$$$$sm_as
=======
.
2229a
>>>>>>> right.py_temp.py
.
2228a
<<<<<<< left.py_temp.py
    B[-1] $= total_width - sum
=======
.
2226a
>>>>>>> right.py_temp.py
.
2224a
<<<<<<< left.py_temp.py
$$$$$$$-1] $= sm_rs
    B[
=======
.
2221a
>>>>>>> right.py_temp.py
.
2220a
<<<<<<< left.py_temp.py
    A[-1
=======
.
2216a
>>>>>>> right.py_temp.py
.
2215a
<<<<<<< left.py_temp.py
    A[
=======
.
2203a
>>>>>>> right.py_temp.py
.
2202a
<<<<<<< left.py_temp.py
    np.fill_diagonal
=======
.
2198a
>>>>>>> right.py_temp.py
.
2197a
<<<<<<< left.py_temp.py
    B $= np.zeros
=======
.
2187a
>>>>>>> right.py_temp.py
.
2186a
<<<<<<< left.py_temp.py
    A $= np.zeros
=======
.
2184a
>>>>>>> right.py_temp.py
.
2183a
<<<<<<< left.py_temp.py
$$$$$$$summed_widths
=======
.
2182a
>>>>>>> right.py_temp.py
.
2181a
<<<<<<< left.py_temp.py
$$$$$$$ sm_as $= np.asarray
=======
.
2180a
>>>>>>> right.py_temp.py
.
2179a
<<<<<<< left.py_temp.py
    sm_rs
=======
.
2177a
>>>>>>> right.py_temp.py
.
2176a
<<<<<<< left.py_temp.py
$$$$$$$equal_heights
=======
.
2173a
>>>>>>> right.py_temp.py
.
2172a
<<<<<<< left.py_temp.py
    eq_rs
=======
.
2170a
>>>>>>> right.py_temp.py
.
2169a
<<<<<<< left.py_temp.py
$$$$$$$equal_heights
=======
.
2168a
>>>>>>> right.py_temp.py
.
2167a
<<<<<<< left.py_temp.py
    n $= len
=======
.
2163a
>>>>>>> right.py_temp.py
.
2161a
<<<<<<< left.py_temp.py
$$$$$$$ max_height
=======
.
2160a
>>>>>>> right.py_temp.py
.
2158a
<<<<<<< left.py_temp.py
$$$$$$$ total_width
=======
.
2157a
>>>>>>> right.py_temp.py
.
2156a
<<<<<<< left.py_temp.py
$$$$$$$ equal_heights
=======
.
2155a
>>>>>>> right.py_temp.py
.
2154a
<<<<<<< left.py_temp.py
$$$$$$$summed_widths
=======
.
2153a
>>>>>>> right.py_temp.py
.
2147a
<<<<<<< left.py_temp.py
$$$$$$$.
def _determine_karray
=======
.
2146a
>>>>>>> right.py_temp.py
.
2145a
<<<<<<< left.py_temp.py
$$$$$$$and likewise for the helpers below
=======
.
2144a
>>>>>>> right.py_temp.py
.
2143a
<<<<<<< left.py_temp.py
# Helper for HBoxDivider/VBoxDivider.
# The variable names are written for a horizontal layout
$$$$$$$,
$$$$$$$ but the calculations
# work identically for vertical layouts 
=======
.
1816a
>>>>>>> right.py_temp.py
.
1804a
<<<<<<< left.py_temp.py
            pad $= mpl.rcParams["figure.subplot.hspace"] * self._yref
=======
.
1775a
>>>>>>> right.py_temp.py
.
1774a
<<<<<<< left.py_temp.py
            Padding between the axes.  float or str arguments are interpreted
            as ``axes_size.from_any
$$$$$$$(
$$$$$$$size
$$$$$$$,
$$$$$$$ AxesY
$$$$$$$(
$$$$$$$<main_axes>
$$$$$$$)
$$$$$$$
$$$$$$$)
$$$$$$$``.  Defaults to
            
$$$$$$$:
$$$$$$$rc
$$$$$$$:
$$$$$$$`figure.subplot.hspace` times the main axes height.
=======
.
1767a
>>>>>>> right.py_temp.py
.
1765a
<<<<<<< left.py_temp.py
$$$$$$$ AxesY
$$$$$$$(
$$$$$$$<main_axes>
$$$$$$$)
$$$$$$$
$$$$$$$)
$$$$$$$``.
=======
.
1764a
>>>>>>> right.py_temp.py
.
1760a
<<<<<<< left.py_temp.py
            The axes height.  float or str arguments are interpreted as
            ``axes_size.from_any
$$$$$$$(
$$$$$$$size
=======
.
1625a
>>>>>>> right.py_temp.py
.
1613a
<<<<<<< left.py_temp.py
            pad $= mpl.rcParams["figure.subplot.wspace"] * self._xref
=======
.
1584a
>>>>>>> right.py_temp.py
.
1583a
<<<<<<< left.py_temp.py
            Padding between the axes.  float or str arguments are interpreted
            as ``axes_size.from_any
$$$$$$$(
$$$$$$$size
$$$$$$$,
$$$$$$$ AxesX
$$$$$$$(
$$$$$$$<main_axes>
$$$$$$$)
$$$$$$$
$$$$$$$)
$$$$$$$``.  Defaults to
            
$$$$$$$:
$$$$$$$rc
$$$$$$$:
$$$$$$$`figure.subplot.wspace` times the main axes width.
=======
.
1576a
>>>>>>> right.py_temp.py
.
1574a
<<<<<<< left.py_temp.py
$$$$$$$ AxesX
$$$$$$$(
$$$$$$$<main_axes>
$$$$$$$)
$$$$$$$
$$$$$$$)
$$$$$$$``.
=======
.
1573a
>>>>>>> right.py_temp.py
.
1569a
<<<<<<< left.py_temp.py
            The axes width.  float or str arguments are interpreted as
            ``axes_size.from_any
$$$$$$$(
$$$$$$$size
=======
.
1001a
            _api.warn_deprecated
$$$$$$$(
$$$$$$$
                "3.5"
$$$$$$$,
$$$$$$$ message$="Support for passing ny1$=None to mean ny+1 is "
                "deprecated since %
$$$$$$$(
$$$$$$$since
$$$$$$$)
$$$$$$$s; in a future version
$$$$$$$,
$$$$$$$ ny1$=None "
                "will mean 'up to the last cell'."
$$$$$$$)
$$$$$$$
.
997a
            _api.warn_deprecated
$$$$$$$(
$$$$$$$
                "3.5"
$$$$$$$,
$$$$$$$ message$="Support for passing nx1$=None to mean nx+1 is "
                "deprecated since %
$$$$$$$(
$$$$$$$since
$$$$$$$)
$$$$$$$s; in a future version
$$$$$$$,
$$$$$$$ nx1$=None "
                "will mean 'up to the last cell'."
$$$$$$$)
$$$$$$$
.
936a

.
930,935c
    A callable object which returns the position and size of a given
    AxesDivider cell.
.
796c
$$$$$$$
            ny1 if ny1 is not None else ny + 1
.
794c
$$$$$$$
            nx1 if nx1 is not None else nx + 1
.
788c
$$$$$$$
            self
.
728a
>>>>>>> right.py_temp.py
.
727a
<<<<<<< left.py_temp.py
$$$$$$$ / fig_h
=======
.
722a
>>>>>>> right.py_temp.py
.
721a
<<<<<<< left.py_temp.py
$$$$$$$ h1 $= y0 + oy[ny] / fig_h
=======
.
719a
>>>>>>> right.py_temp.py
.
718a
<<<<<<< left.py_temp.py
$$$$$$$ / fig_w
=======
.
713a
>>>>>>> right.py_temp.py
.
712a
<<<<<<< left.py_temp.py
$$$$$$$ w1 $= x0 + ox[nx] / fig_w
=======
.
708a
            _api.warn_deprecated
$$$$$$$(
$$$$$$$
                "3.5"
$$$$$$$,
$$$$$$$ message$="Support for passing ny1$=None to mean ny+1 is "
                "deprecated since %
$$$$$$$(
$$$$$$$since
$$$$$$$)
$$$$$$$s; in a future version
$$$$$$$,
$$$$$$$ ny1$=None "
                "will mean 'up to the last cell'."
$$$$$$$)
$$$$$$$
.
704a
            _api.warn_deprecated
$$$$$$$(
$$$$$$$
                "3.5"
$$$$$$$,
$$$$$$$ message$="Support for passing nx1$=None to mean nx+1 is "
                "deprecated since %
$$$$$$$(
$$$$$$$since
$$$$$$$)
$$$$$$$s; in a future version
$$$$$$$,
$$$$$$$ nx1$=None "
                "will mean 'up to the last cell'."
$$$$$$$)
$$$$$$$
.
639a
>>>>>>> right.py_temp.py
.
638a
<<<<<<< left.py_temp.py
$$$$$$$ / fig_h
=======
.
634a
>>>>>>> right.py_temp.py
.
633a
<<<<<<< left.py_temp.py
$$$$$$$ / fig_w
=======
.
597a
>>>>>>> right.py_temp.py
.
596a
<<<<<<< left.py_temp.py
$$$$$$$ fig_h * h
=======
.
590a
>>>>>>> right.py_temp.py
.
589a
<<<<<<< left.py_temp.py
$$$$$$$ fig_w * w
=======
.
561a
>>>>>>> right.py_temp.py
.
556a
<<<<<<< left.py_temp.py
$$$$$$$ fig_h $= self._fig.bbox.size / self._fig.dpi
=======
.
555a
>>>>>>> right.py_temp.py
.
554a
<<<<<<< left.py_temp.py
        fig_w
=======
.
334a
>>>>>>> right.py_temp.py
.
334a
<<<<<<< left.py_temp.py
        See Also
        --------
        .Axes.set_anchor
=======
.
333a
>>>>>>> right.py_temp.py
.
317a
<<<<<<< left.py_temp.py
$$$$$$$ 1 is right or top
$$$$$$$)
$$$$$$$
$$$$$$$,
$$$$$$$ 'C' 
$$$$$$$(
$$$$$$$center
$$$$$$$)
$$$$$$$
$$$$$$$,
$$$$$$$ or a cardinal direction
            
$$$$$$$(
$$$$$$$'SW'
$$$$$$$,
$$$$$$$ southwest
$$$$$$$,
$$$$$$$ is bottom left
$$$$$$$,
$$$$$$$ etc.
$$$$$$$)
$$$$$$$.
=======
.
316a
>>>>>>> right.py_temp.py
.
315a
<<<<<<< left.py_temp.py
$$$$$$$ *y*
$$$$$$$)
$$$$$$$ pair of relative coordinates 
$$$$$$$(
$$$$$$$0 is left or
            bottom
=======
.
314a
>>>>>>> right.py_temp.py
.
313a
<<<<<<< left.py_temp.py
$$$$$$$ ...}
            Either an 
$$$$$$$(
$$$$$$$*x*
=======
.
302a
>>>>>>> right.py_temp.py
.
301a
<<<<<<< left.py_temp.py
$$$$$$$ 
$$$$$$$(
$$$$$$$float
$$$$$$$,
$$$$$$$ float
$$$$$$$)
$$$$$$$ or {'C'
=======
.
6a
>>>>>>> right.py_temp.py
.
6a
<<<<<<< left.py_temp.py
import matplotlib as mpl
=======
.
