def do_3d_projection(self, renderer=None):
    # see _update_scalarmappable docstring for why this must be here
    _update_scalarmappable(self)
    xs, ys, zs = self._offsets3d
    vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs,
                                                    self.axes.M)

    fcs = (_zalpha(self._facecolor3d, vzs) if self._depthshade else
           self._facecolor3d)
    fcs = mcolors.to_rgba_array(fcs, self._alpha)
    super().set_facecolor(fcs)

    ecs = (_zalpha(self._edgecolor3d, vzs) if self._depthshade else
           self._edgecolor3d)
    ecs = mcolors.to_rgba_array(ecs, self._alpha)
    super().set_edgecolor(ecs)
    super().set_offsets(np.column_stack([vxs, vys]))

    if vzs.size > 0:
        return min(vzs)
    else:
        return np.nan
