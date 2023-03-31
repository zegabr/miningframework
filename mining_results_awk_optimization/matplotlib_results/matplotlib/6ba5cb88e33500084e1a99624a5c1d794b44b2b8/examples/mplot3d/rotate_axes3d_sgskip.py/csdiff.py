"""
==================
Rotating a 3D plot
==================

A very simple animation of a rotating 3D plot about all 3 axes.

See wire3d_animation_demo for another simple example of animating a 3D plot.

(This example is skipped when building the documentation gallery because it
intentionally takes a long time to run)
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Grab some example data and plot a basic wireframe.
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

# Set the axis labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Rotate the axes and update
for angle in range(0, 360*4 + 1):
    # Normalize the angle to the range [-180, 180] for display
    angle_norm = (angle + 180) % 360 - 180

    # Cycle through a full rotation of elevation, then azimuth, roll, and all
    elev = azim = roll = 0
    if angle <= 360:
        elev = angle_norm
    elif angle <= 360*2:
        azim = angle_norm
    elif angle <= 360*3:
        roll = angle_norm
    else:
        elev = azim = roll = angle_norm

    # Update the axis view and title
    ax.view_init(elev, azim, roll)
    plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim,
<<<<<<< /home/ze/miningframework/mining_results_version3_3/matplotlib_results/matplotlib/6ba5cb88e33500084e1a99624a5c1d794b44b2b8/examples/mplot3d/rotate_axes3d_sgskip.py/left.py
 roll
=======
 angle, 0
>>>>>>> /home/ze/miningframework/mining_results_version3_3/matplotlib_results/matplotlib/6ba5cb88e33500084e1a99624a5c1d794b44b2b8/examples/mplot3d/rotate_axes3d_sgskip.py/right.py
))
# CReduzido

    plt.draw()
    plt.pause(.001)