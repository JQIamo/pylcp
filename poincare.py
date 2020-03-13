# This file is a modified version of the Bloch sphere class from QuTiP: Quantum
# Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import os
import numpy as np
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib import cm, colors

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)

            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)

            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)
except:
    pass


def plot_vectors_quarter(ax, vectors, projections=True,
                         vector_width=1,
                         vector_mutation=5,
                         sphere=True,
                         sphere_color='#FFDDDD',
                         sphere_alpha=0.1,
                         grid_lines=True,
                         grid_longitudes=np.linspace(-np.pi/2, np.pi/2, 6),
                         grid_latitudes=[np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2],
                         grid_linewidth=0.25,
                         grid_color='gray',
                         xlabel = ['$s$', '$p$'],
                         xlpos = [1.2, -1.2],
                         ylabel = '',
                         ylpos = 1.2,
                         zlabel = '$\\sigma^-$',
                         zlpos = 1.2):

    """
    Plots vectors in the quarter of the sphere:
    """
    #self.plot_axes()
    ax.set_axis_off()
    ax.set_xlim3d(-0.7, 0.7)
    ax.set_ylim3d(-0.7, 0.7)
    ax.set_zlim3d(-0.7, 0.7)

    # back half of sphere
    phi = np.linspace(-np.pi/2, np.pi/2, 25)
    th = np.linspace(0, np.pi/2, 25)

    # Plot the back quarter in the x-z plane:
    x = np.outer(np.zeros(np.size(phi)), np.zeros(np.size(th)))
    y = np.outer(np.sin(phi), np.sin(th))
    z = np.outer(np.ones(np.size(phi)), np.cos(th))

    ax.plot_surface(x, y, z, rstride=2, cstride=2,
                    color=sphere_color, linewidth=0,
                    alpha=sphere_alpha)

    # Plot the back quarter in the x-y plane:
    x = np.cos(phi)
    y = np.sin(phi)
    z = np.zeros(np.size(th))

    ax.plot_trisurf(x, y, z,
                    color=sphere_color, edgecolor='none',
                    alpha=sphere_alpha)

    if grid_lines:
        for grid_latitude in grid_latitudes:
            if grid_latitude!=np.pi/2:
                ax.plot(np.array([0, 0]),
                        np.array([-1, 1])*np.sin(grid_latitude),
                        np.array([1, 1])*np.cos(grid_latitude),
                        linewidth=grid_linewidth,
                        color=grid_color)

        for grid_longitude in grid_longitudes:
            ax.plot(np.array([0, 1])*np.cos(grid_longitude),
                    np.array([0, 1])*np.sin(grid_longitude),
                    np.array([0, 0]),
                    linewidth=grid_linewidth,
                    color=grid_color)

    # Plot the front half of the sphere:
    x = np.outer(np.cos(phi), np.sin(th))
    y = np.outer(np.sin(phi), np.sin(th))
    z = np.outer(np.ones(np.size(phi)), np.cos(th))
    ax.plot_surface(x, y, z, rstride=2, cstride=2,
                           color=sphere_color, linewidth=0,
                           alpha=sphere_alpha)

    if grid_lines:
        for grid_latitude in grid_latitudes:
            ax.plot(np.cos(phi)*np.sin(grid_latitude),
                    np.sin(phi)*np.sin(grid_latitude),
                    np.ones(phi.shape)*np.cos(grid_latitude),
                    linewidth=grid_linewidth,
                    color=grid_color)

        for grid_longitude in grid_longitudes:
            ax.plot(np.cos(grid_longitude)*np.sin(th),
                    np.sin(grid_longitude)*np.sin(th),
                    np.cos(th),
                    linewidth=grid_linewidth,
                    color=grid_color)

    # -X and Y data are switched for plotting purposes
    for k in range(len(vectors)):
        xs3d = vectors[k][1] * np.array([0, 1])
        ys3d = -vectors[k][0] * np.array([0, 1])
        zs3d = vectors[k][2] * np.array([0, 1])

        # decorated style, with arrow heads
        a = Arrow3D(xs3d, ys3d, zs3d,
                    mutation_scale=vector_mutation,
                    lw=vector_width,
                    arrowstyle='-|>',
                    color='C{0:d}'.format(k))

        ax.add_artist(a)

        if projections:
            b = Arrow3D(xs3d, ys3d, [0, 0],
                        mutation_scale=vector_mutation/2,
                        lw=vector_width/2,
                        linestyle='--',
                        arrowstyle='-|>',
                        color='C{0:d}'.format(k))

            ax.add_artist(b)

            b = Arrow3D([0.9*xs3d[-1], 0], [0.9*ys3d[-1]]*2, [0.9*zs3d[-1]]*2,
                        linestyle=':', lw=vector_width/4,
                        mutation_scale=vector_mutation/4,
                        arrowstyle='-',
                        color='C{0:d}'.format(k))
            ax.add_artist(b)

            c = Arrow3D([0, 0], ys3d, zs3d,
                        mutation_scale=vector_mutation/2,
                        lw=vector_width/2,
                        linestyle='-.',
                        arrowstyle='-|>',
                        color='C{0:d}'.format(k))

            ax.add_artist(c)

            c = Arrow3D([0.9*xs3d[-1]]*2, [0.9*ys3d[-1]]*2, [0.9*zs3d[-1], 0],
                        linestyle=':', lw=vector_width/4,
                        mutation_scale=vector_mutation/4,
                        arrowstyle='-',
                        color='C{0:d}'.format(k))
            ax.add_artist(c)

    # axes labels
    opts = {'horizontalalignment': 'center',
            'verticalalignment': 'center'}

    ax.text(0, -xlpos[0], 0, xlabel[0], **opts)
    ax.text(0, -xlpos[1], 0, xlabel[1], **opts)

    ax.text(ylpos, 0, 0, ylabel, **opts)

    ax.text(0, 0, zlpos, zlabel, **opts)


def plot_surface_quarter(ax, X, Y, Z, C,
                         sphere_alpha=1,
                         colormap=cm.jet,
                         grid_lines=True,
                         grid_longitudes=np.linspace(0, np.pi, 8),
                         grid_latitudes=[np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2],
                         grid_linewidth=0.5,
                         grid_color='black',
                         xlabel = ['$s$', '$p$'],
                         xlpos = [1.2, -1.2],
                         ylabel = '',
                         ylpos = 1.2,
                         zlabel = '$\\epsilon^-$',
                         zlpos = 1.2):

    """
    Plots vectors in the quarter of the sphere:
    """
    #self.plot_axes()
    ax.set_axis_off()
    ax.set_xlim3d(-0.7, 0.7)
    ax.set_ylim3d(-0.7, 0.7)
    ax.set_zlim3d(0, 0.7)

    if grid_lines:
        phi = np.linspace(0, np.pi, 25)
        th = np.linspace(0, np.pi/2, 25)

        for grid_latitude in grid_latitudes:
            ax.plot(np.sin(phi)*np.sin(grid_latitude),
                    -np.cos(phi)*np.sin(grid_latitude),
                    np.ones(phi.shape)*np.cos(grid_latitude),
                    linewidth=grid_linewidth,
                    color=grid_color)

        for grid_longitude in grid_longitudes:
            ax.plot(np.sin(grid_longitude)*np.sin(th),
                    -np.cos(grid_longitude)*np.sin(th),
                    np.cos(th),
                    linewidth=grid_linewidth,
                    color=grid_color)

    # Plot the front half of the sphere:
    ax.plot_surface(Y, -X, Z, rstride=1, cstride=1,
                    facecolors=colormap(C))

    # axes labels
    opts = {'horizontalalignment': 'center',
            'verticalalignment': 'center'}

    ax.text(0, -xlpos[0], 0, xlabel[0], **opts)
    ax.text(0, -xlpos[1], 0, xlabel[1], **opts)

    ax.text(ylpos, 0, 0, ylabel, **opts)

    ax.text(0, 0, zlpos, zlabel, **opts)


def plot_surface_half(ax, X, Y, Z, C,
                         sphere_alpha=1,
                         colormap=cm.jet,
                         grid_lines=True,
                         grid_longitudes=np.linspace(0, np.pi, 8),
                         grid_latitudes=np.arange(np.pi/8, np.pi, np.pi/8),
                         grid_linewidth=0.25,
                         grid_color='gray',
                         xlabel = ['$s$', '$p$'],
                         xlpos = [1.2, -1.2],
                         ylabel = '',
                         ylpos = 1.2,
                         zlabel = ['$\\epsilon^-$', '$\\epsilon^+$'],
                         zlpos = 1.2):

    """
    Plots vectors in the quarter of the sphere:
    """
    #self.plot_axes()
    ax.set_axis_off()
    ax.set_xlim3d(-0.7, 0.7)
    ax.set_ylim3d(-0.7, 0.7)
    ax.set_zlim3d(-0.7, 0.7)

    # Plot the front half of the sphere:
    ax.plot_surface(Y, -X, Z, rstride=1, cstride=1,
                    facecolors=colormap(C))


    if grid_lines:
        phi = np.linspace(0, np.pi, 25)
        th = np.linspace(0, np.pi, 25)

        for grid_latitude in grid_latitudes:
            ax.plot(np.sin(phi)*np.sin(grid_latitude),
                    -np.cos(phi)*np.sin(grid_latitude),
                    np.ones(phi.shape)*np.cos(grid_latitude),
                    linewidth=grid_linewidth,
                    color=grid_color)

        for grid_longitude in grid_longitudes:
            ax.plot(np.sin(grid_longitude)*np.sin(th),
                    -np.cos(grid_longitude)*np.sin(th),
                    np.cos(th),
                    linewidth=grid_linewidth,
                    color=grid_color)

    # axes labels
    opts = {'horizontalalignment': 'center',
            'verticalalignment': 'center'}

    ax.text(0, -xlpos[0], 0, xlabel[0], **opts)
    ax.text(0, -xlpos[1], 0, xlabel[1], **opts)

    ax.text(ylpos, 0, 0, ylabel, **opts)

    ax.text(0, 0, zlpos, zlabel[0], **opts)
    ax.text(0, 0, -zlpos, zlabel[1], **opts)

'''
class PoincareQuarter():
    """Class for plotting data on a quarter of the Poincare sphere.  Valid data
    can be either points, vectors, or qobj objects.

    Attributes
    ----------

    axes : instance {None}
        User supplied Matplotlib axes for Bloch sphere animation.
    fig : instance {None}
        User supplied Matplotlib Figure instance for plotting Bloch sphere.
    font_color : str {'black'}
        Color of font used for Bloch sphere labels.
    font_size : int {20}
        Size of font used for Bloch sphere labels.
    frame_alpha : float {0.1}
        Sets transparency of Bloch sphere frame.
    frame_color : str {'gray'}
        Color of sphere wireframe.
    frame_width : int {1}
        Width of wireframe.
    point_color : list {["b","r","g","#CC6600"]}
        List of colors for Bloch sphere point markers to cycle through.
        i.e. By default, points 0 and 4 will both be blue ('b').
    point_marker : list {["o","s","d","^"]}
        List of point marker shapes to cycle through.
    point_size : list {[25,32,35,45]}
        List of point marker sizes. Note, not all point markers look
        the same size when plotted!
    sphere_alpha : float {0.2}
        Transparency of Bloch sphere itself.
    sphere_color : str {'#FFDDDD'}
        Color of Bloch sphere.
    figsize : list {[7,7]}
        Figure size of Bloch sphere plot.  Best to have both numbers the same;
        otherwise you will have a Bloch sphere that looks like a football.
    vector_color : list {["g","#CC6600","b","r"]}
        List of vector colors to cycle through.
    vector_width : int {5}
        Width of displayed vectors.
    vector_style : str {'-|>', 'simple', 'fancy', ''}
        Vector arrowhead style (from matplotlib's arrow style).
    vector_mutation : int {20}
        Width of vectors arrowhead.
    view : list {[-60,30]}
        Azimuthal and Elevation viewing angles.
    xlabel : list {["$x$",""]}
        List of strings corresponding to +x and -x axes labels, respectively.
    xlpos : list {[1.1,-1.1]}
        Positions of +x and -x labels respectively.
    ylabel : list {["$y$",""]}
        List of strings corresponding to +y and -y axes labels, respectively.
    ylpos : list {[1.2,-1.2]}
        Positions of +y and -y labels respectively.
    zlabel : list {[r'$\\left|0\\right>$',r'$\\left|1\\right>$']}
        List of strings corresponding to +z and -z axes labels, respectively.
    zlpos : list {[1.2,-1.2]}
        Positions of +z and -z labels respectively.

    """
    def __init__(self, fig=None, axes=None, view=None, figsize=None,
                 background=False):

        # Figure and axes
        self.fig = fig
        self.axes = axes
        # Background axes, default = False
        self.background = background
        # The size of the figure in inches, default = [5,5].
        self.figsize = figsize if figsize else [5, 5]
        # Azimuthal and Elvation viewing angles, default = [-60,30].
        self.view = view if view else [-60, 30]
        # Color of Bloch sphere, default = #FFDDDD
        self.sphere_color = '#FFDDDD'
        # Transparency of Bloch sphere, default = 0.2
        self.sphere_alpha = 0.1
        # Color of wireframe, default = 'gray'
        self.frame_color = 'gray'
        # Width of wireframe, default = 1
        self.frame_width = 1
        # Transparency of wireframe, default = 0.2
        self.frame_alpha = 0.2
        # Labels for x-axis (in LaTex), default = ['$x$', '']
        self.xlabel = ['$x$', '']
        # Position of x-axis labels, default = [1.2, -1.2]
        self.xlpos = [1.2, -1.2]
        # Labels for y-axis (in LaTex), default = ['$y$', '']
        self.ylabel = ['$y$', '']
        # Position of y-axis labels, default = [1.1, -1.1]
        self.ylpos = [1.2, -1.2]
        # Labels for z-axis (in LaTex),
        # default = [r'$\left|0\right>$', r'$\left|1\right>$']
        self.zlabel = ['', '']
        # Position of z-axis labels, default = [1.2, -1.2]
        self.zlpos = [1.2, -1.2]
        # ---font options

        # ---vector options---
        # List of colors for Bloch vectors, default = ['b','g','r','y']
        self.vector_color = ['g', '#CC6600', 'b', 'r']
        #: Width of Bloch vectors, default = 5
        self.vector_width = 3
        #: Style of Bloch vectors, default = '-|>' (or 'simple')
        self.vector_style = '-|>'
        #: Sets the width of the vectors arrowhead
        self.vector_mutation = 20

        # ---point options---
        # List of colors for Bloch point markers, default = ['b','g','r','y']
        self.point_color = ['b', 'r', 'g', '#CC6600']
        # Size of point markers, default = 25
        self.point_size = [25, 32, 35, 45]
        # Shape of point markers, default = ['o','^','d','s']
        self.point_marker = ['o', 's', 'd', '^']

        # ---data lists---
        # Data for point markers
        self.points = []
        # Data for Bloch vectors
        self.vectors = []
        # Data for annotations
        self.annotations = []
        # Number of times sphere has been saved
        self.savenum = 0
        # Style of points, 'm' for multiple colors, 's' for single color
        self.point_style = []

        # status of rendering
        self._rendered = False

    def __str__(self):
        s = ""
        s += "Bloch data:\n"
        s += "-----------\n"
        s += "Number of points:  " + str(len(self.points)) + "\n"
        s += "Number of vectors: " + str(len(self.vectors)) + "\n"
        s += "\n"
        s += "Bloch sphere properties:\n"
        s += "------------------------\n"
        s += "font_color:      " + str(self.font_color) + "\n"
        s += "font_size:       " + str(self.font_size) + "\n"
        s += "frame_alpha:     " + str(self.frame_alpha) + "\n"
        s += "frame_color:     " + str(self.frame_color) + "\n"
        s += "frame_width:     " + str(self.frame_width) + "\n"
        s += "point_color:     " + str(self.point_color) + "\n"
        s += "point_marker:    " + str(self.point_marker) + "\n"
        s += "point_size:      " + str(self.point_size) + "\n"
        s += "sphere_alpha:    " + str(self.sphere_alpha) + "\n"
        s += "sphere_color:    " + str(self.sphere_color) + "\n"
        s += "figsize:         " + str(self.figsize) + "\n"
        s += "vector_color:    " + str(self.vector_color) + "\n"
        s += "vector_width:    " + str(self.vector_width) + "\n"
        s += "vector_style:    " + str(self.vector_style) + "\n"
        s += "vector_mutation: " + str(self.vector_mutation) + "\n"
        s += "view:            " + str(self.view) + "\n"
        s += "xlabel:          " + str(self.xlabel) + "\n"
        s += "xlpos:           " + str(self.xlpos) + "\n"
        s += "ylabel:          " + str(self.ylabel) + "\n"
        s += "ylpos:           " + str(self.ylpos) + "\n"
        s += "zlabel:          " + str(self.zlabel) + "\n"
        s += "zlpos:           " + str(self.zlpos) + "\n"
        return s

    def _repr_png_(self):
        from IPython.core.pylabtools import print_figure
        self.render()
        fig_data = print_figure(self.fig, 'png')
        plt.close(self.fig)
        return fig_data

    def _repr_svg_(self):
        from IPython.core.pylabtools import print_figure
        self.render()
        fig_data = print_figure(self.fig, 'svg').decode('utf-8')
        plt.close(self.fig)
        return fig_data

    def clear(self):
        """Resets Bloch sphere data sets to empty.
        """
        self.points = []
        self.vectors = []
        self.point_style = []
        self.annotations = []

    def add_points(self, points, meth='s'):
        """Add a list of data points to bloch sphere.

        Parameters
        ----------
        points : array/list
            Collection of data points.

        meth : str {'s', 'm', 'l'}
            Type of points to plot, use 'm' for multicolored, 'l' for points
            connected with a line.

        """
        if not isinstance(points[0], (list, ndarray)):
            points = [[points[0]], [points[1]], [points[2]]]
        points = array(points)
        if meth == 's':
            if len(points[0]) == 1:
                pnts = array([[points[0][0]], [points[1][0]], [points[2][0]]])
                pnts = append(pnts, points, axis=1)
            else:
                pnts = points
            self.points.append(pnts)
            self.point_style.append('s')
        elif meth == 'l':
            self.points.append(points)
            self.point_style.append('l')
        else:
            self.points.append(points)
            self.point_style.append('m')

    def add_states(self, state, kind='vector'):
        """Add a state vector Qobj to Bloch sphere.

        Parameters
        ----------
        state : qobj
            Input state vector.

        kind : str {'vector','point'}
            Type of object to plot.

        """
        if isinstance(state, Qobj):
            state = [state]

        for st in state:
            vec = [expect(sigmax(), st),
                   expect(sigmay(), st),
                   expect(sigmaz(), st)]

            if kind == 'vector':
                self.add_vectors(vec)
            elif kind == 'point':
                self.add_points(vec)

    def add_vectors(self, vectors):
        """Add a list of vectors to Bloch sphere.

        Parameters
        ----------
        vectors : array_like
            Array with vectors of unit length or smaller.

        """
        if isinstance(vectors[0], (list, ndarray)):
            for vec in vectors:
                self.vectors.append(vec)
        else:
            self.vectors.append(vectors)

    def add_annotation(self, state_or_vector, text, **kwargs):
        """Add a text or LaTeX annotation to Bloch sphere,
        parametrized by a qubit state or a vector.

        Parameters
        ----------
        state_or_vector : Qobj/array/list/tuple
            Position for the annotaion.
            Qobj of a qubit or a vector of 3 elements.

        text : str/unicode
            Annotation text.
            You can use LaTeX, but remember to use raw string
            e.g. r"$\\langle x \\rangle$"
            or escape backslashes
            e.g. "$\\\\langle x \\\\rangle$".

        **kwargs :
            Options as for mplot3d.axes3d.text, including:
            fontsize, color, horizontalalignment, verticalalignment.

        """
        if isinstance(state_or_vector, Qobj):
            vec = [expect(sigmax(), state_or_vector),
                   expect(sigmay(), state_or_vector),
                   expect(sigmaz(), state_or_vector)]
        elif isinstance(state_or_vector, (list, ndarray, tuple)) \
                and len(state_or_vector) == 3:
            vec = state_or_vector
        else:
            raise Exception("Position needs to be specified by a qubit " +
                            "state or a 3D vector.")
        self.annotations.append({'position': vec,
                                 'text': text,
                                 'opts': kwargs})

    def make_sphere(self):
        """
        Plots Bloch sphere and data sets.
        """
        self.render(self.fig, self.axes)

    def render(self, fig=None, axes=None):
        """
        Render the Bloch sphere and its data sets in on given figure and axes.
        """
        if self._rendered:
            self.axes.clear()

        self._rendered = True

        # Figure instance for Bloch sphere plot
        if not fig:
            self.fig = plt.figure(figsize=self.figsize)

        if not axes:
            self.axes = Axes3D(self.fig, azim=self.view[0], elev=self.view[1])

        if self.background:
            self.axes.clear()
            self.axes.set_xlim3d(-1.3, 1.3)
            self.axes.set_ylim3d(-1.3, 1.3)
            self.axes.set_zlim3d(-1.3, 1.3)
        else:
            #self.plot_axes()
            self.axes.set_axis_off()
            self.axes.set_xlim3d(-0.7, 0.7)
            self.axes.set_ylim3d(-0.7, 0.7)
            self.axes.set_zlim3d(-0.7, 0.7)

        self.axes.grid(False)
        self.plot_back()
        self.plot_points()
        self.plot_vectors()
        self.plot_front()
        self.plot_axes_labels()
        self.plot_annotations()

    def plot_back(self):
        # back half of sphere
        phi = linspace(-pi/2, pi/2, 25)
        th = linspace(0, pi/2, 25)

        x = outer(zeros(size(phi)), zeros(size(th)))
        y = outer(sin(phi), sin(th))
        z = outer(ones(size(phi)), cos(th))
        self.axes.plot_surface(x, y, z, rstride=2, cstride=2,
                               color=self.sphere_color, linewidth=0,
                               alpha=self.sphere_alpha)
        # wireframe
        """self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5,
                                 color=self.frame_color,
                                 alpha=self.frame_alpha)"""

        x = cos(phi)
        y = sin(phi)
        z = zeros(size(th))

        self.axes.plot_trisurf(x, y, z,
                               color=self.sphere_color, edgecolor='none',
                               alpha=self.sphere_alpha)
        # wireframe
        """self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5,
                                 color=self.frame_color,
                                 alpha=self.frame_alpha)"""
        # equator
        """self.axes.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='z',
                       lw=self.frame_width, color=self.frame_color)
        self.axes.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='x',
                       lw=self.frame_width, color=self.frame_color)"""

    def plot_front(self):
        # front half of sphere
        phi = linspace(-pi/2, pi/2, 25)
        th = linspace(0, pi/2, 25)

        x = outer(cos(phi), sin(th))
        y = outer(sin(phi), sin(th))
        z = outer(ones(size(u)), cos(th))
        self.axes.plot_surface(x, y, z, rstride=2, cstride=2,
                               color=self.sphere_color, linewidth=0,
                               alpha=self.sphere_alpha)
        # wireframe
        """self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5,
                                 color=self.frame_color,
                                 alpha=self.frame_alpha)"""
        # equator
        """self.axes.plot(1.0 * cos(u), 1.0 * sin(u),
                       zs=0, zdir='z', lw=self.frame_width,
                       color=self.frame_color)
        self.axes.plot(1.0 * cos(u), 1.0 * sin(u),
                       zs=0, zdir='x', lw=self.frame_width,
                       color=self.frame_color)"""

    def plot_axes(self):
        # axes
        span = linspace(-1.0, 1.0, 2)
        self.axes.plot(span, 0 * span, zs=0, zdir='z', label='X',
                       lw=self.frame_width, color=self.frame_color)
        self.axes.plot(0 * span, span, zs=0, zdir='z', label='Y',
                       lw=self.frame_width, color=self.frame_color)
        self.axes.plot(0 * span, span, zs=0, zdir='y', label='Z',
                       lw=self.frame_width, color=self.frame_color)

    def plot_axes_labels(self):
        # axes labels
        opts = {'horizontalalignment': 'center',
                'verticalalignment': 'center'}
        self.axes.text(0, -self.xlpos[0], 0, self.xlabel[0], **opts)
        self.axes.text(0, -self.xlpos[1], 0, self.xlabel[1], **opts)

        self.axes.text(self.ylpos[0], 0, 0, self.ylabel[0], **opts)
        self.axes.text(self.ylpos[1], 0, 0, self.ylabel[1], **opts)

        self.axes.text(0, 0, self.zlpos[0], self.zlabel[0], **opts)
        self.axes.text(0, 0, self.zlpos[1], self.zlabel[1], **opts)

        for a in (self.axes.w_xaxis.get_ticklines() +
                  self.axes.w_xaxis.get_ticklabels()):
            a.set_visible(False)
        for a in (self.axes.w_yaxis.get_ticklines() +
                  self.axes.w_yaxis.get_ticklabels()):
            a.set_visible(False)
        for a in (self.axes.w_zaxis.get_ticklines() +
                  self.axes.w_zaxis.get_ticklabels()):
            a.set_visible(False)

    def plot_vectors(self):
        # -X and Y data are switched for plotting purposes
        for k in range(len(self.vectors)):

            xs3d = self.vectors[k][1] * array([0, 1])
            ys3d = -self.vectors[k][0] * array([0, 1])
            zs3d = self.vectors[k][2] * array([0, 1])

            color = self.vector_color[mod(k, len(self.vector_color))]

            if self.vector_style == '':
                # simple line style
                self.axes.plot(xs3d, ys3d, zs3d,
                               zs=0, zdir='z', label='Z',
                               lw=self.vector_width, color=color)
            else:
                # decorated style, with arrow heads
                a = Arrow3D(xs3d, ys3d, zs3d,
                            mutation_scale=self.vector_mutation,
                            lw=self.vector_width,
                            arrowstyle=self.vector_style,
                            color=color)

                self.axes.add_artist(a)

    def plot_points(self):
        # -X and Y data are switched for plotting purposes
        for k in range(len(self.points)):
            num = len(self.points[k][0])
            dist = [sqrt(self.points[k][0][j] ** 2 +
                         self.points[k][1][j] ** 2 +
                         self.points[k][2][j] ** 2) for j in range(num)]
            if any(abs(dist - dist[0]) / dist[0] > 1e-12):
                # combine arrays so that they can be sorted together
                zipped = list(zip(dist, range(num)))
                zipped.sort()  # sort rates from lowest to highest
                dist, indperm = zip(*zipped)
                indperm = array(indperm)
            else:
                indperm = arange(num)
            if self.point_style[k] == 's':
                self.axes.scatter(
                    real(self.points[k][1][indperm]),
                    - real(self.points[k][0][indperm]),
                    real(self.points[k][2][indperm]),
                    s=self.point_size[mod(k, len(self.point_size))],
                    alpha=1,
                    edgecolor='none',
                    zdir='z',
                    color=self.point_color[mod(k, len(self.point_color))],
                    marker=self.point_marker[mod(k, len(self.point_marker))])

            elif self.point_style[k] == 'm':
                pnt_colors = array(self.point_color *
                                   int(ceil(num / float(len(self.point_color)))))

                pnt_colors = pnt_colors[0:num]
                pnt_colors = list(pnt_colors[indperm])
                marker = self.point_marker[mod(k, len(self.point_marker))]
                s = self.point_size[mod(k, len(self.point_size))]
                self.axes.scatter(real(self.points[k][1][indperm]),
                                  -real(self.points[k][0][indperm]),
                                  real(self.points[k][2][indperm]),
                                  s=s, alpha=1, edgecolor='none',
                                  zdir='z', color=pnt_colors,
                                  marker=marker)

            elif self.point_style[k] == 'l':
                color = self.point_color[mod(k, len(self.point_color))]
                self.axes.plot(real(self.points[k][1]),
                               -real(self.points[k][0]),
                               real(self.points[k][2]),
                               alpha=0.75, zdir='z',
                               color=color)

    def plot_annotations(self):
        # -X and Y data are switched for plotting purposes
        for annotation in self.annotations:
            vec = annotation['position']
            opts = {'fontsize': self.font_size,
                    'color': self.font_color,
                    'horizontalalignment': 'center',
                    'verticalalignment': 'center'}
            opts.update(annotation['opts'])
            self.axes.text(vec[1], -vec[0], vec[2],
                           annotation['text'], **opts)

    def show(self):
        """
        Display Bloch sphere and corresponding data sets.
        """
        self.render(self.fig, self.axes)
        if self.fig:
            plt.show(self.fig)
'''
"""
def _hide_tick_lines_and_labels(axis):
    '''
    Set visible property of ticklines and ticklabels of an axis to False
    '''
    for a in axis.get_ticklines() + axis.get_ticklabels():
        a.set_visible(False)"""
