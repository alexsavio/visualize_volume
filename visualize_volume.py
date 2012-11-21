#!/usr/bin/python

#-------------------------------------------------------------------------------------
# Matplotlib option
#-------------------------------------------------------------------------------
def visualize_3slices (vol, x=None, y=None, z=None, fig=None):
    import numpy as np
    import scipy.ndimage as scn
    import pylab as pl

    from matplotlib.widgets import Slider, Button

    if not x:
        x = np.floor(vol.shape[0]/2)
    if not y:
        y = np.floor(vol.shape[1]/2)
    if not z:
        z = np.floor(vol.shape[2]/2)

    if not fig:
        fig = pl.figure()
    else:
        fig = pl.figure(fig.number)

    pl.subplot (2,2,1)
    pl.imshow(scn.rotate(vol[x,...],90), cmap=pl.cm.gray)
    #pl.imshow(vol[x,...], cmap=pl.cm.gray)
    pl.title ('X == ' + str(x))

    pl.subplot (2,2,2)
    pl.imshow(scn.rotate(vol[:,y,:], 90), cmap=pl.cm.gray)
    pl.title ('Y == ' + str(y))

    pl.subplot (2,2,3)
    pl.imshow(scn.rotate(vol[...,z], 90), cmap=pl.cm.gray)
    pl.title ('Z == ' + str(z))

    pl.subplot (2,2,4)

    axcolor = 'lightgoldenrodyellow'
    xval = pl.axes([0.50, 0.20, 0.40, 0.03], axisbg=axcolor)
    yval = pl.axes([0.50, 0.15, 0.40, 0.03], axisbg=axcolor)
    zval = pl.axes([0.50, 0.10, 0.40, 0.03], axisbg=axcolor)

    xsli = Slider(xval, 'X', 0, vol.shape[0]-1, valinit=x, valfmt='%i')
    ysli = Slider(yval, 'Y', 0, vol.shape[1]-1, valinit=y, valfmt='%i')
    zsli = Slider(zval, 'Z', 0, vol.shape[2]-1, valinit=z, valfmt='%i')

    def update(val):
        x = np.around(xsli.val)
        y = np.around(ysli.val)
        z = np.around(zsli.val)
        #debug_here()
        visualize_3slices (vol, x, y, z, fig)
        #fig.set_data()

    xsli.on_changed(update)
    ysli.on_changed(update)
    zsli.on_changed(update)

    pl.show()

    return fig
#-------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------
# Mayavi2 options
#-------------------------------------------------------------------------------------
def visualize_cutplanes (img, first_idx    = 10, 
                              second_idx   = 10,
                              first_plane  = 'x_axes',
                              second_plane = 'y_axes',
                        ):

    from mayavi import mlab
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(img),
                            plane_orientation=first_plane,
                            slice_index=first_idx,
                            colormap='gray',
                        )
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(img),
                            plane_orientation=second_plane,
                            slice_index=second_idx,
                            colormap='gray',
                        )
    mlab.outline()

#-------------------------------------------------------------------------------------
def visualize_dynplane (img):
    from mayavi import mlab
    src = mlab.pipeline.scalar_field(img)
    mlab.pipeline.iso_surface(src, contours=[img.min()+0.1*img.ptp(), ], opacity=0.1)
    mlab.pipeline.iso_surface(src, contours=[img.max()-0.1*img.ptp(), ],)
    mlab.pipeline.image_plane_widget(src,
                            plane_orientation='z_axes',
                            slice_index=10,
                            colormap='gray',
                        )

#-------------------------------------------------------------------------------------
def visualize_contour (img):
    from mayavi import mlab
    mlab.contour3d(img, colormap='gray')

#-------------------------------------------------------------------------------------
def visualize_render (img, vmin=0, vmax=0.8):
    from mayavi import mlab
    mlab.pipeline.volume(mlab.pipeline.scalar_field(img), vmin=vmin, vmax=vmax, colormap='gray')

