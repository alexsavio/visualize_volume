#!/usr/bin/python

#-------------------------------------------------------------------------------------
# Matplotlib option
#-------------------------------------------------------------------------------
def visualize_3slices (vol, vol2=None, x=None, y=None, z=None, fig=None, vol2_colormap = None, vol2_trans_val = 0, interpolation = 'nearest'):
    import numpy as np
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

    try:
        if not np.ma.is_masked(vol2):
            vol2 = np.ma.masked_equal(vol2, 0)
    except:
        pass

    if vol2_colormap == None:
        vol2_colormap = pl.cm.hot

    pl.subplot (2,2,1)
    pl.axis('off')
    pl.imshow(np.rot90(vol[x,...]), cmap=pl.cm.gray, interpolation=interpolation)
    try:
        pl.imshow(np.rot90(vol2[x,...]), cmap=vol2_colormap, interpolation=interpolation)
    except:
        pass
    #pl.imshow(vol[x,...], cmap=pl.cm.gray)
    pl.title ('X:' + str(int(x)))

    pl.subplot (2,2,2)
    pl.axis('off')
    pl.imshow(np.rot90(vol[:,y,:]), cmap=pl.cm.gray, interpolation=interpolation)
    try:
        pl.imshow(np.rot90(vol2[:,y,:]), cmap=vol2_colormap, interpolation=interpolation)
    except:
        pass

    pl.title ('Y:' + str(int(y)))

    pl.subplot (2,2,3)
    pl.axis ('off')
    pl.imshow(np.rot90(vol[...,z]), cmap=pl.cm.gray, interpolation=interpolation)
    try:
        pl.imshow(np.rot90(vol2[...,z]), cmap=vol2_colormap, interpolation=interpolation)
    except:
        pass
    pl.title ('Z:' + str(int(z)))

    pl.subplot (2,2,4)

    pl.axis('off')
    axcolor = 'lightgoldenrodyellow'
    xval = pl.axes([0.60, 0.20, 0.20, 0.03], axisbg=axcolor)
    yval = pl.axes([0.60, 0.15, 0.20, 0.03], axisbg=axcolor)
    zval = pl.axes([0.60, 0.10, 0.20, 0.03], axisbg=axcolor)

    xsli = Slider(xval, 'X', 0, vol.shape[0]-1, valinit=x, valfmt='%i')
    ysli = Slider(yval, 'Y', 0, vol.shape[1]-1, valinit=y, valfmt='%i')
    zsli = Slider(zval, 'Z', 0, vol.shape[2]-1, valinit=z, valfmt='%i')

    def update(val):
        x = np.around(xsli.val)
        y = np.around(ysli.val)
        z = np.around(zsli.val)
        #debug_here()
        visualize_3slices (vol, vol2, x, y, z, fig)
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

#-------------------------------------------------------------------------------------
# pyqtgraph option
#-------------------------------------------------------------------------------
#def visualize (vol):

#    import sys
#    import numpy as np
#    import nibabel as nib

#    f = '/old_data/data/vbm_svm/subjects_nifti/control_OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.nii'
#    s = nib.load(f).get_data()
#    vol = s.copy()

#    import numpy as np
#    import scipy
#    from pyqtgraph.Qt import QtCore, QtGui
#    import pyqtgraph as pg

#    app = QtGui.QApplication([])

#    ## Create window with three ImageView widgets
#    win = QtGui.QMainWindow()
#    win.resize(800,800)
#    cw = QtGui.QWidget()
#    win.setCentralWidget(cw)
#    l = QtGui.QGridLayout()
#    cw.setLayout(l)
#    imv1 = pg.ImageView()
#    imv2 = pg.ImageView()
#    imv3 = pg.ImageView()
#    l.addWidget(imv1, 0, 0)
#    l.addWidget(imv2, 1, 0)
#    l.addWidget(imv3, 0, 1)
#    win.show()

#    roi = pg.TestROI(pos=[10, 10], size=pg.Point(2,2), pen='r') 
#    imv1.addItem(roi)

#    def update():
#        global vol, imv1, imv2, imv3
#        d2 = roi.getArrayRegion(vol, imv1.imageItem, axes=(1,2))
#        imv2.setImage(d2)

#    imv1.setImage(vol)

#    volmin = np.min(vol)
#    volmax = np.max(vol)
#    imv1.setHistogramRange(volmin, volmax)

#    lowlvl = 0
#    uprlvl = 1
#    if volmin < 0: 
#        lowlvl = volmin

#    if volmax > 1: 
#        uprlvl = volmax

#    imv1.setLevels(lowlvl, uprlvl)

#    update()
#    
#    if sys.flags.interactive != 1:
#        app.exec_()
















