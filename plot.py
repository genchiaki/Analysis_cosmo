import yt
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from yt.data_objects.particle_filters import add_particle_filter
#from yt.analysis_modules.halo_finding.api import HaloFinder
from yt.utilities.physical_constants import \
    gravitational_constant_cgs as G
import numpy as np
import struct
import os
import sys
from yt.visualization.api import get_multi_plot
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
from yt.units import mh, kboltz, G
pi = 3.14159265

WIN10 = True

RAY    = True
RAY_C  = True
SLICE  = False # True
PROF   = False # True
TABLE  = False
PRESEN = False

extension = 'pdf'

if WIN10:
    import matplotlib.font_manager as font_manager
    font_dirs = ['/mnt/c/Windows/Fonts', ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    font_list = font_manager.createFontList(font_files)
    font_manager.fontManager.ttflist.extend(font_list)

if WIN10:
    plt.rcParams['font.family'] = 'Times New Roman'
else:
    plt.rcParams['font.family'] = 'P052'
#   plt.rcParams['font.family'] = 'stix'
plt.rcParams["mathtext.fontset"] = "stix"
fontsize_suptitle = 32
fontsize_suptitle_s = 18
fontsize_title    = 32
fontsize_title_s  = 28
fontsize_boxsize  = 32
fontsize_label    = 32
fontsize_cblabel  = 28
fontsize_tick     = 28
fontsize_label_s  = 22
fontsize_cblabel_s= 22
fontsize_tick_s   = 22
fontsize_legend   = 20
fontsize_legend_s = 16

indir = 'fig2'
#indir = 'test512-L2_1stP3m_TestB'

HydrogenFractionByMass   = 0.76
DeuteriumToHydrogenRatio = 3.4e-5 * 2.0
HeliumToHydrogenRatio    = (1.0 - HydrogenFractionByMass) / HydrogenFractionByMass
SolarMetalFractionByMass = 0.01295
SolarIronAbundance = 7.50

def Q_HI(m):
    x = np.log10(m)
    x2 = x * x
    if m > 9 and m <= 500:
        return 10.0**(43.61 + 4.9*x   - 0.83*x2)
    elif m > 5 and m <= 9:
        return 10.0**(39.29 + 8.55*x)
    else:
        return 0

def Q_HeI(m):
    x = np.log10(m)
    x2 = x * x
    if m > 9 and m <= 500:
        return 10.0**(42.51 + 5.69*x  - 1.01*x2)
    elif m > 5 and m <= 9:
        return 10.0**(29.24 + 18.49*x)
    else:
        return 0

def Q_HeII(m):
    x = np.log10(m)
    x2 = x * x
    if m > 9 and m <= 500:
        return 10.0**(26.71 + 18.14*x - 3.58*x2)
    elif m > 5 and m <= 9:
        return 10.0**(26.71 + 18.14*x - 3.58*x2)
    else:
        return 0

def Q_LW(m):
    x = np.log10(m)
    x2 = x * x
    if m > 9 and m <= 500:
        return 10.0**(44.03 + 4.59*x  - 0.77*x2)
    elif m > 5 and m <= 9:
        return 10.0**(44.03 + 4.59*x  - 0.77*x2)
    else:
        return 0
 
def f_shield_H2I_WH11(NH2I, T):
    x = NH2I / 5e14
#   print("%13.5e %13.5e" % (NH2I, x))
    b_doppler = np.sqrt(2.0 * kboltz.d * T / mh.d) / 1e5
#   print("%13.5e %13.5e %13.5e %13.5e" % (kboltz.d, T, mh.d, b_doppler))
    f_shield = 0.965 / (1.0 + x/b_doppler)**1.1 + 0.035 * np.exp(-8.5e-4 * np.sqrt(1.0 + x))/np.sqrt(1.0 + x)
    return f_shield

def Jeans_length(nH, T):
    rho = nH * mh.d / HydrogenFractionByMass
    gamma = 5.0/3.0
    mu    = 1.23 * mh.d
    cs2 = gamma * kboltz.d * T / mu
    return np.sqrt(pi * cs2 / (G.d * rho))

def t_ff(nH):
    rho = nH * mh.d / HydrogenFractionByMass
    return np.sqrt(3.0*pi/(32.0*G.d*rho))

def k7(T):
    return 3.0e-16 * (T/3.e2)**0.95 * np.exp(-T/9.32e3)

def k8(T):
    return 1.35e-9*(T**9.8493e-2 + 3.2852e-1 * T**5.5610e-1 + 2.771e-7 * T**2.1826) \
         / (1. + 6.191e-3 * T**1.0461 + 8.9712e-11 * T**3.0424 + 3.2576e-14 * T**3.7741)

#     k7 = 3.0e-16_DKIND * (T/3.e2_DKIND)**0.95_DKIND
#    &   * exp(-T/9.32e3_DKIND) / kunit

#     k8 = 1.35e-9_DKIND*(T**9.8493e-2_DKIND + 3.2852e-1_DKIND
#    &   * T**5.5610e-1_DKIND + 2.771e-7_DKIND * T**2.1826_DKIND)
#    &   / (1._DKIND + 6.191e-3_DKIND * T**1.0461_DKIND
#    &   + 8.9712e-11_DKIND * T**3.0424_DKIND
#    &   + 3.2576e-14_DKIND * T**3.7741_DKIND) / kunit

#          1.35\E{-9} \frac{T^{9.8493\E{-2}} + 3.2852\E{-1}  T^{5.5610\E{-1}} + 2.771\E{-7}  T^{2.1826}}
#          {1.0 + 6.191\E{-3}  T^{1.0461} + 8.9712\E{-11}  T^{3.0424} + 3.2576\E{-14}  T^{3.7741}}

#def z2t(redshift):
#    return 10.3438263 * (141.0 / (redshift + 1.0))**(3.0/2.0) ## Myr
#print(z2t(25.034418454577) - z2t(25.075830449034), t_ff(1e6)/3.1557e13)

TopGridDimension = int(sys.argv[1])
NestedLevel      = int(sys.argv[2])
Threshold        = sys.argv[3]
Test             = sys.argv[4]
Test_name = ['TestA', 'TestB', 'TestC']
Test_text = ['(a) Test A', '(b) Test B', '(c) Test C']
inumber0 = int(sys.argv[5])
inumber1 = int(sys.argv[6])
inumber2 = int(sys.argv[7])

colors  = ['red', 'green', 'blue']


if RAY:
    i_rad   =  0
    i_nH    =  1
    i_Tg    =  2
    i_yHI   =  3
    i_yelec =  4
    i_yH2I  =  5
    i_yHM   =  6
    i_kHI   =  7
    i_kH2It =  8
    i_kH2I0 =  9
    i_sH2I  = 10
    i_NH2I  = 11
    i_kH2I  = 12
    i_sH2I1 = 13
    i_NH2I1 = 14
    i_kH2I1 = 15
    i_sH2I3 = 16
    i_NH2I3 = 17
    i_kH2I3 = 18
    i_kHDIt = 19
    i_kHDI0 = 20
    i_sHDI  = 21
    i_NHDI  = 22
    i_kHDI  = 23
    i_sHDI1 = 24
    i_NHDI1 = 25
    i_kHDI1 = 26
    i_sHDI3 = 27
    i_NHDI3 = 28
    i_kHDI3 = 29

    data_prof = [None] * 3

    for inumber in range(inumber0, inumber1+1, inumber2):

        # READ DATA
#       fn_prof = [
#           'test512-L2_1stP3_TestA/Profile-1d_%04d.dat' % inumber
#         , 'test512-L2_1stP3_TestB/Profile-1d_%04d.dat' % inumber
#         , 'test512-L2_1stP3_TestC/Profile-1d_%04d.dat' % inumber
#            ]

        for iTest in range(3):
#       for iTest in [1]:
            outdir = 'test%d-L%d_%s_%s/' % (TopGridDimension, NestedLevel, Threshold, Test_name[iTest])
            fn_prof = outdir + 'Profile-1d_%04d.dat' % (inumber)
            print(fn_prof)
            data_prof[iTest] = np.loadtxt(fn_prof)

        fp_Dfront = open(indir + '/ray_Dfront_test%d-L%d_%s_%04d.dat' % (TopGridDimension, NestedLevel, Threshold, inumber), mode='w')
        fp_H2ring = open(indir + '/ray_H2ring_test%d-L%d_%s_%04d.dat' % (TopGridDimension, NestedLevel, Threshold, inumber), mode='w')

        # Time
        outdir = 'test%d-L%d_%s_%s/' % (TopGridDimension, NestedLevel, Threshold, Test_name[iTest])
        fn_star = outdir + 'Stars_PopIII_%04d.dat' % (inumber)
        starfp = open(fn_star, 'rb')
        nPopIII = struct.unpack('i', starfp.read(4))[0]
        xPopIII = np.zeros(nPopIII)
        yPopIII = np.zeros(nPopIII)
        zPopIII = np.zeros(nPopIII)
        cPopIII = np.zeros(nPopIII)
        tPopIII = np.zeros(nPopIII)
        dPopIII = np.zeros(nPopIII)
        MPopIII = np.zeros(nPopIII)
        for iPopIII in range(nPopIII):
            xPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            yPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            zPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            cPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            dPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            tPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            MPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
        starfp.close()
        iPopIII0 = np.argmin(cPopIII)
        t_SF = tPopIII[iPopIII0]

        # maximum nH and yH2
#       print("nH max")
        x_nHmax = np.zeros(3)
        x_Dfront_min = np.zeros(3)
        x_Dfront_max = np.zeros(3)
        x_H2ring_min = np.zeros(3)
        x_H2ring_max = np.zeros(3)
        for iTest in range(3):
#       for iTest in [1]:
#           print("#", Test_name[iTest])
            i_nHmax = np.argmax(data_prof[iTest][:, i_nH])
            x_nHmax[iTest] = data_prof[iTest][i_nHmax, i_rad]
            nH_nHmax       = data_prof[iTest][i_nHmax, i_nH]
            Tg_nHmax       = data_prof[iTest][i_nHmax, i_Tg]
            yHI_nHmax      = data_prof[iTest][i_nHmax, i_yHI  ]
            yelec_nHmax    = data_prof[iTest][i_nHmax, i_yelec]
            yH2I_nHmax     = data_prof[iTest][i_nHmax, i_yH2I ]
            yHM_nHmax      = data_prof[iTest][i_nHmax, i_yHM  ]
            NH2I0_nHmax = data_prof[iTest][i_nHmax, i_NH2I ]
            NH2I1_nHmax = data_prof[iTest][i_nHmax, i_NH2I1]
            NH2I3_nHmax = data_prof[iTest][i_nHmax, i_NH2I3]
            if iTest == 0:
                NH2I_nHmax = data_prof[iTest][i_nHmax, i_NH2I ]
            if iTest == 1:
                NH2I_nHmax = data_prof[iTest][i_nHmax, i_NH2I1]
            if iTest == 2:
                NH2I_nHmax = data_prof[iTest][i_nHmax, i_NH2I3]

            if iTest == 0:
                kH2I_nHmax = data_prof[iTest][i_nHmax, i_kH2I ]
            if iTest == 1:
                kH2I_nHmax = data_prof[iTest][i_nHmax, i_kH2I1]
            if iTest == 2:
                kH2I_nHmax = data_prof[iTest][i_nHmax, i_kH2I3]

            # shielding fraction
            f_shield = f_shield_H2I_WH11(NH2I_nHmax, Tg_nHmax)
            # shell width
            i_Dfront, = np.where(data_prof[iTest][:, i_nH] > 0.1 * nH_nHmax)
            if False:
                x_Dfront_min[iTest] = np.min(data_prof[iTest][i_Dfront, i_rad])
                x_Dfront_max[iTest] = np.max(data_prof[iTest][i_Dfront, i_rad])
            else:
                i_Dfront_min = np.min(i_Dfront)
                i_Dfront_max = np.max(i_Dfront)
                x_Dfront_min[iTest] = 10.0**np.interp(np.log10(0.1 * nH_nHmax)
                    , np.log10(data_prof[iTest][i_Dfront_min-1:i_Dfront_min+1, i_nH ])
                    , np.log10(data_prof[iTest][i_Dfront_min-1:i_Dfront_min+1, i_rad]))
                x_Dfront_max[iTest] = 10.0**np.interp(np.log10(0.1 * nH_nHmax)
                    , np.log10(data_prof[iTest][i_Dfront_max+1:i_Dfront_max-1:-1, i_nH ])
                    , np.log10(data_prof[iTest][i_Dfront_max+1:i_Dfront_max-1:-1, i_rad]))
            dx_Dfront = x_Dfront_max[iTest] - x_Dfront_min[iTest]

            print("#", Test_name[iTest])
            if iTest == 0:
                # J_LW
                # reference values
                E_LW = 1.60184e-12 * 12.8 # erg
                x_ref = 100.00 # pc
                Q_ref = 8e47 # s^-1
                fsh_ref = 1.0
                Dnu = 1.60184e-12 * (13.6-11.2) / 6.63e-27
                J_LW = (E_LW * Q_ref) / (4.0 * np.pi * (x_ref * 3.0856e18)**2) / Dnu / (4.0*np.pi) * fsh_ref
                print("J_LW %13.7e" % J_LW)

            if iTest == 0:
                # calculate the H2 dissociation/formation timescales
                # from simulation
                print("%13.5e %13.5e" % (
                     1.0 / kH2I_nHmax
                   , yH2I_nHmax / (k8(Tg_nHmax) * yHI_nHmax * yHM_nHmax * nH_nHmax)))
                # reference values
                x_ref = 0.16 # pc
                Q_ref = 8e47 # s^-1
                fsh_ref = 0.7
                tdiss = (4.0 * np.pi * (x_ref * 3.0856e18)**2) / (Q_ref * 3.71e-18) / fsh_ref
                print("tdiss %13.7f %13.7e %13.7f %13.5e" % (
                    x_ref
                  , Q_ref
                  , fsh_ref
                  , tdiss  ))
                yH2I_ref = 2e-10
                yHI_ref = 1.0
                yHM_ref = 5e-14
                nH_ref  = 1e6
                k8_ref = 3e-9
                tform = yH2I_ref / (k8_ref * yHI_ref * yHM_ref * nH_ref)
                print("tform %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e" % (
                    yH2I_ref
                  , yHI_ref
                  , yHM_ref
                  , nH_ref
                  , k8_ref
                  , tform  ))
            if iTest == 1:
                # calculate the dumping timscale of spikes
                gamma = 5.0/3.0
                mu    = 1.23 * mh.d
                Tg_ref = 300
                cs = np.sqrt(gamma * kboltz.d * Tg_ref / mu)
                length = 3e-3 * 3.0856e18
                t_cross = length / cs / 3.1557e13
                print("%13.5e" % t_cross)
            if iTest == 2:
                # calculate the Jeans length
                nH_ref = 1e6
                Tg_ref = 600
                print("%13.5e %13.7f %13.5e" % (nH_ref, Tg_ref, Jeans_length(nH_ref, Tg_ref) / 3.0856e18))

            fp_Dfront.write("%13s "   % Test_name[iTest])
            fp_Dfront.write("%13.7f " % x_nHmax[iTest]  )
            fp_Dfront.write("%13.5e " % nH_nHmax        )
            fp_Dfront.write("%13.5e " % Tg_nHmax        )
            fp_Dfront.write("%13.5e " % yH2I_nHmax      )
            fp_Dfront.write("%13.5e " % NH2I_nHmax      )
           #fp_Dfront.write("%13.5e " % NH2I0_nHmax     )
           #fp_Dfront.write("%13.5e " % NH2I1_nHmax     )
           #fp_Dfront.write("%13.5e " % NH2I3_nHmax     )
            fp_Dfront.write("%13.5e " % f_shield        )
            fp_Dfront.write("%13.5e " % kH2I_nHmax      )
            fp_Dfront.write("%13.7f " % dx_Dfront       )
            fp_Dfront.write("\n");

#       print("yH2I max")
        for iTest in range(3):
#       for iTest in [1]:
            i_yH2Imax = np.argmax(data_prof[iTest][:, i_yH2I][data_prof[iTest][:, i_rad] < x_nHmax[iTest]])
            x_yH2Imax       = data_prof[iTest][i_yH2Imax, i_rad ]
            nH_yH2Imax      = data_prof[iTest][i_yH2Imax, i_nH  ]
            Tg_yH2Imax      = data_prof[iTest][i_yH2Imax, i_Tg  ]
            yH2I_yH2Imax    = data_prof[iTest][i_yH2Imax, i_yH2I]
            if iTest == 0:
                NH2I_yH2Imax = data_prof[iTest][i_yH2Imax, i_NH2I ]
            if iTest == 1:
                NH2I_yH2Imax = data_prof[iTest][i_yH2Imax, i_NH2I1]
            if iTest == 2:
                NH2I_yH2Imax = data_prof[iTest][i_yH2Imax, i_NH2I3]

            # shell width
            if iTest < 2: factor = 10.0
            else:         factor =  3.0
            i_H2ring, = np.where( (data_prof[iTest][:, i_rad ] < 0.3)
                                & (data_prof[iTest][:, i_yH2I] > yH2I_yH2Imax / factor))
            if False:
                x_H2ring_min[iTest] = np.min(data_prof[iTest][i_H2ring, i_rad])
                x_H2ring_max[iTest] = np.max(data_prof[iTest][i_H2ring, i_rad])
            else:
                if np.size(i_H2ring):
                    i_H2ring_min = np.min(i_H2ring)
                    i_H2ring_max = np.max(i_H2ring)
                    x_H2ring_min[iTest] = 10.0**np.interp(np.log10(0.1 * yH2I_yH2Imax)
                        , np.log10(data_prof[iTest][i_H2ring_min-1:i_H2ring_min+1, i_yH2I])
                        , np.log10(data_prof[iTest][i_H2ring_min-1:i_H2ring_min+1, i_rad ]))
                    x_H2ring_max[iTest] = 10.0**np.interp(np.log10(0.1 * yH2I_yH2Imax)
                        , np.log10(data_prof[iTest][i_H2ring_max+1:i_H2ring_max-1:-1, i_yH2I])
                        , np.log10(data_prof[iTest][i_H2ring_max+1:i_H2ring_max-1:-1, i_rad ]))
                else:
                    x_H2ring_min[iTest] = 0
                    x_H2ring_max[iTest] = 0
            dx_H2ring = x_H2ring_max[iTest] - x_H2ring_min[iTest]
     
            fp_H2ring.write("%13s "   % Test_name[iTest])
            fp_H2ring.write("%13.7f " % x_yH2Imax       )
            fp_H2ring.write("%13.5e " % nH_yH2Imax      )
            fp_H2ring.write("%13.5e " % Tg_yH2Imax      )
            fp_H2ring.write("%13.5e " % yH2I_yH2Imax    )
            fp_H2ring.write("%13.5e " % NH2I_yH2Imax    )
            fp_H2ring.write("%13.7f " % dx_H2ring       )
            fp_H2ring.write("\n");

        fp_Dfront.close()
        fp_H2ring.close()

        # fiducial values
#       print("NH2 %13.5e" % (1.0e-7*1.0e5 * 0.01*3.0856e18))

        # FIGURE
        ncols = 2; xfig = 12.0
#       nrows = 2; yfig =  9.0
        nrows = 3; yfig = 12.0
        if str(Threshold) == '1stP3':
            label_mass = r'$M_{\rm PopIII,1} = 10.4 {\rm M}_{\bigodot}$'
            label_nHth = r'$n_{\rm H, th} = 10^6 \ {\rm cm}^{-3}$'
            xmin = -2.0; xmax = 2.0
#           xmin = -1.2; xmax = -0.5
        if str(Threshold) == '1stP3h':
            label_mass = r'$M_{\rm PopIII,1} = 10.4 {\rm M}_{\bigodot}$'
            label_nHth = r'$n_{\rm H, th} = 10^8 \ {\rm cm}^{-3}$'
            xmin = -3.0; xmax = 1.0
        if str(Threshold) == '1stP3m':
            label_mass = r'$M_{\rm PopIII,1} = 40.0 {\rm M}_{\bigodot}$'
            label_nHth = r'$n_{\rm H, th} = 10^6 \ {\rm cm}^{-3}$'
#########   xmin = -2.0; xmax = 2.0
            xmin = -1.0; xmax = 0.0
        labels = [['(a)', '(b)'], ['(c)', '(d)'], ['(e)', '(f)']]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(xfig,yfig), constrained_layout=True)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.94, wspace=0.4, hspace=0.0)
        fig.align_labels()
        label_time = r"$t_{\rm SF} = %0.3f$ kyr" % t_SF
        fig.suptitle("%s %s %s" % (label_mass, label_nHth, label_time), fontsize=fontsize_suptitle, y=0.99)
        for irow in range(nrows):
            for icol in range(ncols):
                ax = axs[irow][icol]
                ax.tick_params(labelsize=fontsize_tick)
                ax.annotate(labels[irow][icol], xy=(0.03, 0.97), xycoords=ax.transAxes, fontsize=fontsize_title, va='top' , ha='left')
 
                ymin = np.zeros(3)
                ymax = np.zeros(3)

                for iTest in range(3):
#               for iTest in [1]:
                       
                    xx = np.log10(data_prof[iTest][:, i_rad])
               
                    if icol == 0:
                        if irow == 0:
                            yy = np.log10(data_prof[iTest][:, i_nH])
                            pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2, label=Test_name[iTest])
                            ax.legend(fontsize=fontsize_legend, loc='upper right')
                        if irow == 1:
                            yy = np.log10(data_prof[iTest][:, i_yelec])
                            pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2)
                       #    y2 = np.log10(data_prof[iTest][:, i_yHM])
                       #    pl = ax.plot(xx, y2, zorder=2, c=colors[iTest], linewidth=2, linestyle='--')
                       #    ax.annotate(r'e$^-$', xy=(0.6, 0.65), xycoords=ax.transAxes
                       #          , color='black', size=fontsize_legend
                       #          , va='center', ha='center'
                       #            )
                       #    ax.annotate(r'H$^-$', xy=(0.6, 0.25), xycoords=ax.transAxes
                       #          , color='black', size=fontsize_legend
                       #          , va='center', ha='center'
                       #            )
                        if irow == 2:
                            if iTest == 0:
                                yy = np.log10(data_prof[iTest][:, i_NH2I])
                                pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2, label=r'$\int n({\rm H_2}) {\rm d}s$')
                            if iTest == 1:
                                yy = np.log10(data_prof[iTest][:, i_NH2I1])
                                pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2, label=r'$n({\rm H_2}) \left| \rho / \nabla \rho \right|$')
#                               yy = np.log10(data_prof[iTest][:, i_NH2I ])
#                               pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2, linestyle='--')
                            if iTest == 2:
                                yy = np.log10(data_prof[iTest][:, i_NH2I3])
                                pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2, label=r'$n({\rm H_2}) \lambda _{\rm J}$')
#                               yy = np.log10(data_prof[iTest][:, i_NH2I ])
#                               pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2, linestyle='--')
                            ax.legend(fontsize=fontsize_legend, loc='lower right')
                    if icol == 1:
                        if irow == 0:
                            yy = np.log10(data_prof[iTest][:, i_Tg])
                            pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2)
                        if irow == 1:
                            yy = np.log10(data_prof[iTest][:, i_yH2I])
                            pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2)
                        if irow == 2:
                            if iTest == 0:
                                yy = np.log10(data_prof[iTest][:, i_kH2I])
                                pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2)
                            if iTest == 1:
                                yy = np.log10(data_prof[iTest][:, i_kH2I1])
                                pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2)
                            if iTest == 2:
                                yy = np.log10(data_prof[iTest][:, i_kH2I3])
                                pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2)
#                       if irow == 0:
#                           if iTest == 0:
#                               yy = np.log10(data_prof[iTest][:, i_sH2I])
#                               pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2, label=r'$N({\rm H_2}) / n({\rm H_2})$')
#                           if iTest == 1:
#                               yy = np.log10(data_prof[iTest][:, i_sH2I1])
#                               pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2, label=r'$\left| \rho / \nabla \rho \right|$')
#                           if iTest == 2:
#                               yy = np.log10(data_prof[iTest][:, i_sH2I3])
#                               pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2, label=r'$\lambda _{\rm J}$')
#                               ax.legend(fontsize=fontsize_legend)

#                   pl = ax.plot(xx, yy, zorder=2, c=colors[iTest], linewidth=2)

                    ymin[iTest] = np.min(yy[(xx > xmin) & (xx < xmax) & (yy != -np.inf)])
                    ymax[iTest] = np.max(yy[(xx > xmin) & (xx < xmax) & (yy != -np.inf)])
                  # if icol == 0 and irow == 1:
                  #     y2min = np.min(y2[(xx > xmin) & (xx < xmax) & (y2 != -np.inf)])
                  #     y2max = np.max(y2[(xx > xmin) & (xx < xmax) & (y2 != -np.inf)])
                  #     ymin[iTest] = np.min([ymin[iTest], y2min])
                  #     ymax[iTest] = np.max([ymax[iTest], y2max])

                yminmin = np.min(ymin)
                ymaxmax = np.max(ymax)
                y0 = (yminmin + ymaxmax) / 2.0
                dy =  ymaxmax - yminmin
                yminmin = y0 - 0.55* dy
                ymaxmax = y0 + 0.7 * dy
                if str(Threshold) == '1stP3' or str(Threshold) == '1stP3m':
                    if str(Threshold) == '1stP3' : iTest = 0
                    if str(Threshold) == '1stP3m': iTest = 1
                    ax.fill_between([np.log10(x_H2ring_min[iTest]), np.log10(x_H2ring_max[iTest])]
                                   , [yminmin, yminmin], [ymaxmax, ymaxmax], facecolor='none', edgecolor='#e1c3e1', hatch='/' *3)
                    ax.fill_between([np.log10(x_Dfront_min[iTest]), np.log10(x_Dfront_max[iTest])]
                                   , [yminmin, yminmin], [ymaxmax, ymaxmax], facecolor='none', edgecolor='#ffe5b8', hatch='\\'*3)
                    if icol == 1 and irow == 0:
                        xxx = np.log10(x_H2ring_max[iTest])
                        yyy = yminmin + 0.02 * (ymaxmax - yminmin)
                        ax.annotate(r'H$_2$-ring', xy=(xxx, yyy), xycoords='data'
                              , color='purple', size=fontsize_legend, rotation=90
                              , va='bottom', ha='right'
                                )
#                       xxx = (np.log10(x_Dfront_min[iTest]) + np.log10(x_Dfront_max[iTest])) / 2
                        xxx = np.log10(x_Dfront_max[iTest])
                        yyy = yminmin + 0.98 * (ymaxmax - yminmin)
                        ax.annotate(r'D-type front', xy=(xxx, yyy), xycoords='data'
                              , color='orange', size=fontsize_legend, rotation=90
                              , va='top', ha='right'
                                )
                if str(Threshold) == '1stP3h':
                    for iTest in range(3):
                        pl = ax.plot([np.log10(x_nHmax[iTest]), np.log10(x_nHmax[iTest])]
                                   , [yminmin, ymaxmax], zorder=1, c=colors[iTest], linestyle=':', linewidth=1.5)

                if irow < nrows - 1:
                    ax.set_xlabel('')
                    ax.tick_params(labelbottom = False)
                else:
                    ax.set_xlabel(r"log [ Distance / pc ]"                      , fontsize=fontsize_label)
                    ax.tick_params(labelbottom = True)
                if icol == 0:
                    if irow == 0:
                        ax.set_ylabel(r"log [ $n_{\rm H}$ / cm$^{-3}$ ]"        , fontsize=fontsize_label)
                    if irow == 1:
#                       ax.set_ylabel(r"log [ $y({\rm H}^{+})$ ]"               , fontsize=fontsize_label)
                        ax.set_ylabel(r"log [ $y({\rm e}^{-})$ ]"               , fontsize=fontsize_label)
                  #     ax.set_ylabel(r"log [ $y({\rm e}^{-}, {\rm H}^{-})$ ]"  , fontsize=fontsize_label)
                    if irow == 2:
                        ax.set_ylabel(r"log [ $N_{{\rm H}_2}$ / cm$^{-2}$ ]"    , fontsize=fontsize_label)
                if icol == 1:
                    if irow == 0:
                        ax.set_ylabel(r"log [ $T$ / K ]"                        , fontsize=fontsize_label)
                    if irow == 1:
                        ax.set_ylabel(r"log [ $y({\rm H}_{2})$ ]"               , fontsize=fontsize_label)
                    if irow == 2:
                        ax.set_ylabel(r"log [ $k _{\rm diss}$ / s$^{-1}$ ]"     , fontsize=fontsize_label)
#                   if irow == 1:
#                       ax.set_ylabel(r"log [ $\lambda _{\rm sh} ({\rm H}_2)$ ]", fontsize=fontsize_label)
        

                if str(Threshold) == '1stP3m':
                    ax.set_xticks(np.linspace(-1, 0, 3))
#                   ax.set_xticks(np.linspace(-3, 2, 6))
                    ax.set_xticklabels([r"$-1$", r"$-0.5$", r"$0$"])
#                   if inumber >= 31: ax.set_xticks(np.linspace(-3, 2, 51))
                else:
                    ax.set_xticks(np.linspace(-3, 2, 6))
                if (icol == 0 and irow == 0) or (icol == 0 and irow == 1):
                    ax.set_yticks(np.linspace(-8, 8,  9))
                    ax.set_yticks(np.linspace(-8, 8, 17), minor=True)
                if (icol == 0 and irow == 2) or (icol == 1 and irow == 1):
                    ax.set_yticks(np.linspace(-25, 25, 11))
                    ax.set_yticks(np.linspace(-25, 25, 51), minor=True)
                if (icol == 1 and irow == 2):
                    ax.set_yticks(np.linspace(-20, 0, 11))
                    ax.set_yticks(np.linspace(-20, 0, 21), minor=True)
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([yminmin, ymaxmax])

        # save figure
        fig.savefig(indir + "/rad_%s_%04d.%s" % (Threshold, inumber, extension))
        plt.close('all')


if RAY_C:
        # column density with the same nH, T, yH2 for TestC
        ncols = 1; xfig =  5.0
        nrows = 1; yfig =  4.1
        if str(Threshold) == '1stP3' or str(Threshold) == '1stP3m':
            label_nHth = r'$n_{\rm H, th} = 10^6 \ {\rm cm}^{-3}$'
            xmin = -2.0; xmax = 2.0
        if str(Threshold) == '1stP3h':
            label_nHth = r'$n_{\rm H, th} = 10^8 \ {\rm cm}^{-3}$'
            xmin = -3.0; xmax = 1.0
        labels = [['(a)', '(b)'], ['(c)', '(d)'], ['(e)', '(f)']]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(xfig,yfig), constrained_layout=True)
        fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.83, wspace=0.4, hspace=0.0)
        fig.align_labels()
        label_time = r"$t_{\rm SF} = %0.3f$ kyr" % t_SF
#       fig.suptitle("%s %s" % (label_nHth, label_time), fontsize=fontsize_suptitle_s, y=0.99)
        fig.suptitle("%s\n%s %s" % (label_mass, label_nHth, label_time), fontsize=fontsize_suptitle_s, y=0.992)
        for irow in range(nrows):
            for icol in range(ncols):
                ax = axs # [irow][icol]
                ax.tick_params(labelsize=fontsize_tick_s)
                ax.annotate(Test_name[2], xy=(0.03, 0.97), xycoords=ax.transAxes, fontsize=fontsize_title_s, va='top' , ha='left')
 
                ymin = np.zeros(3)
                ymax = np.zeros(3)

                for iTest in [2]:
                       
                    xx = np.log10(data_prof[iTest][:, i_rad])
               
                    if True:
                        if True:
                            if iTest == 2:
#                               nH2I = data_prof[iTest][:, i_yH2I] * data_prof[iTest][:, i_nH]
                                yy = np.log10(data_prof[iTest][:, i_NH2I ])
                                pl = ax.plot(xx, yy, zorder=2, c=colors[0], linewidth=1.5, label=r'$\int n({\rm H_2}) {\rm d}s$')

                                yy = np.log10(data_prof[iTest][:, i_NH2I1])
                                pl = ax.plot(xx, yy, zorder=2, c=colors[1], linewidth=1.5, label=r'$n({\rm H_2}) \left| \rho / \nabla \rho \right|$')

                                yy = np.log10(data_prof[iTest][:, i_NH2I3])
                                pl = ax.plot(xx, yy, zorder=2, c=colors[2], linewidth=1.5, label=r'$n({\rm H_2}) \lambda _{\rm J}$')

                                ax.legend(fontsize=fontsize_legend_s)

                    ymin[iTest] = np.min(yy[(xx > xmin) & (xx < xmax) & (yy != -np.inf)])
                    ymax[iTest] = np.max(yy[(xx > xmin) & (xx < xmax) & (yy != -np.inf)])
                    if icol == 0 and irow == 1:
                        y2min = np.min(y2[(xx > xmin) & (xx < xmax) & (y2 != -np.inf)])
                        y2max = np.max(y2[(xx > xmin) & (xx < xmax) & (y2 != -np.inf)])
                        ymin[iTest] = np.min([ymin[iTest], y2min])
                        ymax[iTest] = np.max([ymax[iTest], y2max])

                yminmin = np.min(ymin)
                ymaxmax = np.max(ymax)
                y0 = (yminmin + ymaxmax) / 2.0
                dy =  ymaxmax - yminmin
                yminmin = y0 - 0.5 * dy
                ymaxmax = y0 + 0.7 * dy
                if str(Threshold) == '1stP3' or str(Threshold) == '1stP3m':
                    if str(Threshold) == '1stP3' : iTest = 0
                    if str(Threshold) == '1stP3m': iTest = 1
                    ax.fill_between([np.log10(x_H2ring_min[iTest]), np.log10(x_H2ring_max[iTest])]
                                   , [yminmin, yminmin], [ymaxmax, ymaxmax], facecolor='none', edgecolor='#e1c3e1', hatch='/' *3)
                    ax.fill_between([np.log10(x_Dfront_min[iTest]), np.log10(x_Dfront_max[iTest])]
                                   , [yminmin, yminmin], [ymaxmax, ymaxmax], facecolor='none', edgecolor='#ffe5b8', hatch='\\'*3)
                    if True:
                        xxx = np.log10(x_H2ring_max[iTest])
                        yyy = yminmin + 0.02 * (ymaxmax - yminmin)
                        ax.annotate(r'H$_2$-ring', xy=(xxx, yyy), xycoords='data'
                              , color='purple', size=fontsize_legend_s, rotation=90
                              , va='bottom', ha='right'
                                )
                        xxx = (np.log10(x_Dfront_min[iTest]) + np.log10(x_Dfront_max[iTest])) / 2
                        yyy = yminmin + 0.02 * (ymaxmax - yminmin)
                        ax.annotate(r'D-type front', xy=(xxx, yyy), xycoords='data'
                              , color='orange', size=fontsize_legend_s, rotation=90
                              , va='bottom', ha='center'
                                )
                if str(Threshold) == '1stP3h':
                    if True:
                        pl = ax.plot([np.log10(x_nHmax[iTest]), np.log10(x_nHmax[iTest])]
                                   , [yminmin, ymaxmax], zorder=1, c=colors[iTest], linestyle=':', linewidth=1.5)

                if irow < nrows - 1:
                    ax.set_xlabel('')
                    ax.tick_params(labelbottom = False)
                else:
                    ax.set_xlabel(r"log [ Distance / pc ]", fontsize=fontsize_label_s)
                    ax.tick_params(labelbottom = True)
                if True:
                    if True:
                        ax.set_ylabel(r"log [ $N_{{\rm H}_2}$ / cm$^{-2}$ ]", fontsize=fontsize_label_s)

                ax.set_xticks(np.linspace(-3, 2, 6))
#               ax.set_xticks(np.linspace(-2, 2, 9), minor=True)
                if True:
                    ax.set_yticks(np.linspace(-25, 25, 11))
                    ax.set_yticks(np.linspace(-25, 25, 51), minor=True)
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([yminmin, ymaxmax])

        # save figure
        fig.savefig(indir + "/rad%s_%s_%04d.%s" % (Test, Threshold, inumber, extension))
        plt.close('all')

if SLICE:
#   TopGridDimension = 512
#   NestedLevel      =   2
    incs = range(inumber0, inumber1+1, inumber2)

#   TopGridDimension =  64
#   NestedLevel      =   2
#   incs = [34]

    xtime = ''

    fields = [
        'Hydrogen_number_density'
      , 'temperature_corr'
      , 'y_H2I'
             ]
    labels = ['(a)', '(b)', '(c)']
    nvar = len(fields)
    
    ibox = '0'
#   s_box = 0.2
#   boxsize = r'0.2 $h^{-1} (1+z)^{-1}$ Mpc'
    if str(Threshold) == '1stP3' or str(Threshold) == '1stP3m':
        s_box = 2.0
        boxsize = r'2 pc'
    if str(Threshold) == '1stP3h':
        s_box = 0.2
        boxsize = r'0.2 pc'
    reso = 800
    nnc = len(incs)
    arrow_length = 0.4
    
    Y, X = np.mgrid[0:reso+1, 0:reso+1]
    X = X / float(reso) * s_box
    Y = Y / float(reso) * s_box
    X = X - 0.5 * s_box
    Y = Y - 0.5 * s_box
    #print(X)

    for iTest in range(3):
#   for iTest in [1]:

        for iinc in range(nnc):
            inc = incs[iinc]

            xfig = 18.0
            yfig = 6.0
            fig, axs = plt.subplots(nrows=1, ncols=len(fields), figsize=(xfig,yfig/0.9), constrained_layout=True)
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.9 , wspace=0.0, hspace=0.0)

            if TopGridDimension == 64:
                outdir = 'test%d-L%d' % (TopGridDimension, NestedLevel)
            if TopGridDimension == 512:
                outdir = 'test%d-L%d_%s_%s/' % (TopGridDimension, NestedLevel, Threshold, Test_name[iTest])

            # stars
            starfile = outdir + ('Stars_PopIII_%04d.dat' % (inc))
#           print(starfile)
            starfp = open(starfile, 'rb')
            nPopIII = struct.unpack('i', starfp.read(4))[0]
            xPopIII = np.zeros(nPopIII)
            yPopIII = np.zeros(nPopIII)
            zPopIII = np.zeros(nPopIII)
            cPopIII = np.zeros(nPopIII)
            tPopIII = np.zeros(nPopIII)
            dPopIII = np.zeros(nPopIII)
            MPopIII = np.zeros(nPopIII)
            for iPopIII in range(nPopIII):
                xPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                yPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                zPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                cPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                dPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                tPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                MPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
#               print(xPopIII[iPopIII], yPopIII[iPopIII], tPopIII[iPopIII], MPopIII[iPopIII])
            starfp.close()
            iPopIII0 = np.argmin(cPopIII)


            for ivar in range(nvar):
                ax = axs[ivar]

                # slice data
                infile = outdir + ('Slice_z_%s_%04d.dat' % (fields[ivar], inc))

                infp = open(infile, 'rb')
                outdata = np.fromfile(infp, dtype='d',sep='')
#               if str(fields[ivar]) == 'temperature_corr':
#                   Z =         (outdata).reshape(reso,reso)
#               else:
#                   Z = np.log10(outdata).reshape(reso,reso)
                Z = np.log10(outdata).reshape(reso,reso)
                infp.close()

                if str(Threshold) == '1stP3':
                    label_mass = r'$M_{\rm PopIII,1} = 10.4 {\rm M}_{\bigodot}$'
                    label_nHth = r'$n_{\rm H, th} = 10^6 \ {\rm cm}^{-3}$'
                    xmin = -2.0; xmax = 2.0
#                   xmin = -1.2; xmax = -0.5
                if str(Threshold) == '1stP3h':
                    label_mass = r'$M_{\rm PopIII,1} = 10.4 {\rm M}_{\bigodot}$'
                    label_nHth = r'$n_{\rm H, th} = 10^8 \ {\rm cm}^{-3}$'
                    xmin = -3.0; xmax = 1.0
                if str(Threshold) == '1stP3m':
                    label_mass = r'$M_{\rm PopIII,1} = 40.0 {\rm M}_{\bigodot}$'
                    label_nHth = r'$n_{\rm H, th} = 10^6 \ {\rm cm}^{-3}$'
              # if str(Threshold) == '1stP3' or str(Threshold) == '1stP3m':
              #     label_nHth = r'$n_{\rm H, th} = 10^6 \ {\rm cm}^{-3}$'
              # if str(Threshold) == '1stP3h':
              #     label_nHth = r'$n_{\rm H, th} = 10^8 \ {\rm cm}^{-3}$'
                label_time = r'%s $t_{\rm SF} = %.3f$ kyr' % (Test_name[iTest], tPopIII[iPopIII0])
#               fig.suptitle(label_time, fontsize=fontsize_suptitle, y=0.98)
                fig.suptitle("%s %s %s" % (label_mass, label_nHth, label_time), fontsize=fontsize_suptitle, y=0.99)
              ##if ivar == 0:
              ##    ax.set_title("%s" % Test_text[iTest], fontsize=fontsize_suptitle, fontweight='bold', loc='left')
#             ##fig.align_labels()
        
                if str(fields[ivar]) == 'Hydrogen_number_density':
                    cmap = 'bds_highcontrast'
                if str(fields[ivar]) == 'temperature_corr':
                    cmap = 'hot'
                if str(fields[ivar]) == 'y_H2I':
                    cmap = 'BLUE'
               
                # slice data
                pcolor = ax.pcolormesh(X, Y, Z, cmap=cmap)

                # stars
#               pscatt = ax.scatter(xPopIII - xPopIII[iPopIII0], yPopIII - yPopIII[iPopIII0]
#                 , s=300.0, zorder=500, c='cyan', edgecolor='black', marker='*')
                pscatt = ax.scatter(xPopIII - xPopIII[iPopIII0], yPopIII - yPopIII[iPopIII0]
                  , s=10.0*MPopIII, zorder=500, c='cyan', marker='o')
##              if ivar == 0:
##                  if nSN == 1:
##                      xtime = r'TestC $t_{\rm SF} = %.3f$ kyr' % tPopIII[iPopIII0]
##                  if nSN == 3:
##                      xtime = r'$t_{\rm SF} = %.3f$ kyr' % tPopIII[iPopIII0]
###                 ax.annotate((0.05, 0.05), 't = %.3f kyr' % tPopIII[iPopIII0], coord_system='axis', color='white')
##                  ax.annotate(xtime, xy=(0.02, 0.98), xycoords=ax.transAxes
##                        , color='white', size=fontsize_title
##                        , va='top', ha='left'
##                          )
              # if nSN == 1:
              #     ax.annotate(labels[ivar], xy=(0.02, 0.98), xycoords=ax.transAxes
              #           , color='white', size=fontsize_title
              #           , va='top', ha='left'
              #             )
            
                ax.set_aspect('equal', adjustable='box')
                ax.tick_params(labelbottom=False,
                               labelleft=False,
                               labelright=False,
                               labeltop=False)
                ax.tick_params(bottom=False,
                               left=False,
                               right=False,
                               top=False)
#               if ivar==0:
#                   ax.annotate(xtime, xy=(0.03, 0.97), xycoords=ax.transAxes
#                             , color='white', fontsize=fontsize_title
#                             , va='top', ha='left'
#                               )
                if str(fields[ivar]) == 'Hydrogen_number_density':
                   xccolor='white'
                if str(fields[ivar]) == 'temperature_corr':
                   xccolor='white'
                if str(fields[ivar]) == 'y_H2I':
                   xccolor='white'
                if ivar==nvar-1:
                    ax.annotate(boxsize, xy=(0.93, 0.5), xycoords=ax.transAxes
                              , color=xccolor, fontsize=fontsize_boxsize, rotation=90
                              , va='center', ha='center'
                #             , bbox={'facecolor':'black', 'alpha':0.5, 'pad':2}
                                  )
                    ax.annotate('', xy=(0.93, 0.0), xycoords=ax.transAxes
                                  , xytext=(0.93, arrow_length), textcoords=ax.transAxes
                                  , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                                   headlength=10, connectionstyle='arc3',
                                                   facecolor=xccolor, edgecolor=xccolor)
                                  )
                    ax.annotate('', xy=(0.93, 1.0), xycoords=ax.transAxes
                                  , xytext=(0.93, 1.0-arrow_length), textcoords=ax.transAxes
                                  , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                                   headlength=10, connectionstyle='arc3',
                                                   facecolor=xccolor, edgecolor=xccolor)
                                  )
                cmin = Z.min(); cmax = Z.max(); 
                if str(fields[ivar]) == 'Hydrogen_number_density':
                    pcolor.set_clim(cmin, cmax)
                if str(fields[ivar]) == 'temperature_corr':
                    pcolor.set_clim(cmin, cmax)
                if str(fields[ivar]) == 'y_H2I':
                    pcolor.set_clim(-4, cmax)
####                pcolor.set_clim(-10, 0)
                ticks = np.linspace(-10, 10, 21)
              ##if str(fields[ivar]) == 'Hydrogen_number_density':
              ##    if str(Threshold) == '1stP3' or str(Threshold) == '1stP3m':
              ##        pcolor.set_clim(3.0, 5.8)
              ##    if str(Threshold) == '1stP3h':
              ##        pcolor.set_clim(5.0, 7.8)
              ##    ticks = np.linspace(3, 5, 3)
              ##if str(fields[ivar]) == 'temperature_corr':
              ##    pcolor.set_clim(2.0, 4.5)
              ##    ticks = np.linspace( 2, 4, 3)
              ##if str(fields[ivar]) == 'y_H2I':
              ##    pcolor.set_clim(-15, 0)
              ###   ticks = np.linspace(-15, 0, 16)
              ##    ticks = np.linspace(-15, 0,  4)
                axins = inset_axes(ax,
                                   width="50%",  # width = 5% of parent_bbox width
                                   height="5%",  # height : 50%
                                   loc='lower left',
                                   bbox_to_anchor=(0.05, 0.15, 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                  )
                colbar = fig.colorbar(pcolor, orientation='horizontal', ax=ax, cax=axins
#                      ,aspect=10,pad=-0.22 ,shrink=0.50
                       ,ticks=ticks
                          )
                colbar.outline.set_edgecolor(xccolor)
                if str(fields[ivar]) == 'Hydrogen_number_density':
                    xclabel = r'log [ Density / cm$^{-3}$ ]'
                if str(fields[ivar]) == 'temperature_corr':
                    xclabel = r'Temperature [ K ]'
                if str(fields[ivar]) == 'y_H2I':
                    xclabel = r'$y({\rm H_2})$'
                colbar.set_label(xclabel, fontsize=fontsize_cblabel, color=xccolor)
                colbar.ax.tick_params(labelsize=fontsize_tick, color=xccolor, labelcolor=xccolor)
        
            fig.savefig(indir + "/snapshots_%s_%s_%04d.png" % (Threshold, Test_name[iTest], inc))
            print(inc)
            
            plt.close('all')
 

if PRESEN:
    incs = range(inumber0, inumber1+1, inumber2)

    xtime = ''

    fields = [
        'Hydrogen_number_density'
      , 'temperature_corr'
      , 'y_H2I'
             ]
    nvar = len(fields)
    
    ibox = '0'
#   s_box = 0.2
#   boxsize = r'0.2 $h^{-1} (1+z)^{-1}$ Mpc'
    if str(Threshold) == '1stP3' or str(Threshold) == '1stP3m':
        s_box = 2.0
        boxsize = r'2 pc'
    if str(Threshold) == '1stP3h':
        s_box = 0.2
        boxsize = r'0.2 pc'
    reso = 800
    nnc = len(incs)
    arrow_length = 0.4
    
    Y, X = np.mgrid[0:reso+1, 0:reso+1]
    X = X / float(reso) * s_box
    Y = Y / float(reso) * s_box
    X = X - 0.5 * s_box
    Y = Y - 0.5 * s_box
    #print(X)

    if TopGridDimension == 64:
        nSN = 1
    if TopGridDimension == 512:
        nSN = 3

    for iinc in range(nnc):
        inc = incs[iinc]

        if str(Test) == 'TestA':
            iSN = 0
        if str(Test) == 'TestB':
            iSN = 1
        if str(Test) == 'TestC':
            iSN = 2

        if TopGridDimension == 64:
            outdir = 'test%d-L%d' % (TopGridDimension, NestedLevel)
        if TopGridDimension == 512:
            outdir = 'test%d-L%d_%s_%s/' % (TopGridDimension, NestedLevel, Threshold, Test_name[iSN])

        # stars
        starfile = outdir + ('Stars_PopIII_%04d.dat' % (inc))
#       print(starfile)
        starfp = open(starfile, 'rb')
        nPopIII = struct.unpack('i', starfp.read(4))[0]
        xPopIII = np.zeros(nPopIII)
        yPopIII = np.zeros(nPopIII)
        zPopIII = np.zeros(nPopIII)
        cPopIII = np.zeros(nPopIII)
        tPopIII = np.zeros(nPopIII)
        dPopIII = np.zeros(nPopIII)
        MPopIII = np.zeros(nPopIII)
        for iPopIII in range(nPopIII):
            xPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            yPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            zPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            cPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            dPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            tPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            MPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
#           print(xPopIII[iPopIII], yPopIII[iPopIII], tPopIII[iPopIII])
        starfp.close()
        iPopIII0 = np.argmin(cPopIII)


#       for ivar in range(nvar):
        for ivar in range(1):
            xfig = 6.0
            yfig = 6.0
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(xfig,yfig), constrained_layout=True)
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

            ax = axs

            # slice data
            infile = outdir + ('Slice_z_%s_%04d.dat' % (fields[ivar], inc))

            infp = open(infile, 'rb')
            outdata = np.fromfile(infp, dtype='d',sep='')
#           if str(fields[ivar]) == 'temperature_corr':
#               Z =         (outdata).reshape(reso,reso)
#           else:
#               Z = np.log10(outdata).reshape(reso,reso)
            Z = np.log10(outdata).reshape(reso,reso)
            infp.close()

            if str(fields[ivar]) == 'Hydrogen_number_density':
                cmap = 'bds_highcontrast'
            if str(fields[ivar]) == 'temperature_corr':
                cmap = 'hot'
            if str(fields[ivar]) == 'y_H2I':
                cmap = 'BLUE'
           
            # slice data
            pcolor = ax.pcolormesh(X, Y, Z, cmap=cmap)

            # stars
#           pscatt = ax.scatter(xPopIII - xPopIII[iPopIII0], yPopIII - yPopIII[iPopIII0]
#             , s=300.0, zorder=500, c='cyan', edgecolor='black', marker='*')
            pscatt = ax.scatter(xPopIII - xPopIII[iPopIII0], yPopIII - yPopIII[iPopIII0]
              , s=10.0*MPopIII, zorder=500, c='cyan', marker='o')
            if ivar == 0:
                ax.annotate(r'%s' % Test_text[iSN], xy=(0.02, 0.98), xycoords=ax.transAxes
                      , color='white', size=fontsize_suptitle
                      , va='top', ha='left'
                        )
                ax.annotate(r'$t_{\rm SF} = %.3f$ kyr' % tPopIII[iPopIII0], xy=(0.02, 0.90), xycoords=ax.transAxes
                      , color='white', size=fontsize_title
                      , va='top', ha='left'
                        )
        
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelbottom=False,
                           labelleft=False,
                           labelright=False,
                           labeltop=False)
            ax.tick_params(bottom=False,
                           left=False,
                           right=False,
                           top=False)
            if ivar==0:
                ax.annotate(xtime, xy=(0.03, 0.97), xycoords=ax.transAxes
                          , color='white', fontsize=fontsize_title
                          , va='top', ha='left'
                            )
            if str(fields[ivar]) == 'Hydrogen_number_density':
               xccolor='white'
            if str(fields[ivar]) == 'temperature_corr':
               xccolor='white'
            if str(fields[ivar]) == 'y_H2I':
               xccolor='white'
            # scale
            ax.annotate(boxsize, xy=(0.93, 0.5), xycoords=ax.transAxes
                      , color=xccolor, fontsize=fontsize_suptitle, rotation=90
                      , va='center', ha='center'
#                     , bbox={'facecolor':'black', 'alpha':0.5, 'pad':2}
                          )
            ax.annotate('', xy=(0.93, 0.0), xycoords=ax.transAxes
                          , xytext=(0.93, arrow_length), textcoords=ax.transAxes
                          , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                           headlength=10, connectionstyle='arc3',
                                           facecolor=xccolor, edgecolor=xccolor)
                          )
            ax.annotate('', xy=(0.93, 1.0), xycoords=ax.transAxes
                          , xytext=(0.93, 1.0-arrow_length), textcoords=ax.transAxes
                          , arrowprops=dict(shrink=0, width=1, headwidth=8,
                                           headlength=10, connectionstyle='arc3',
                                           facecolor=xccolor, edgecolor=xccolor)
                          )
            cmin = Z.min(); cmax = Z.max(); 
            if str(fields[ivar]) == 'Hydrogen_number_density':
#               if cmin < -4.0: cmin = -4.0 
#               pcolor.set_clim(cmin, cmax)
                if str(Threshold) == '1stP3' or str(Threshold) == '1stP3m':
                    pcolor.set_clim(3.0, 5.8)
                if str(Threshold) == '1stP3h':
                    pcolor.set_clim(5.0, 7.8)
                ticks=np.linspace(3, 5, 3)
            if str(fields[ivar]) == 'temperature_corr':
#               if cmin < 0.5: cmin = 0.5 
#               pcolor.set_clim(cmin, cmax)
                pcolor.set_clim(2.0, 4.5)
                ticks=np.linspace( 2, 4, 3)
            if str(fields[ivar]) == 'y_H2I':
#               if cmin < 0.0: cmin = 0.0 
#               pcolor.set_clim(cmin, cmax)
                pcolor.set_clim(-15, 0)
                ticks=np.linspace(-15, 0, 16)
            axins = inset_axes(ax,
                               width="50%",  # width = 5% of parent_bbox width
                               height="5%",  # height : 50%
                               loc='lower left',
                               bbox_to_anchor=(0.05, 0.15, 1, 1),
                               bbox_transform=ax.transAxes,
                               borderpad=0,
                              )
            colbar = fig.colorbar(pcolor, orientation='horizontal', ax=ax, cax=axins
#                  ,aspect=10,pad=-0.22 ,shrink=0.50
#                  ,ticks=ticks
                      )
            colbar.outline.set_edgecolor(xccolor)
            if str(fields[ivar]) == 'Hydrogen_number_density':
                xclabel = r'log [ Density / cm$^{-3}$ ]'
            if str(fields[ivar]) == 'temperature_corr':
                xclabel = r'Temperature [ K ]'
            if str(fields[ivar]) == 'y_H2I':
                xclabel = r'$y({\rm H_2})$'
            colbar.set_label(xclabel, fontsize=fontsize_cblabel, color=xccolor)
            colbar.ax.tick_params(labelsize=fontsize_tick, color=xccolor, labelcolor=xccolor)
        
            fig.savefig(indir + "/snapshots_%s_%s_%04d_%s.png" % (Threshold, Test_name[iSN], inc, fields[ivar]))
            
            plt.close('all')



if PROF:
    i_rad   =  0
    i_nH    =  1
    i_Tg    =  2
    i_yelec =  3
    i_yHI   =  4
    i_yHII  =  5
    i_yHeI  =  6
    i_yHeII =  7
    i_yHeIII=  8
    i_yHM   =  9
    i_yH2I  = 10
    i_yH2II = 11
    i_yDI   = 12
    i_yDII  = 13
    i_yHDI  = 14
    i_yDM   = 15
    i_yHDII = 16
    i_yHeHII= 17
    i_kHI   = 18
    i_kHeI  = 19
    i_kHeII = 20
    i_kH2I  = 21
    i_kH2It = 22
    i_sH2I1 = 23
    i_kH2I1 = 24
    i_sH2I3 = 25
    i_kH2I3 = 26

    data_prof = np.empty(3, dtype=list)
    nPopIII = np.zeros(3, dtype=int)
    rPopIII = np.zeros(3, dtype=list)
    for iSN in range(3):
        outdir = 'test%d-L%d_%s_%s' % (TopGridDimension, NestedLevel, Threshold, Test_name[iSN])
        inumber = 48
        data_prof[iSN] = np.loadtxt('%s/prof_%04d.dat' % (outdir, inumber))

        fn_star = outdir + '/Stars_PopIII_%04d.dat' % (inumber)
        starfp = open(fn_star, 'rb')
        nPopIII[iSN] = struct.unpack('i', starfp.read(4))[0]
        xPopIII = np.zeros(nPopIII[iSN])
        yPopIII = np.zeros(nPopIII[iSN])
        zPopIII = np.zeros(nPopIII[iSN])
        cPopIII = np.zeros(nPopIII[iSN])
        tPopIII = np.zeros(nPopIII[iSN])
        dPopIII = np.zeros(nPopIII[iSN])
        MPopIII = np.zeros(nPopIII[iSN])
        for iPopIII in range(nPopIII[iSN]):
            xPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            yPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            zPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            cPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            dPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            tPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            MPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
        starfp.close()
        iPopIII0 = np.argmin(cPopIII)
        t_SF = tPopIII[iPopIII0]

        # distance of secondary from primary
        rPopIII[iSN] = np.zeros(nPopIII[iSN])
        for iPopIII in range(nPopIII[iSN]):
            dx = xPopIII[iPopIII] - xPopIII[iPopIII0]
            dy = yPopIII[iPopIII] - yPopIII[iPopIII0]
            dz = zPopIII[iPopIII] - zPopIII[iPopIII0]
            rPopIII[iSN][iPopIII] = np.sqrt(dx**2 + dy**2 + dz**2)

    if str(Threshold) == '1stP3':
        label_mass = r'$M_{\rm PopIII,1} = 10.4 {\rm M}_{\bigodot}$'
        label_nHth = r'$n_{\rm H, th} = 10^6 \ {\rm cm}^{-3}$'
        xmin = -2.0; xmax = 2.0
#       xmin = -1.2; xmax = -0.5
    if str(Threshold) == '1stP3h':
        label_mass = r'$M_{\rm PopIII,1} = 10.4 {\rm M}_{\bigodot}$'
        label_nHth = r'$n_{\rm H, th} = 10^8 \ {\rm cm}^{-3}$'
        xmin = -3.0; xmax = 1.0
   #if str(Threshold) == '1stP3m':
   #    label_mass = r'$M_{\rm PopIII,0} = 40.0 {\rm M}_{\bigodot}$'
   #    label_nHth = r'$n_{\rm H, th} = 10^6 \ {\rm cm}^{-3}$'
   #if str(Threshold) == '1stP3' or str(Threshold) == '1stP3m':
   #    label_nHth = r'$n_{\rm H, th} = 10^6 \ {\rm cm}^{-3}$'
   #    xmin = -2.0; xmax = 2.0
   #if str(Threshold) == '1stP3h':
   #    label_nHth = r'$n_{\rm H, th} = 10^8 \ {\rm cm}^{-3}$'
   #    xmin = -3.0; xmax = 1.0
    labels = [['(a)', '(b)'], ['(c)', '(d)'], ['(e)', '(f)']]
    label_time = r"$t_{\rm SF} = %0.3f$ kyr" % t_SF

    def J_LW(kdiss):
        TimeUnits = 3.996888808944e+14
        E_LW = 1.60184e-12 * 12.8 # erg
        sigma_H2 = 3.71e-18 # cm**2
        eV = 1.60184e-12
        hp = 6.6260693E-27
        Dnu = (13.6-11.2)*eV / hp
        print("%13.5e" % Dnu)
        return E_LW * (kdiss/TimeUnits) / sigma_H2 / Dnu / (4.0*np.pi)
#   print(data_prof[0][:,i_kH2I ], J_LW(data_prof[0][:,i_kH2I ]))

    xfig = 5
    yfig = 7.7
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(xfig,yfig), constrained_layout=True)
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.12, top=0.90, wspace=0.4, hspace=0.0)
    fig.suptitle("%s\n%s %s" % (label_mass, label_nHth, label_time), fontsize=fontsize_suptitle_s, y=0.98)
    # ION FRACTION
    ax = axs[0]
    ax.annotate('(a)', xy=(0.03, 0.97), xycoords=ax.transAxes, fontsize=fontsize_title_s, va='top' , ha='left')
    ax.tick_params(labelsize=fontsize_tick_s)
    ax.tick_params(labelsize=fontsize_tick_s)
    for iSN in range(3):
        xx = np.log10(data_prof[iSN][:,i_rad]*1000) # convert to comoving kpc
        yy1 = np.log10(data_prof[iSN][:,i_yHII  ])
        yy2 = np.log10(data_prof[iSN][:,i_yHeII ])
        yy3 = np.log10(data_prof[iSN][:,i_yHeIII])
        ax.plot(xx, yy1, color=colors[iSN], linestyle='-' , label=Test_name[iSN])
        ax.plot(xx, yy2, color=colors[iSN], linestyle='--')
        ax.plot(xx, yy3, color=colors[iSN], linestyle=':' )
##      ixIfront = np.argmin(np.abs(yy1 - (-2)))
##      xIfront = np.interp(-2, yy1[ixIfront-1: ixIfront+2], xx[ixIfront-1: ixIfront+2])
##      print("ifront", 10.0**(xx[ixIfront]))
        if nPopIII[iSN] > 1:
            x_int = np.log10(rPopIII[iSN]/1000*(1.0+25.062211586783)*0.6774) # convert to comoving kpc
            y_int = np.interp(x_int, xx, yy1)
            ax.scatter(x_int, y_int, color=colors[iSN])
    ax.annotate(r'H$^+$', xy=(0.32, 0.6), xycoords=ax.transAxes
          , color='black', size=fontsize_legend_s
          , va='center', ha='center'
            )
    ax.annotate(r'He$^+$', xy=(0.12, 0.28), xycoords=ax.transAxes
          , color='black', size=fontsize_legend_s
          , va='center', ha='center'
            )
    ax.legend(fontsize=fontsize_legend_s)
#   ax.set_xlabel(r"log [Distance / $h^{-1} (1+z)^{-1}$ kpc)]", fontsize=fontsize_label_s)
    ax.set_ylabel(r"log [ $y$(H$^+$, He$^+$)]", fontsize=fontsize_label_s)
    ax.set_xticks(np.linspace(-4, 4, 5))
    ax.set_xticks(np.linspace(-4, 4, 9), minor=True)
    ax.set_yticks(np.linspace(-6,  0, 4))
    ax.set_yticks(np.linspace(-6,  1, 8), minor=True)
    ax.set_xlim([-3.2, 3])
    ax.set_ylim([-6, 1.5])
    # LW INTENSITY
    ax = axs[1]
    ax.annotate('(b)', xy=(0.03, 0.97), xycoords=ax.transAxes, fontsize=fontsize_title_s, va='top' , ha='left')
    ax.tick_params(labelsize=fontsize_tick_s)
    ax.tick_params(labelsize=fontsize_tick_s)
    for iSN in range(3):
        xx = np.log10(data_prof[iSN][:,i_rad]*1000) # convert to comoving kpc
        if iSN == 0: yy = np.log10(J_LW(data_prof[0][:,i_kH2I ]) / 1.0e-21)
        if iSN == 1: yy = np.log10(J_LW(data_prof[1][:,i_kH2I1]) / 1.0e-21)
        if iSN == 2: yy = np.log10(J_LW(data_prof[2][:,i_kH2I3]) / 1.0e-21)
        ax.plot(xx, yy, color=colors[iSN], label=Test_name[iSN])
#       if iSN == 2:
#           ax.plot(xx, np.log10(J_LW(data_prof[2][:,i_kH2It])), color=colors[iSN], label=Test_name[iSN], linestyle='--')
##      ixquench = np.argmin(np.abs(yy - (-1)))
##      xquench = np.interp(-1, yy[ixquench-1: ixquench+2], xx[ixquench-1: ixquench+2])
##      print("quench", 10.0**(xx[ixquench]))
        yquench = np.interp(1, xx, yy)
        print(yquench)
        if nPopIII[iSN] > 1:
            x_int = np.log10(rPopIII[iSN]/1000*(1.0+25.062211586783)*0.6774) # convert to comoving kpc
            y_int = np.interp(x_int, xx, yy)
            ax.scatter(x_int, y_int, color=colors[iSN])
#   ax.legend(fontsize=fontsize_legend_s)
    ax.set_xlabel(r"log [Distance / $h^{-1} (1+z)^{-1}$ kpc)]", fontsize=fontsize_label_s)
    ax.set_ylabel(r"log [$J_{21}$ ]", fontsize=fontsize_label_s)
    # $10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ str$^{-1}$ 
    ax.set_xticks(np.linspace(-4, 4, 5))
    ax.set_xticks(np.linspace(-4, 4, 9), minor=True)
    ax.set_yticks(np.linspace(-5, 5, 3))
    ax.set_yticks(np.linspace(-6, 10, 17), minor=True)
    ax.set_xlim([-3.2, 3])
    ax.set_ylim([-6, 10])
    fig.savefig(indir + "/prof_%s_%04d.pdf" % (Threshold, inumber))
    plt.close('all')



if TABLE:
    # tex format
    def f2s(f):
        if f == 0.0:
            return '$0.0$'
        elif f<0.01 or f>=1000.0:
            ind = np.floor(np.log10(f))
            val = f / 10.0**ind
#           print(f, val, ind)
            return '$%.2f\\E{%d}$' % (val, ind)
        elif f<0.1:
            return '$%.4f$' % f
        elif f<1.0:
            return '$%.3f$' % f
        elif f<10.0:
            return '$%.2f$' % f
        elif f<100.0:
            return '$%.1f$' % f
        else:
            return '$%.0f$' % f

    nSN = 3
    inumber = 48

    # consistency check
    data_S02 = np.loadtxt('/home/gen/stars/tables/Schaerer02.dat')

    xfig = 6
    yfig = 9
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(xfig,yfig), constrained_layout=True)
#   fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    axs[0].tick_params(labelsize=fontsize_tick)
    axs[1].tick_params(labelsize=fontsize_tick)
    pl = axs[0].loglog(data_S02[:,0], data_S02[:,4], zorder=2, color='black', linewidth=2, label=r'$Q({\rm H})$   ')
    pl = axs[0].loglog(data_S02[:,0], data_S02[:,5], zorder=2, color='blue' , linewidth=2, label=r'$Q({\rm He})$  ')
    pl = axs[0].loglog(data_S02[:,0], data_S02[:,6], zorder=2, color='red'  , linewidth=2, label=r'$Q({\rm He}^+)$')
    pl = axs[0].loglog(data_S02[:,0], data_S02[:,7], zorder=2, color='green', linewidth=2, label=r'$Q({\rm H}_2)$ ')
    pl = axs[1].semilogx(data_S02[:,0], data_S02[:,8]/1.0e6, zorder=2, color='black', linewidth=2, label=r'$t_{\rm life}$ ')
    
    for Threshold in ['1stP3', '1stP3h', '1stP3m']:

        for iSN in range(3):

            outdir = 'test%d-L%d_%s_%s/' % (TopGridDimension, NestedLevel, Threshold, Test_name[iSN])
         
            # Pop III star data
            fn_star = outdir + 'Stars_PopIII_%04d.dat' % (inumber)
            starfp = open(fn_star, 'rb')
            nPopIII = struct.unpack('i', starfp.read(4))[0]
            xPopIII = np.zeros(nPopIII)
            yPopIII = np.zeros(nPopIII)
            zPopIII = np.zeros(nPopIII)
            cPopIII = np.zeros(nPopIII)
            tPopIII = np.zeros(nPopIII)
            dPopIII = np.zeros(nPopIII)
            MPopIII = np.zeros(nPopIII)
            for iPopIII in range(nPopIII):
                xPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                yPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                zPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                cPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                dPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                tPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
                MPopIII[iPopIII] = struct.unpack('d', starfp.read(8))[0]
            starfp.close()

            # primary Pop III star
            iPopIII0 = np.argmin(cPopIII)
#           t_SF0 = cPopIII[iPopIII0]

            # distance of secondary from primary
            rPopIII = np.zeros(nPopIII)
            for iPopIII in range(nPopIII):
                dx = xPopIII[iPopIII] - xPopIII[iPopIII0]
                dy = yPopIII[iPopIII] - yPopIII[iPopIII0]
                dz = zPopIII[iPopIII] - zPopIII[iPopIII0]
                rPopIII[iPopIII] = np.sqrt(dx**2 + dy**2 + dz**2)
           

            indices = np.argsort(cPopIII)
            Q_HI_tot = 0.0
            for iPopIII in indices:
                pscat0 = axs[0].scatter(MPopIII[iPopIII], Q_HI  (MPopIII[iPopIII]), s=100.0, zorder=500, c='black', marker='o')
                pscat1 = axs[0].scatter(MPopIII[iPopIII], Q_HeI (MPopIII[iPopIII]), s=100.0, zorder=500, c='blue' , marker='o')
                pscat2 = axs[0].scatter(MPopIII[iPopIII], Q_HeII(MPopIII[iPopIII]), s=100.0, zorder=500, c='red'  , marker='o')
                pscat3 = axs[0].scatter(MPopIII[iPopIII], Q_LW  (MPopIII[iPopIII]), s=100.0, zorder=500, c='green', marker='o')
                pscat4 = axs[1].scatter(MPopIII[iPopIII],       (dPopIII[iPopIII]), s=100.0, zorder=500, c='black', marker='o')
#               print(tPopIII[iPopIII0] , tPopIII[iPopIII])
                print("%6s & %6s & $%5.1f$ & $%5.1f$ & $%5.1f$ & %8s & %9s & %9s & %9s & %9s\\\\" % (
                       Threshold
                     , Test_name[iSN]
                     , MPopIII[iPopIII]
#                    , tPopIII[iPopIII0] - tPopIII[iPopIII] # cPopIII[iPopIII] - t_SF0 # 
                     ,(cPopIII[iPopIII0] - cPopIII[iPopIII])*1.0e3
                     , dPopIII[iPopIII]
                     , f2s(rPopIII[iPopIII])
                     , f2s(Q_HI  (MPopIII[iPopIII]))
                     , f2s(Q_HeI (MPopIII[iPopIII]))
                     , f2s(Q_HeII(MPopIII[iPopIII]))
                     , f2s(Q_LW  (MPopIII[iPopIII]))
                     ))
                Q_HI_tot += Q_HI  (MPopIII[iPopIII])
#           print("%13.5e" % Q_HI_tot)
        print("\\hline \\\\")


        axs[0].legend(fontsize=fontsize_legend)
        axs[0].set_ylabel(r"$Q$ [s$^{-1}$]"                   , fontsize=fontsize_label)
        axs[1].set_ylabel(r"$t_{\rm life}$ [Myr]"             , fontsize=fontsize_label)
        axs[1].set_xlabel(r"$M_{\rm PopIII}$ [M$_{\bigodot}$]", fontsize=fontsize_label)
#       ax.set_xlim([xmin, xmax])
#       ax.set_ylim([yminmin, ymaxmax])
        fig.savefig("mQ.%s" % (extension))
        plt.close('all')
