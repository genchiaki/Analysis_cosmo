import yt
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
#from yt.data_objects.particle_filters import add_particle_filter
#from yt.analysis_modules.halo_finding.api import HaloFinder
from yt.extensions.astro_analysis.halo_analysis.halo_catalog import HaloCatalog
from yt.data_objects.particle_filters import add_particle_filter
from yt.extensions.astro_analysis.halo_finding.rockstar.api import RockstarHaloFinder
'''
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
from yt.data_objects.particle_filters import add_particle_filter
from yt.extensions.astro_analysis.halo_analysis.halo_finding.rockstar.api import RockstarHaloFinder
'''
import numpy as np
from numpy import linalg as LA
import datetime
import sys
import struct
from yt.units import mp, kboltz, G
pi = 3.14159265

HydrogenFractionByMass   = 0.76
DeuteriumToHydrogenRatio = 3.4e-5 * 2.0
HeliumToHydrogenRatio    = (1.0 - HydrogenFractionByMass) / HydrogenFractionByMass
SolarMetalFractionByMass = 0.01295
SolarIronAbundance = 7.50

def DarkMatter(pfilter, data):
    return (((data[("all", "particle_type")] == 1) & (data['creation_time'] < 0)) | (data[("all", "particle_type")] == 4))
add_particle_filter("dark_matter", function=DarkMatter, filtered_type='all', requires=["particle_type", "creation_time"])

def PopIIIStars(pfilter, data):
    return ((data[("all", "particle_type")] == 5) & (data['particle_mass'].in_units('Msun') > 1e-3))
add_particle_filter("PopIII", function=PopIIIStars, filtered_type='all', requires=["particle_type", "particle_mass"])

def BlackHoles(pfilter, data):
    return (data[("all", "particle_type")] == 6)
add_particle_filter("BH", function=BlackHoles, filtered_type='all', requires=["particle_type"])

def PopIIStars(pfilter, data):
    return (data[("all", "particle_type")] == 7)
add_particle_filter("PopII", function=PopIIStars, filtered_type='all', requires=["particle_type"])

def f_shield_H2I_WH11(NH2I, T):
    x = NH2I / ds.arr(5e14, "1/cm**2")
    b_doppler = np.sqrt(2.0 * kboltz * T / mp) / ds.arr(1e5, "cm/s")
    f_shield = 0.965 / (1.0 + x/b_doppler)**1.1 + 0.035 * np.exp(-8.5e-4 * np.sqrt(1.0 + x))/np.sqrt(1.0 + x)
    return f_shield

TopGridDimension = int(sys.argv[1])
NestedLevel      = int(sys.argv[2])
Threshold        = sys.argv[3]
Test             = sys.argv[4]

def Q_LW(m):
    x = np.log10(m)
    x2 = x * x
    return 10.0**(44.03 + 4.59*x  - 0.77*x2)

HALO     = False
HALOFIND = False
SLICE    = True
PROJ     = False
PARTICLE = False
PHASE    = False
PROF     = True
RAY      = True
TRACE_MASS      = False
TRACE_ABUNDANCE = False

if NestedLevel < 2:
     MultiSpecies   = 0
     MetalChemistry = 0
     GrainGrowth    = 0
     DustSpecies    = 0
     CosmologySimulation  = True
     StarParticleCreation = False
     MultiMetals          = False
else:
     MultiSpecies   = 4
     MetalChemistry = 1
     GrainGrowth    = 1
     DustSpecies    = 2
     CosmologySimulation  = True
     StarParticleCreation = True
     MultiMetals          = True


# Test A (default)
if str(Test) == "TestA":
    RadiativeTransferOpticallyThinH2 = 0
    RadiativeTransferUseH2Shielding  = 1
    H2_self_shielding = 0

# Test B and C
if str(Test) == "TestB" or str(Test) == "TestC":
    RadiativeTransferOpticallyThinH2 = 1
    RadiativeTransferUseH2Shielding  = 0
    if str(Test) == "TestB":
        H2_self_shielding = 1
    if str(Test) == "TestC":
        H2_self_shielding = 3

#if TopGridDimension == 64:
#    RadiativeTransferOpticallyThinH2 = 1
#    RadiativeTransferUseH2Shielding  = 1
#    H2_self_shielding = 0


if TopGridDimension == 64:
    indir = '/scratch1/05562/tg848620/enzo-dev/run/CosmologySimulation/test%d/test%d-L%d/' % (TopGridDimension, TopGridDimension, NestedLevel)
if TopGridDimension == 512:
    indir ='./'
#indir = '/media/genchiaki/enzo2/scratch/enzo-dev/run/CosmologySimulation/test%d/test%d-L%d/' % (TopGridDimension, TopGridDimension, NestedLevel)
#indir = '/scratch1/05562/tg848620/enzo-dev/run/CosmologySimulation/test%d/test%d-L%d_1stP3_TestC/' % (TopGridDimension, TopGridDimension, NestedLevel)
#indir = '/mnt/raid/scratch/enzo-dev/run/CosmologySimulation/test%d/test%d-L%d_1stP3_%s/' % (TopGridDimension, TopGridDimension, NestedLevel, Test)

if TopGridDimension == 64:
    outdir = 'test%d-L%d/' % (TopGridDimension, NestedLevel)
if TopGridDimension == 512:
    outdir = '/scratch1/05562/tg848620/enzo-dev/run/CosmologySimulation/Analysis_cosmo/test%d-L%d_%s_%s/' % (TopGridDimension, NestedLevel, Threshold, Test)

inumber0 = int(sys.argv[5])
inumber1 = int(sys.argv[6])
inumber2 = int(sys.argv[7])
version  =  ''

#if HALO:
yt.enable_parallelism() # rockstar halofinding requires parallelism

if HALOFIND:
    # First, we make sure that this script is being run using mpirun with
    # at least 3 processors as indicated in the comments above.
    assert(yt.communication_system.communicators[-1].size >= 3)

if TRACE_MASS:
    fn_trace_mass = 'time_series_mass_%04d-%04d.dat' % (inumber0, inumber1)
    fp_trace_mass = open(fn_trace_mass, mode='w')
if TRACE_ABUNDANCE:
    fn_trace_ab = 'time_series_%04d-%04d.dat' % (inumber0, inumber1)
    fp_trace_ab = open(fn_trace_ab, mode='w')


for inumber in range(inumber0, inumber1+1, inumber2):
    number  = '%04d' % inumber

    fn = indir + "DD" + number + version + "/output_" + number # dataset to load
    ds = yt.load(fn) # load data
    if yt.is_root():
        print(fn)
        print(datetime.datetime.now(), flush=True)

    ad = ds.all_data()
    if yt.is_root():
        print(datetime.datetime.now(), flush=True)

    GravConst = ds.arr(6.67428e-8, "cm**3 * g**(-1) * s**(-2)")

    def _cell_size(field, data): return data["cell_volume"]**(1.0/3.0)
    ds.add_field(("gas", "cell_size"), function=_cell_size, sampling_type="cell", units="code_length")
    if MultiSpecies > 0:
        def _Hydrogen_number_density(field, data): return HydrogenFractionByMass * data["density"] /mp
        ds.add_field(("gas", "Hydrogen_number_density"), function=_Hydrogen_number_density, sampling_type="cell", units="cm**(-3)")
        def _Compressional_heating_rate(field, data): return data["pressure"] * data["velocity_divergence_absolute"] / data["density"]
        ds.add_field(("gas", "Compressional_heating_rate"), function=_Compressional_heating_rate, sampling_type="cell", units="erg/g/s")
    if MetalChemistry:
##      def _Zmet(field, data): return data["SN_Colour"] / data["Density"] / SolarMetalFractionByMass
##      ds.add_field(("gas", "Zmet"), function=_Zmet, sampling_type="cell", units="1")
        def _Zmet(field, data): return (data["SN_Colour"] + data["Metal_Density"]) / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet"), function=_Zmet, sampling_type="cell", units="1")
    if MultiMetals:
        def _Zmet0(field, data): return data["Z_Field0"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet0"), function=_Zmet0, sampling_type="cell", units="1")
        def _Zmet1(field, data): return data["Z_Field1"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet1"), function=_Zmet1, sampling_type="cell", units="1")
        def _Zmet2(field, data): return data["Z_Field2"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet2"), function=_Zmet2, sampling_type="cell", units="1")
        def _Zmet3(field, data): return data["Z_Field3"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet3"), function=_Zmet3, sampling_type="cell", units="1")
        def _Zmet4(field, data): return data["Z_Field4"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet4"), function=_Zmet4, sampling_type="cell", units="1")
        def _Zmet5(field, data): return data["Z_Field5"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet5"), function=_Zmet5, sampling_type="cell", units="1")
        def _Zmet6(field, data): return data["Z_Field6"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet6"), function=_Zmet6, sampling_type="cell", units="1")
        def _Zmet7(field, data): return data["Z_Field7"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet7"), function=_Zmet7, sampling_type="cell", units="1")
        def _Zmet8(field, data): return data["Z_Field8"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet8"), function=_Zmet8, sampling_type="cell", units="1")
        def _Zmet9(field, data): return data["Z_Field9"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet9"), function=_Zmet9, sampling_type="cell", units="1")
        def _Zmet10(field, data): return data["Z_Field10"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet10"), function=_Zmet10, sampling_type="cell", units="1")
        def _Zmet11(field, data): return data["Z_Field11"] / data["Density"] / SolarMetalFractionByMass
        ds.add_field(("gas", "Zmet11"), function=_Zmet11, sampling_type="cell", units="1")

    if MultiSpecies > 0:
        def _y_elec(field, data): return data["Electron_Density"] / data["Density"] /HydrogenFractionByMass/1.0
        ds.add_field(("gas", "y_elec"), function=_y_elec, sampling_type="cell", units="1")
        def _y_HI(field, data): return data["HI_Density"] / data["Density"] /HydrogenFractionByMass/1.0
        ds.add_field(("gas", "y_HI"), function=_y_HI, sampling_type="cell", units="1")
        def _y_HII(field, data): return data["HII_Density"] / data["Density"] /HydrogenFractionByMass/1.0 
        ds.add_field(("gas", "y_HII"), function=_y_HII, sampling_type="cell", units="1")
        def _y_HeI(field, data): return data["HeI_Density"] / data["Density"] /HydrogenFractionByMass/4.0 
        ds.add_field(("gas", "y_HeI"), function=_y_HeI, sampling_type="cell", units="1")
        def _y_HeII(field, data): return data["HeII_Density"] / data["Density"] /HydrogenFractionByMass/4.0 
        ds.add_field(("gas", "y_HeII"), function=_y_HeII, sampling_type="cell", units="1")
        def _y_HeIII(field, data): return data["HeIII_Density"] / data["Density"] /HydrogenFractionByMass/4.0 
        ds.add_field(("gas", "y_HeIII"), function=_y_HeIII, sampling_type="cell", units="1")
    if MultiSpecies > 1:
        def _y_HM(field, data): return data["HM_Density"] / data["Density"] /HydrogenFractionByMass/1.0
        ds.add_field(("gas", "y_HM"), function=_y_HM, sampling_type="cell", units="1")
        def _y_H2I(field, data): return data["H2I_Density"] / data["Density"] /HydrogenFractionByMass/2.0
        ds.add_field(("gas", "y_H2I"), function=_y_H2I, sampling_type="cell", units="1")
        def _y_H2II(field, data): return data["H2II_Density"] / data["Density"] /HydrogenFractionByMass/2.0
        ds.add_field(("gas", "y_H2II"), function=_y_H2II, sampling_type="cell", units="1")
    if MultiSpecies > 2:
        def _y_DI(field, data): return data["DI_Density"] / data["Density"] /HydrogenFractionByMass/2.0
        ds.add_field(("gas", "y_DI"), function=_y_DI, sampling_type="cell", units="1")
        def _y_DII(field, data): return data["DII_Density"] / data["Density"] /HydrogenFractionByMass/2.0
        ds.add_field(("gas", "y_DII"), function=_y_DII, sampling_type="cell", units="1")
        def _y_HDI(field, data): return data["HDI_Density"] / data["Density"] /HydrogenFractionByMass/3.0
        ds.add_field(("gas", "y_HDI"), function=_y_HDI, sampling_type="cell", units="1")
    if MultiSpecies > 3:
        def _y_DM(field, data): return data["DM_Density"] / data["Density"] /HydrogenFractionByMass/2.0
        ds.add_field(("gas", "y_DM"), function=_y_DM, sampling_type="cell", units="1")
        def _y_HDII(field, data): return data["HDII_Density"] / data["Density"] /HydrogenFractionByMass/3.0
        ds.add_field(("gas", "y_HDII"), function=_y_HDII, sampling_type="cell", units="1")
        def _y_HeHII(field, data): return data["HeHII_Density"] / data["Density"] /HydrogenFractionByMass/5.0
        ds.add_field(("gas", "y_HeHII"), function=_y_HeHII, sampling_type="cell", units="1")
    if MetalChemistry > 0:
        def _y_CI(field, data): return data["CI_Density"] / data["Density"] /HydrogenFractionByMass/12.0 
        ds.add_field(("gas", "y_CI"), function=_y_CI, sampling_type="cell", units="1")
        def _y_CII(field, data): return data["CII_Density"] / data["Density"] /HydrogenFractionByMass/12.0 
        ds.add_field(("gas", "y_CII"), function=_y_CII, sampling_type="cell", units="1")
        def _y_CO(field, data): return data["CO_Density"] / data["Density"] /HydrogenFractionByMass/28.0 
        ds.add_field(("gas", "y_CO"), function=_y_CO, sampling_type="cell", units="1")
        def _y_CO2(field, data): return data["CO2_Density"] / data["Density"] /HydrogenFractionByMass/44.0 
        ds.add_field(("gas", "y_CO2"), function=_y_CO2, sampling_type="cell", units="1")
        def _y_OI(field, data): return data["OI_Density"] / data["Density"] /HydrogenFractionByMass/16.0 
        ds.add_field(("gas", "y_OI"), function=_y_OI, sampling_type="cell", units="1")
        def _y_OH(field, data): return data["OH_Density"] / data["Density"] /HydrogenFractionByMass/17.0 
        ds.add_field(("gas", "y_OH"), function=_y_OH, sampling_type="cell", units="1")
        def _y_H2O(field, data): return data["H2O_Density"] / data["Density"] /HydrogenFractionByMass/18.0 
        ds.add_field(("gas", "y_H2O"), function=_y_H2O, sampling_type="cell", units="1")
        def _y_O2(field, data): return data["O2_Density"] / data["Density"] /HydrogenFractionByMass/32.0 
        ds.add_field(("gas", "y_O2"), function=_y_O2, sampling_type="cell", units="1")
        def _y_SiI(field, data): return data["SiI_Density"] / data["Density"] /HydrogenFractionByMass/28.0
        ds.add_field(("gas", "y_SiI"), function=_y_SiI, sampling_type="cell", units="1")
        def _y_SiOI(field, data): return data["SiOI_Density"] / data["Density"] /HydrogenFractionByMass/44.0
        ds.add_field(("gas", "y_SiOI"), function=_y_SiOI, sampling_type="cell", units="1")
        def _y_SiO2I(field, data): return data["SiO2I_Density"] / data["Density"] /HydrogenFractionByMass/60.0 
        ds.add_field(("gas", "y_SiO2I"), function=_y_SiO2I, sampling_type="cell", units="1")
        def _y_CH(field, data): return data["CH_Density"] / data["Density"] /HydrogenFractionByMass/13.0 
        ds.add_field(("gas", "y_CH"), function=_y_CH, sampling_type="cell", units="1")
        def _y_CH2(field, data): return data["CH2_Density"] / data["Density"] /HydrogenFractionByMass/14.0 
        ds.add_field(("gas", "y_CH2"), function=_y_CH2, sampling_type="cell", units="1")
        def _y_COII(field, data): return data["COII_Density"] / data["Density"] /HydrogenFractionByMass/28.0 
        ds.add_field(("gas", "y_COII"), function=_y_COII, sampling_type="cell", units="1")
        def _y_OII(field, data): return data["OII_Density"] / data["Density"] /HydrogenFractionByMass/16.0 
        ds.add_field(("gas", "y_OII"), function=_y_OII, sampling_type="cell", units="1")
        def _y_OHII(field, data): return data["OHII_Density"] / data["Density"] /HydrogenFractionByMass/17.0 
        ds.add_field(("gas", "y_OHII"), function=_y_OHII, sampling_type="cell", units="1")
        def _y_H2OII(field, data): return data["H2OII_Density"] / data["Density"] /HydrogenFractionByMass/18.0 
        ds.add_field(("gas", "y_H2OII"), function=_y_H2OII, sampling_type="cell", units="1")
        def _y_H3OII(field, data): return data["H3OII_Density"] / data["Density"] /HydrogenFractionByMass/19.0 
        ds.add_field(("gas", "y_H3OII"), function=_y_H3OII, sampling_type="cell", units="1")
        def _y_O2II(field, data): return data["O2II_Density"] / data["Density"] /HydrogenFractionByMass/32.0 
        ds.add_field(("gas", "y_O2II"), function=_y_O2II, sampling_type="cell", units="1")
        if GrainGrowth > 0:
            if DustSpecies > 0:
                def _y_Mg(field, data): return data["Mg_Density"] / data["Density"] /HydrogenFractionByMass/24.0 
                ds.add_field(("gas", "y_Mg"), function=_y_Mg, sampling_type="cell", units="1")
            if DustSpecies > 1:
                def _y_Al(field, data): return data["Al_Density"] / data["Density"] /HydrogenFractionByMass/27.0 
                ds.add_field(("gas", "y_Al"), function=_y_Al, sampling_type="cell", units="1")
                def _y_S(field, data): return data["S_Density"] / data["Density"] /HydrogenFractionByMass/32.0 
                ds.add_field(("gas", "y_S"), function=_y_S, sampling_type="cell", units="1")
                def _y_Fe(field, data): return data["Fe_Density"] / data["Density"] /HydrogenFractionByMass/56.0 
                ds.add_field(("gas", "y_Fe"), function=_y_Fe, sampling_type="cell", units="1")
    if GrainGrowth > 0:
        if DustSpecies > 0:
            def _y_MgSiO3(field, data): return data["MgSiO3_Density"] / data["Density"] /HydrogenFractionByMass/100.0
            ds.add_field(("gas", "y_MgSiO3"), function=_y_MgSiO3, sampling_type="cell", units="1")
            def _y_AC(field, data): return data["AC_Density"] / data["Density"] /HydrogenFractionByMass/12.0
            ds.add_field(("gas", "y_AC"), function=_y_AC, sampling_type="cell", units="1")
        if DustSpecies > 1:
            def _y_SiM(field, data): return data["SiM_Density"] / data["Density"] /HydrogenFractionByMass/28.0
            ds.add_field(("gas", "y_SiM"), function=_y_SiM, sampling_type="cell", units="1")
            def _y_FeM(field, data): return data["FeM_Density"] / data["Density"] /HydrogenFractionByMass/56.0
            ds.add_field(("gas", "y_FeM"), function=_y_FeM, sampling_type="cell", units="1")
            def _y_Mg2SiO4(field, data): return data["Mg2SiO4_Density"] / data["Density"] /HydrogenFractionByMass/140.0
            ds.add_field(("gas", "y_Mg2SiO4"), function=_y_Mg2SiO4, sampling_type="cell", units="1")
            def _y_Fe3O4(field, data): return data["Fe3O4_Density"] / data["Density"] /HydrogenFractionByMass/232.0
            ds.add_field(("gas", "y_Fe3O4"), function=_y_Fe3O4, sampling_type="cell", units="1")
            def _y_SiO2D(field, data): return data["SiO2D_Density"] / data["Density"] /HydrogenFractionByMass/60.0
            ds.add_field(("gas", "y_SiO2D"), function=_y_SiO2D, sampling_type="cell", units="1")
            def _y_MgO(field, data): return data["MgO_Density"] / data["Density"] /HydrogenFractionByMass/40.0
            ds.add_field(("gas", "y_MgO"), function=_y_MgO, sampling_type="cell", units="1")
            def _y_FeS(field, data): return data["FeS_Density"] / data["Density"] /HydrogenFractionByMass/88.0
            ds.add_field(("gas", "y_FeS"), function=_y_FeS, sampling_type="cell", units="1")
            def _y_Al2O3(field, data): return data["Al2O3_Density"] / data["Density"] /HydrogenFractionByMass/102.0
            ds.add_field(("gas", "y_Al2O3"), function=_y_Al2O3, sampling_type="cell", units="1")

    if MultiSpecies > 0:
        def _ThermalEnergy(field, data): return data["GasEnergy"] * data["cell_mass"]
        ds.add_field(("gas", "ThermalEnergy"), function=_ThermalEnergy, sampling_type="cell", units="erg")
        def _TotEnergy(field, data): return data["TotalEnergy"] * data["cell_mass"]
        ds.add_field(("gas", "TotEnergy"), function=_TotEnergy, sampling_type="cell", units="erg")
    if MetalChemistry:
##      def _MetalMass(field, data): return data["SN_Colour"] * data["cell_volume"]
##      ds.add_field(("gas", "MetalMass"), function=_MetalMass, sampling_type="cell", units="code_mass")
        def _MetalMass(field, data): return data["Metal_Density"] * data["cell_volume"]
        ds.add_field(("gas", "MetalMass"), function=_MetalMass, sampling_type="cell", units="code_mass")

    ds.add_particle_filter('dark_matter')
    min_dm_mass = ad.quantities.extrema(('dark_matter','particle_mass')).in_units("Msun")
    if yt.is_root():
        print(min_dm_mass)
        print(datetime.datetime.now(), flush=True)

    if MultiSpecies > 0:
        def _molecular_weight(field, data):
            return data["Density"] / (
                   data["Electron_Density"]
                 + data[      "HI_Density"]
                 + data[     "HII_Density"]
                 + data[     "HeI_Density"] / 4.0
                 + data[    "HeII_Density"] / 4.0
                 + data[   "HeIII_Density"] / 4.0
                 + data[      "HM_Density"]
                 + data[     "H2I_Density"] / 2.0
                 + data[    "H2II_Density"] / 2.0
                 ) * mp
        ds.add_field(("gas", "molecular_weight"), function=_molecular_weight, sampling_type="cell", units="g")
        def _adiabatic_index(field, data):
            gamma0=  1.0 + (
                   data["Electron_Density"]
                 + data[      "HI_Density"]
                 + data[     "HII_Density"]
                 + data[     "HeI_Density"] / 4.0
                 + data[    "HeII_Density"] / 4.0
                 + data[   "HeIII_Density"] / 4.0
                 + data[      "HM_Density"]
                 + data[     "H2I_Density"] / 2.0
                 + data[    "H2II_Density"] / 2.0
                 ) / (
                                   1.5 * data["Electron_Density"]
                 +                 1.5 * data[      "HI_Density"]
                 +                 1.5 * data[     "HII_Density"]
                 +                 1.5 * data[     "HeI_Density"] / 4.0
                 +                 1.5 * data[    "HeII_Density"] / 4.0
                 +                 1.5 * data[   "HeIII_Density"] / 4.0
                 +                 1.5 * data[      "HM_Density"]
                 +                 2.5 * data[     "H2I_Density"] / 2.0
                 +                 2.5 * data[    "H2II_Density"] / 2.0
                 )
            iteration = 0;
            while True:
                k6100 = ds.arr(8.421956e-13, 'erg')
                kT = (gamma0-1.0) * data["GasEnergy"] * data["molecular_weight"].in_units("code_mass") 
                T6100 = k6100 / kT
                exp_T6100 = np.exp(T6100)
                gamma_minus1_inv_H2 = 0.5*(5.0 + 2.0 * T6100**2 * exp_T6100 / (exp_T6100-1.0)**2)
                gamma_minus1_inv_H2[T6100 > 100.0] = 2.5
                gamma =  1.0 + (
                       data["Electron_Density"]
                     + data[      "HI_Density"]
                     + data[     "HII_Density"]
                     + data[     "HeI_Density"] / 4.0
                     + data[    "HeII_Density"] / 4.0
                     + data[   "HeIII_Density"] / 4.0
                     + data[      "HM_Density"]
                     + data[     "H2I_Density"] / 2.0
                     + data[    "H2II_Density"] / 2.0
                     ) / (
                                       1.5 * data["Electron_Density"]
                     +                 1.5 * data[      "HI_Density"]
                     +                 1.5 * data[     "HII_Density"]
                     +                 1.5 * data[     "HeI_Density"] / 4.0
                     +                 1.5 * data[    "HeII_Density"] / 4.0
                     +                 1.5 * data[   "HeIII_Density"] / 4.0
                     +                 1.5 * data[      "HM_Density"]
                     + gamma_minus1_inv_H2 * data[     "H2I_Density"] / 2.0
                     + gamma_minus1_inv_H2 * data[    "H2II_Density"] / 2.0
                     )
                red = np.abs(gamma - gamma0)/gamma0;
                if ((len(red)>0 and red.max() < 1.0e-10) or len(red)==0) or iteration > 100:
                    break
                gamma0 = gamma
                iteration = iteration + 1
            return gamma
        ds.add_field(("gas", "adiabatic_index"), function=_adiabatic_index, sampling_type="cell")
        def _temperature_corr(field, data): return (data["adiabatic_index"]-1.0) * data["GasEnergy"] * data["molecular_weight"] / kboltz
        ds.add_field(("gas", "temperature_corr"), function=_temperature_corr, sampling_type="cell", units="K")
        def _sound_speed_corr(field, data): return (data["adiabatic_index"] * kboltz * data["temperature_corr"] / data["molecular_weight"])**0.5
        ds.add_field(("gas", "sound_speed_corr"), function=_sound_speed_corr, sampling_type="cell", units="code_velocity")
        def _pressure_corr(field, data): return data["Density"] * kboltz * data["temperature_corr"] / data["molecular_weight"]
        ds.add_field(("gas", "pressure_corr"), function=_pressure_corr, sampling_type="cell", units="code_pressure")
       
        def _HydrogenMass(field, data): 
            if MultiSpecies > 0:
                HydrogenDensity = (
                   data[   "HI_Density"]
                 + data[  "HII_Density"]
                )
            if MultiSpecies > 1:
                HydrogenDensity = HydrogenDensity + (
                   data[   "HM_Density"]
                 + data[  "H2I_Density"]/ 2.0 * 2.0
                 + data[ "H2II_Density"]/ 2.0 * 2.0
                )
            if MultiSpecies > 2:
                HydrogenDensity = HydrogenDensity + (
                   data[  "HDI_Density"]/ 3.0
                )
            if MultiSpecies > 3:
                HydrogenDensity = HydrogenDensity + (
                   data[ "HDII_Density"]/ 3.0
                 + data["HeHII_Density"]/ 5.0
                )
            if MetalChemistry:
                HydrogenDensity = HydrogenDensity + (
                   data[   "OH_Density"]/17.0
                 + data[  "H2O_Density"]/18.0 * 2.0
                 + data[   "CH_Density"]/13.0
                 + data[  "CH2_Density"]/14.0 * 2.0
                 + data[ "OHII_Density"]/17.0
                 + data["H2OII_Density"]/18.0 * 2.0
                 + data["H3OII_Density"]/19.0 * 3.0
                )
            return HydrogenDensity * 1.0 * data["cell_volume"]
        ds.add_field(("gas", "HydrogenMass"), function=_HydrogenMass, sampling_type="cell", units="code_mass")
       
        def _HydrogenFraction(field, data): return data["HydrogenMass"] / data["cell_mass"] / HydrogenFractionByMass
        ds.add_field(("gas", "HydrogenFraction"), function=_HydrogenFraction, sampling_type="cell", units="1")

    if MetalChemistry:
        def _CarbonMass(field, data): return (
            data[  "CI_Density"]/12.0
          + data[ "CII_Density"]/12.0
          + data[  "CO_Density"]/28.0
          + data[ "CO2_Density"]/44.0
          + data[  "CH_Density"]/13.0
          + data[ "CH2_Density"]/14.0
          + data["COII_Density"]/28.0
          + data[  "AC_Density"]/12.0
          ) * 12.0 * data["cell_volume"]
        ds.add_field(("gas", "CarbonMass"), function=_CarbonMass, sampling_type="cell", units="code_mass")
    
        def _OxygenMass(field, data): return (
            data[     "CO_Density"]/28.0
          + data[    "CO2_Density"]/22.0
          + data[     "OI_Density"]/16.0
          + data[     "OH_Density"]/17.0
          + data[    "H2O_Density"]/18.0
          + data[     "O2_Density"]/16.0
          + data[   "SiOI_Density"]/44.0
          + data[  "SiO2I_Density"]/30.0
          + data[   "COII_Density"]/28.0
          + data[    "OII_Density"]/16.0
          + data[   "OHII_Density"]/17.0
          + data[  "H2OII_Density"]/18.0
          + data[  "H3OII_Density"]/19.0
          + data[   "O2II_Density"]/16.0
          + data["Mg2SiO4_Density"]/35.0
          + data[ "MgSiO3_Density"]/33.3
          + data[  "Fe3O4_Density"]/58.0
          + data[  "SiO2D_Density"]/30.0
          + data[    "MgO_Density"]/40.0
          + data[  "Al2O3_Density"]/34.0
          ) * 16.0 * data["cell_volume"]
        ds.add_field(("gas", "OxygenMass"), function=_OxygenMass, sampling_type="cell", units="code_mass")
    
        def _SiliconMass(field, data): return (
            data[    "SiI_Density"]/ 28.0
          + data[   "SiOI_Density"]/ 44.0
          + data[  "SiO2I_Density"]/ 60.0
          + data[    "SiM_Density"]/ 28.0
          + data["Mg2SiO4_Density"]/140.0
          + data[ "MgSiO3_Density"]/100.0
          + data[  "SiO2D_Density"]/60.0
          ) * 28.0 * data["cell_volume"]
        ds.add_field(("gas", "SiliconMass"), function=_SiliconMass, sampling_type="cell", units="code_mass")
    
        def _IronMass(field, data): return (
            data[   "Fe_Density"]/56.0
          + data[  "FeM_Density"]/56.0
          + data["Fe3O4_Density"]/77.3
          + data[  "FeS_Density"]/88.0
          ) * 56.0 * data["cell_volume"]
        ds.add_field(("gas", "IronMass"), function=_IronMass, sampling_type="cell", units="code_mass")

        def _CarbonFraction(field, data): return data["CarbonMass"] / data["MetalMass"]
        ds.add_field(("gas", "CarbonFraction"), function=_CarbonFraction, sampling_type="cell", units="1")
        def _OxygenFraction(field, data): return data["OxygenMass"] / data["MetalMass"]
        ds.add_field(("gas", "OxygenFraction"), function=_OxygenFraction, sampling_type="cell", units="1")
        def _SiliconFraction(field, data): return data["SiliconMass"] / data["MetalMass"]
        ds.add_field(("gas", "SiliconFraction"), function=_SiliconFraction, sampling_type="cell", units="1")
        def _IronFraction(field, data): return data["IronMass"] / data["MetalMass"]
        ds.add_field(("gas", "IronFraction"), function=_IronFraction, sampling_type="cell", units="1")
    
        def _CarbonAbundance(field, data): return 1.0e12 * (data["CarbonMass"] / 12.0 / data["HydrogenMass"])
        ds.add_field(("gas", "CarbonAbundance"), function=_CarbonAbundance, sampling_type="cell")
        def _IronAbundanceToSolar(field, data): return 1.0e12 * (data["IronMass"] / 56.0 / data["HydrogenMass"]) / 10.0**SolarIronAbundance
        ds.add_field(("gas", "IronAbundanceToSolar"), function=_IronAbundanceToSolar, sampling_type="cell")
    
        def _CarbonDustMass(field, data): return data["AC_Density"] * data["cell_volume"]
        ds.add_field(("gas", "CarbonDustMass"), function=_CarbonDustMass, sampling_type="cell", units="code_mass")
        def _CarbonCondensationEfficiency(field, data): return data["CarbonDustMass"] / data["CarbonMass"]
        ds.add_field(("gas", "CarbonCondensationEfficiency"), function=_CarbonCondensationEfficiency, sampling_type="cell")

    if MultiSpecies > 0:
        nHmax = ad.max("Hydrogen_number_density")
        posi_nHmax = ad.argmax("density")
        if yt.is_root():
            print("Max density %e at %f %f %f" % (
                  nHmax
                , posi_nHmax[0].in_units("code_length")
                , posi_nHmax[1].in_units("code_length")
                , posi_nHmax[2].in_units("code_length")))
            print(datetime.datetime.now(), flush=True)
        
        cs_nHmax = ad.argmax("density", axis="sound_speed_corr")
        l_J = (pi * cs_nHmax**2 / GravConst / ad.max("Density"))**0.5
        '''
        nHmax      = ds.arr(6.840720e+05, "1/cm**3")
        posi_nHmax = ds.arr([0.528827, 0.498918, 0.524557], "code_length")
        cs_nHmax   = ds.arr(3.107246e+00,"km/s")
        l_J        = ds.arr(5.628728e-01, "pc")
        '''
        if yt.is_root():
            print("dens %e cs %e l_J %e pc" % (nHmax, cs_nHmax.in_units("km/s"), (l_J.in_units("pc")) ) )
            print(datetime.datetime.now(), flush=True)

# ELLIPTICITY OF GAS CLOUD
    if nHmax > 1.0e10:
        cp = posi_nHmax
        nH_thr = nHmax / 3.0
        filament = ds.sphere(cp, 10.0*l_J).cut_region([("obj['Hydrogen_number_density'] > %e" % nH_thr)])
            
        def _moi00(field, data): return data["cell_mass"] * ((data["y"]-cp[1])**2 + (data["z"]-cp[2])**2)
        ds.add_field(("gas", "moi00"), function=_moi00, sampling_type="cell", units="code_mass*code_length**2")
        def _moi01(field, data): return data["cell_mass"] * ( -(data["x"]-cp[0]) * (data["y"]-cp[1]))
        ds.add_field(("gas", "moi01"), function=_moi01, sampling_type="cell", units="code_mass*code_length**2")
        def _moi02(field, data): return data["cell_mass"] * ( -(data["x"]-cp[0]) * (data["z"]-cp[2]))
        ds.add_field(("gas", "moi02"), function=_moi02, sampling_type="cell", units="code_mass*code_length**2")
        def _moi10(field, data): return data["cell_mass"] * ( -(data["y"]-cp[1]) * (data["x"]-cp[0]))
        ds.add_field(("gas", "moi10"), function=_moi10, sampling_type="cell", units="code_mass*code_length**2")
        def _moi11(field, data): return data["cell_mass"] * ((data["x"]-cp[0])**2 + (data["z"]-cp[2])**2)
        ds.add_field(("gas", "moi11"), function=_moi11, sampling_type="cell", units="code_mass*code_length**2")
        def _moi12(field, data): return data["cell_mass"] * ( -(data["y"]-cp[1]) * (data["z"]-cp[2]))
        ds.add_field(("gas", "moi12"), function=_moi12, sampling_type="cell", units="code_mass*code_length**2")
        def _moi20(field, data): return data["cell_mass"] * ( -(data["z"]-cp[2]) * (data["x"]-cp[0]))
        ds.add_field(("gas", "moi20"), function=_moi20, sampling_type="cell", units="code_mass*code_length**2")
        def _moi21(field, data): return data["cell_mass"] * ( -(data["z"]-cp[2]) * (data["y"]-cp[1]))
        ds.add_field(("gas", "moi21"), function=_moi21, sampling_type="cell", units="code_mass*code_length**2")
        def _moi22(field, data): return data["cell_mass"] * ((data["x"]-cp[0])**2 + (data["y"]-cp[1])**2)
        ds.add_field(("gas", "moi22"), function=_moi22, sampling_type="cell", units="code_mass*code_length**2")
      
        if l_J > ds.arr(1.0, "pc"):
            ref_unit_mass = "Msun"
            ref_unit_leng = "kpc"
        else:
            ref_unit_mass = "Msun"
            ref_unit_leng = "au"
        ref_unit_moi = ref_unit_mass + '*' + ref_unit_leng + '**2'
        axes_unit = ref_unit_leng
      
        moi = np.array([[0.0]*3]*3)
        moi[0, 0] = filament.sum('moi00').in_units(ref_unit_moi)
        moi[1, 0] = filament.sum('moi01').in_units(ref_unit_moi)
        moi[2, 0] = filament.sum('moi02').in_units(ref_unit_moi)
        moi[0, 1] = filament.sum('moi10').in_units(ref_unit_moi)
        moi[1, 1] = filament.sum('moi11').in_units(ref_unit_moi)
        moi[2, 1] = filament.sum('moi12').in_units(ref_unit_moi)
        moi[0, 2] = filament.sum('moi20').in_units(ref_unit_moi)
        moi[1, 2] = filament.sum('moi21').in_units(ref_unit_moi)
        moi[2, 2] = filament.sum('moi22').in_units(ref_unit_moi)
        mtot = np.array(filament.sum('cell_mass').in_units(ref_unit_mass))
#       print(moi)
#       print(mtot)
        
        moi_l, moi_lv = LA.eig(moi)
#       print(moi_l)
#       print(moi_lv)
        
        lax = np.array([0.0]*3)
        lax[0] = (2.5*(-moi_l[0]+moi_l[1]+moi_l[2])/mtot)**0.5
        lax[1] = (2.5*( moi_l[0]-moi_l[1]+moi_l[2])/mtot)**0.5   
        lax[2] = (2.5*( moi_l[0]+moi_l[1]-moi_l[2])/mtot)**0.5
#       print(lax)
        laxis = ds.arr(lax, ref_unit_leng)
        aaxis = ds.arr(moi_lv, 'dimensionless')
#       print(laxis)
#       print('axis0 %13.7f [%13.7f, %13.7f, %13.7f]' % (laxis[0], aaxis[0,0], aaxis[1,0], aaxis[2,0]))
#       print('axis1 %13.7f [%13.7f, %13.7f, %13.7f]' % (laxis[1], aaxis[0,1], aaxis[1,1], aaxis[2,1]))
#       print('axis2 %13.7f [%13.7f, %13.7f, %13.7f]' % (laxis[2], aaxis[0,2], aaxis[1,2], aaxis[2,2]))
      
        iaxis_max = np.argmax(lax)
        iaxis_min = np.argmin(lax)
        iaxis_mid = 3 - iaxis_max - iaxis_min
#       print("%d %d %d" % (iaxis_max, iaxis_mid, iaxis_min))
        axis0 = moi_lv[:, iaxis_max]
        axis1 = moi_lv[:, iaxis_mid]
        axis2 = moi_lv[:, iaxis_min]
#       print('axis0 %13.7f [%13.7f, %13.7f, %13.7f]' % (lax[iaxis_max], axis0[0], axis0[1], axis0[2]))
#       print('axis1 %13.7f [%13.7f, %13.7f, %13.7f]' % (lax[iaxis_mid], axis1[0], axis1[1], axis1[2]))
#       print('axis2 %13.7f [%13.7f, %13.7f, %13.7f]' % (lax[iaxis_min], axis2[0], axis2[1], axis2[2]))
        
        laaxis = ds.arr([[0.0]*3]*3, 'code_length')
        laaxis[0, 0] = cp[0] + laxis[0].in_units('code_length') * aaxis[0, 0]
        laaxis[1, 0] = cp[1] + laxis[0].in_units('code_length') * aaxis[1, 0]
        laaxis[2, 0] = cp[2] + laxis[0].in_units('code_length') * aaxis[2, 0]
        laaxis[0, 1] = cp[0] + laxis[1].in_units('code_length') * aaxis[0, 1]
        laaxis[1, 1] = cp[1] + laxis[1].in_units('code_length') * aaxis[1, 1]
        laaxis[2, 1] = cp[2] + laxis[1].in_units('code_length') * aaxis[2, 1]
        laaxis[0, 2] = cp[0] + laxis[2].in_units('code_length') * aaxis[0, 2]
        laaxis[1, 2] = cp[1] + laxis[2].in_units('code_length') * aaxis[1, 2]
        laaxis[2, 2] = cp[2] + laxis[2].in_units('code_length') * aaxis[2, 2]
        
        endpoint0 = np.array(laaxis[:, 0])
        endpoint1 = np.array(laaxis[:, 1])
        endpoint2 = np.array(laaxis[:, 2])
        if yt.is_root():
            print('endpoint0 [%13.7f, %13.7f, %13.7f]' % (endpoint0[0], endpoint0[1], endpoint0[2]))
            print('endpoint1 [%13.7f, %13.7f, %13.7f]' % (endpoint1[0], endpoint1[1], endpoint1[2]))
            print('endpoint2 [%13.7f, %13.7f, %13.7f]' % (endpoint2[0], endpoint2[1], endpoint2[2]))
            print(datetime.datetime.now(), flush=True)

#   box = ds.box(ds.arr([0.4,0.4,0.4],"code_length")
#              , ds.arr([0.6,0.6,0.6],"code_length"))

#   if MetalChemistry:
#       GasMass_in_ad    = ad.sum("cell_mass" ).in_units("Msun")
#       MetalMass_in_ad  = ad.sum("MetalMass" ).in_units("Msun")
#       CarbonMass_in_ad = ad.sum("CarbonMass").in_units("Msun")
#       OxygenMass_in_ad = ad.sum("OxygenMass").in_units("Msun")
#       IronMass_in_ad   = ad.sum("IronMass"  ).in_units("Msun")
#       CarbonFrac_in_ad = CarbonMass_in_ad / MetalMass_in_ad
#       OxygenFrac_in_ad = OxygenMass_in_ad / MetalMass_in_ad
#       IronFrac_in_ad   =   IronMass_in_ad / MetalMass_in_ad
#   
#       CarbonDustMass_in_ad = ad.sum("CarbonDustMass").in_units("Msun")
#       CarbonDustFrac_in_ad = CarbonDustMass_in_ad / MetalMass_in_ad
#   
#       if yt.is_root():
#           print("Metal Mass %11.5f Msun in all region" % MetalMass_in_ad)
#           print("  C  frac %13.5e" % (CarbonFrac_in_ad))
#           print("  O  frac %13.5e" % (OxygenFrac_in_ad))
#           print(" Fe  frac %13.5e" % (  IronFrac_in_ad))

####### TOO SLOW
####### enriched = ad.cut_region(["obj['Zmet'] > 1.0e-8"])
####### GasMass_in_enriched    = enriched.sum("cell_mass" ).in_units("Msun")
####### MetalMass_in_enriched  = enriched.sum("MetalMass" ).in_units("Msun")
####### CarbonMass_in_enriched = enriched.sum("CarbonMass").in_units("Msun")
####### OxygenMass_in_enriched = enriched.sum("OxygenMass").in_units("Msun")
####### IronMass_in_enriched   = enriched.sum("IronMass"  ).in_units("Msun")
####### CarbonFrac_in_enriched = CarbonMass_in_enriched / MetalMass_in_enriched
####### OxygenFrac_in_enriched = OxygenMass_in_enriched / MetalMass_in_enriched
####### IronFrac_in_enriched   =   IronMass_in_enriched / MetalMass_in_enriched
#######
####### CarbonDustMass_in_enriched = enriched.sum("CarbonDustMass").in_units("Msun")
####### CarbonDustFrac_in_enriched = CarbonDustMass_in_enriched / MetalMass_in_enriched
#######
####### if yt.is_root():
#######     print("Metal Mass %f Msun in enriched region" % MetalMass_in_enriched)
#######     print("  C  frac %13.5e" % (CarbonFrac_in_enriched))
#######     print("  O  frac %13.5e" % (OxygenFrac_in_enriched))
#######     print(" Fe  frac %13.5e" % (  IronFrac_in_enriched))

    if TRACE_MASS:
        fp_trace_mass.write("%23.15e " % ds.current_time)
##      fp_trace_mass.write("%5d " % nBH)
        fp_trace_mass.write("%13.5e "% (GasMass_in_ad   ) )
        fp_trace_mass.write("%13.5e "% (MetalMass_in_ad ) )
        fp_trace_mass.write("%13.5e "% (CarbonMass_in_ad / MetalMass_in_ad) )
        fp_trace_mass.write("%13.5e "% (OxygenMass_in_ad / MetalMass_in_ad) )
        fp_trace_mass.write("%13.5e "% (IronMass_in_ad   / MetalMass_in_ad) )
        fp_trace_mass.write("%13.5e "% (GasMass_in_enriched   ) )
        fp_trace_mass.write("%13.5e "% (MetalMass_in_enriched ) )
        fp_trace_mass.write("%13.5e "% (CarbonMass_in_enriched / MetalMass_in_enriched) )
        fp_trace_mass.write("%13.5e "% (OxygenMass_in_enriched / MetalMass_in_enriched) )
        fp_trace_mass.write("%13.5e "% (IronMass_in_enriched   / MetalMass_in_enriched) )
        fp_trace_mass.write("\n")                                          


    if HALO:
        if HALOFIND:
            ds.add_particle_filter('dark_matter')
            # Determine highest resolution DM particle mass in sim by looking
            # at the extrema of the dark_matter particle_mass field.
            ad = ds.all_data()
            min_dm_mass = ad.quantities.extrema(('dark_matter','particle_mass'))[0]
            # Define a new particle filter to isolate all highest resolution DM particles
            # and apply it to dataset
            def MaxResDarkMatter(pfilter, data):
                return data["particle_mass"] <= 1.01 * min_dm_mass 
            add_particle_filter("max_res_dark_matter", function=MaxResDarkMatter
                              , filtered_type='dark_matter', requires=["particle_mass"])
            ds.add_particle_filter('max_res_dark_matter')
            # If desired, we can see the total number of DM and High-res DM particles
            if yt.is_root():
             print("Simulation has %d DM particles." %
             ad['dark_matter','particle_type'].shape)
             print("Simulation has %d Highest Res DM particles." %
             ad['max_res_dark_matter', 'particle_type'].shape)
             print(datetime.datetime.now(), flush=True)
            # Run the halo catalog on the dataset only on the highest resolution dark matter
            # particles
            hc = HaloCatalog(data_ds=ds, finder_method='rockstar'
                           , finder_kwargs={'dm_only':True, 'particle_type':'max_res_dark_matter'})
            hc.create()

        else:
            # Load the halo list from a rockstar output for this dataset
            halos = yt.load(indir + 'rockstar_halos/halos_0.0.bin')
#           halos = yt.load(indir + 'rockstar_halos_' + number + version + '/halos_0.0.bin')
            # Create the halo catalog from this halo list
            hc = HaloCatalog(halos_ds=halos)
            hc.load()

        halo_ad = hc.halos_ds.all_data()
        halo_masses = halo_ad['particle_mass'][:]
        halo_radii  = halo_ad['virial_radius'][:]
        halo_x      = halo_ad['particle_position_x'][:]
        halo_y      = halo_ad['particle_position_y'][:]
        halo_z      = halo_ad['particle_position_z'][:]

        if yt.is_root():
            for ihalo in range(halo_masses.size):
                if halo_masses[ihalo] > ds.arr(1.0e5, 'Msun'):
                    print("%5d/%9d %9.5f %9.5f %9.5f %13.5e %13.5e" % (
                       ihalo, halo_masses.size
                     , halo_x     [ihalo].in_units('Mpccm/h')
                     , halo_y     [ihalo].in_units('Mpccm/h')
                     , halo_z     [ihalo].in_units('Mpccm/h')
                     , halo_masses[ihalo].in_units('Msun')
                     , halo_radii [ihalo].in_units('kpc')   ))
            print(datetime.datetime.now(), flush=True)
    
        ihalo_max = halo_masses.argmax()
        halo_x_max       = halo_x[ihalo_max].in_units('code_length')
        halo_y_max       = halo_y[ihalo_max].in_units('code_length')
        halo_z_max       = halo_z[ihalo_max].in_units('code_length')
        halo_mass_max    = halo_masses[ihalo_max].in_units('Msun')
        halo_radius_max  = halo_radii [ihalo_max].in_units('kpc')

        if yt.is_root():
            print("posi %23.17f, %23.17f, %23.17f" % (halo_x_max, halo_y_max, halo_z_max))
            print("M200 %23.15e" % halo_mass_max)
            print("R200 %23.17f" % halo_radius_max)
            print(datetime.datetime.now(), flush=True)

        if StarParticleCreation:
          idPopIII = []; idBH     = []; idPopII  = []
          cPopIII  = []; cBH      = []; cPopII   = []
          dPopIII  = []; dBH      = []; dPopII   = []
          tPopIII  = []; tBH      = []; tPopII   = []
          MPopIII  = []; MBH      = []; MPopII   = []
          xPopIII  = []; xBH      = []; xPopII   = []
          yPopIII  = []; yBH      = []; yPopII   = []
          zPopIII  = []; zBH      = []; zPopII   = []
          nPopIII  = 0 ; nBH      = 0 ; nPopII   = 0

          for ihalo in range(halo_masses.size):
            if halo_masses[ihalo] > ds.arr(1.0e5, 'Msun'):
              halo = ds.sphere([halo_x[ihalo], halo_y[ihalo], halo_z[ihalo]], halo_radii[ihalo])

              ds.add_particle_filter('PopIII')
              idPopIII    .extend( halo[("PopIII", "particle_index")]                              )
#             if halo[("PopIII", "particle_index")] is not None:
              nPopIII = halo[("PopIII", "particle_index")].size
              if nPopIII:
                  cPopIII .extend( halo[("PopIII", "creation_time")].in_units("Myr")               )
                  dPopIII .extend( halo[("PopIII", "dynamical_time")].in_units("Myr")              )
                  tPopIII .extend( halo[("PopIII", "age")].in_units("Myr")                         )
                  MPopIII .extend( halo[("PopIII", "particle_mass")].in_units("Msun")              )
                  xPopIII .extend( halo[("PopIII", "particle_position_x")].in_units("code_length") )
                  yPopIII .extend( halo[("PopIII", "particle_position_y")].in_units("code_length") )
                  zPopIII .extend( halo[("PopIII", "particle_position_z")].in_units("code_length") )
                  print("PopIII hosting halo %13.7f %13.7f %13.7f %13.5e" % (
                        halo_x[ihalo].in_units("code_length")
                      , halo_y[ihalo].in_units("code_length")
                      , halo_z[ihalo].in_units("code_length")
                      , halo_masses[ihalo].in_units("Msun")
                       ));

              ds.add_particle_filter('BH'    )
              idBH        .extend( halo[("BH"    , "particle_index")]                              )
              nBH     = halo[("BH"    , "particle_index")].size
              if nBH    :
                  cBH     .extend( halo[("BH"    , "creation_time")].in_units("Myr")               )
                  dBH     .extend( halo[("BH"    , "dynamical_time")].in_units("Myr")              )
                  tBH     .extend( halo[("BH"    , "age")].in_units("Myr")                         )
                  MBH     .extend( halo[("BH"    , "particle_mass")].in_units("Msun")              )
                  xBH     .extend( halo[("BH"    , "particle_position_x")].in_units("code_length") )
                  yBH     .extend( halo[("BH"    , "particle_position_y")].in_units("code_length") )
                  zBH     .extend( halo[("BH"    , "particle_position_z")].in_units("code_length") )

              ds.add_particle_filter('PopII' )
              idPopII     .extend( halo[("PopII" , "particle_index")]                              )
              nPopII  = halo[("PopII" , "particle_index")].size
              if nPopII :
                  cPopII  .extend( halo[("PopII" , "creation_time")].in_units("Myr")               )
                  dPopII  .extend( halo[("PopII" , "dynamical_time")].in_units("Myr")              )
                  tPopII  .extend( halo[("PopII" , "age")].in_units("Myr")                         )
                  MPopII  .extend( halo[("PopII" , "particle_mass")].in_units("Msun")              )
                  xPopII  .extend( halo[("PopII" , "particle_position_x")].in_units("code_length") )
                  yPopII  .extend( halo[("PopII" , "particle_position_y")].in_units("code_length") )
                  zPopII  .extend( halo[("PopII" , "particle_position_z")].in_units("code_length") )
            
    else:
        if StarParticleCreation:
          ds.add_particle_filter('PopIII')
          idPopIII    = ad[("PopIII", "particle_index")]
          nPopIII = idPopIII.size
          if nPopIII:
              cPopIII = ad[("PopIII", "creation_time")].in_units("Myr")
              dPopIII = ad[("PopIII", "dynamical_time")].in_units("Myr")
              tPopIII = ad[("PopIII", "age")].in_units("kyr")
              MPopIII = ad[("PopIII", "particle_mass")].in_units("Msun")
              xPopIII = ad[("PopIII", "particle_position_x")].in_units("code_length")
              yPopIII = ad[("PopIII", "particle_position_y")].in_units("code_length")
              zPopIII = ad[("PopIII", "particle_position_z")].in_units("code_length")
          if yt.is_root():
              print("PopIII")
              print(datetime.datetime.now(), flush=True)
         
          ds.add_particle_filter('BH')
          idBH        = ad[("BH"    , "particle_index")]
          nBH     = idBH    .size
          if nBH    :
              cBH     = ad[("BH"    , "creation_time")].in_units("Myr")
              dBH     = ad[("BH"    , "dynamical_time")].in_units("Myr")
              tBH     = ad[("BH"    , "age")].in_units("kyr")
              MBH     = ad[("BH"    , "particle_mass")].in_units("Msun")
              xBH     = ad[("BH"    , "particle_position_x")].in_units("code_length")
              yBH     = ad[("BH"    , "particle_position_y")].in_units("code_length")
              zBH     = ad[("BH"    , "particle_position_z")].in_units("code_length")
          if yt.is_root():
              print("BH    ")
              print(datetime.datetime.now(), flush=True)
         
          ds.add_particle_filter('PopII')
          idPopII     = ad[("PopII" , "particle_index")]
          nPopII  = idPopII .size
          if nPopII :
              cPopII  = ad[("PopII" , "creation_time")].in_units("Myr")
              dPopII  = ad[("PopII" , "dynamical_time")].in_units("Myr")
              tPopII  = ad[("PopII" , "age")].in_units("kyr")
              MPopII  = ad[("PopII" , "particle_mass")].in_units("Msun")
              xPopII  = ad[("PopII" , "particle_position_x")].in_units("code_length")
              yPopII  = ad[("PopII" , "particle_position_y")].in_units("code_length")
              zPopII  = ad[("PopII" , "particle_position_z")].in_units("code_length")
          if yt.is_root():
              print("PopII ")
              print(datetime.datetime.now(), flush=True)

##        idPopIII = []; idBH     = []; idPopII  = []
##        cPopIII  = []; cBH      = []; cPopII   = []
##        dPopIII  = []; dBH      = []; dPopII   = []
##        tPopIII  = []; tBH      = []; tPopII   = []
##        MPopIII  = []; MBH      = []; MPopII   = []
##        xPopIII  = []; xBH      = []; xPopII   = []
##        yPopIII  = []; yBH      = []; yPopII   = []
##        zPopIII  = []; zBH      = []; zPopII   = []
##        nPopIII  = 0 ; nBH      = 0 ; nPopII   = 0
##    
##        if TopGridDimension == 64:
##            id_ref = 669229
##        if TopGridDimension == 512:
##            id_ref = 279696579
##        idPopIII = ad[("particle_index"     )][ad[("particle_index")] == id_ref]
##        cPopIII  = ad[("creation_time"      )][ad[("particle_index")] == id_ref].in_units("Myr")
##        dPopIII  = ad[("dynamical_time"     )][ad[("particle_index")] == id_ref].in_units("Myr")
##        tPopIII  = ad[("age"                )][ad[("particle_index")] == id_ref].in_units("Myr")
##        MPopIII  = ad[("particle_mass"      )][ad[("particle_index")] == id_ref].in_units("Msun")
##        xPopIII  = ad[("particle_position_x")][ad[("particle_index")] == id_ref].in_units("code_length")
##        yPopIII  = ad[("particle_position_y")][ad[("particle_index")] == id_ref].in_units("code_length")
##        zPopIII  = ad[("particle_position_z")][ad[("particle_index")] == id_ref].in_units("code_length")
##        nPopIII = cPopIII.size
          '''
          idPopIII = [279696579]
          nPopIII = 1
          cPopIII = ds.arr([130.02987], "Myr")
          dPopIII = ds.arr([17.06192], "Myr")
          tPopIII = ds.arr([16.92085], "Myr")
          MPopIII = ds.arr([10.39301], "Msun")
          xPopIII = ds.arr([0.52883], "code_length")
          yPopIII = ds.arr([0.49892], "code_length")
          zPopIII = ds.arr([0.52456], "code_length")
          nBH = 0
          nPopII = 0
          '''

    if StarParticleCreation:
        if yt.is_root():
            print("PopIII")
            for iPopIII in range(nPopIII):
              print(" #%5d %13d  %13.5f %13.5f %13.5f  %13.5f  %13.5f  %13.5f  %13.5f" % (
                iPopIII, idPopIII[iPopIII]
              , xPopIII[iPopIII], yPopIII[iPopIII], zPopIII[iPopIII]
              , MPopIII[iPopIII]
              , cPopIII[iPopIII]
              , tPopIII[iPopIII]
              , dPopIII[iPopIII]))
            print("BH    ")
            for iBH     in range(nBH    ):
              print(" #%5d %13d  %13.5f %13.5f %13.5f  %13.5f  %13.5f  %13.5f  %13.5f" % (
                iBH    , idBH    [iBH    ]
              , xBH    [iBH    ], yBH    [iBH    ], zBH    [iBH    ]
              , MBH    [iBH    ]
              , cBH    [iBH    ]
              , tBH    [iBH    ]
              , dBH    [iBH    ]))
            print("PopII ")
            for iPopII  in range(nPopII ):
              print(" #%5d %13d  %13.5f %13.5f %13.5f  %13.5f  %13.5f  %13.5f  %13.5f" % (
                iPopII , idPopII [iPopII ]
              , xPopII [iPopII ], yPopII [iPopII ], zPopII [iPopII ]
              , MPopII [iPopII ]
              , cPopII [iPopII ]
              , tPopII [iPopII ]
              , dPopII [iPopII ]))
            print(datetime.datetime.now(), flush=True)

        if MultiSpecies > 1:
            # index of most massive Pop III star
            iPopIII = np.argmax(MPopIII)
#           print([xPopIII[iPopIII], yPopIII[iPopIII], zPopIII[iPopIII]])
#           print(posi_nHmax)

            def _H2I_kdiss_thin(field, data):
                if RadiativeTransferUseH2Shielding:
                    sigma_H2 = ds.arr(3.71e-18, 'cm**2')
                    H2I_kdiss_thin = np.zeros(data["x"].shape) 
                    for iPopIII in range(nPopIII):
                        dx = data["x"] - xPopIII[iPopIII]
                        dy = data["y"] - yPopIII[iPopIII]
                        dz = data["z"] - zPopIII[iPopIII]
                        dr2 = dx*dx + dy*dy + dz*dz
                        Q = ds.arr(Q_LW(MPopIII[iPopIII]), '1/s')
                        H2I_kdiss_thin += Q * sigma_H2 / (4.0 * pi * dr2)
                    return H2I_kdiss_thin
                else:
                    return data["H2I_kdiss"]
            ds.add_field(("gas", "H2I_kdiss_thin"), function=_H2I_kdiss_thin, sampling_type="cell", units="1/code_time")

            def _H2I_self_shielding_length1(field, data):
                return data["density"] / data["density_gradient_magnitude"]
            ds.add_field(("gas", "H2I_self_shielding_length1"), function=_H2I_self_shielding_length1, sampling_type="cell", units="code_length")
            def _H2I_column_density_corr1(field, data):
                return data["H2I_Density"] / (2.0*mp) * data["H2I_self_shielding_length1"]
            ds.add_field(("gas", "H2I_column_density_corr1"), function=_H2I_column_density_corr1, sampling_type="cell", units="1/code_length**2")
            def _H2I_kdiss_corr1(field, data):
                f_shield = f_shield_H2I_WH11(data["H2I_column_density_corr1"], data["temperature_corr"])
                kdiss_corr = f_shield * data["H2I_kdiss_thin"]
                kdiss_corr_min = ds.arr(1.0e-20, "1/code_time")
                kdiss_corr[kdiss_corr < kdiss_corr_min] = kdiss_corr_min
                return kdiss_corr
            ds.add_field(("gas", "H2I_kdiss_corr1"), function=_H2I_kdiss_corr1, sampling_type="cell", units="1/code_time")
 
            def _H2I_self_shielding_length3(field, data):
                return data["sound_speed_corr"] * np.sqrt( pi / (GravConst * data["density"]) )
            ds.add_field(("gas", "H2I_self_shielding_length3"), function=_H2I_self_shielding_length3, sampling_type="cell", units="code_length")
            def _H2I_column_density_corr3(field, data):
                return data["H2I_Density"] / (2.0*mp) * data["H2I_self_shielding_length3"]
            ds.add_field(("gas", "H2I_column_density_corr3"), function=_H2I_column_density_corr3, sampling_type="cell", units="1/code_length**2")
            def _H2I_kdiss_corr3(field, data):
                f_shield = f_shield_H2I_WH11(data["H2I_column_density_corr3"], data["temperature_corr"])
                kdiss_corr = f_shield * data["H2I_kdiss_thin"]
                kdiss_corr_min = ds.arr(1.0e-20, "1/code_time")
                kdiss_corr[kdiss_corr < kdiss_corr_min] = kdiss_corr_min
                return kdiss_corr
            ds.add_field(("gas", "H2I_kdiss_corr3"), function=_H2I_kdiss_corr3, sampling_type="cell", units="1/code_time")

        if MultiSpecies > 2:
            # index of most massive Pop III star
            iPopIII = np.argmax(MPopIII)
#           print([xPopIII[iPopIII], yPopIII[iPopIII], zPopIII[iPopIII]])
#           print(posi_nHmax)

            def _HDI_kdiss_thin(field, data):
                if RadiativeTransferUseH2Shielding:
                    dx = data["x"] - xPopIII[iPopIII]
                    dy = data["y"] - yPopIII[iPopIII]
                    dz = data["z"] - zPopIII[iPopIII]
                    dr2 = dx*dx + dy*dy + dz*dz
                    dist0 = ds.arr(1, "pc")
                    HDI_kdiss_0 = ds.arr(1.9e-8, "1/s")
                    return HDI_kdiss_0 * dist0**2 / dr2
                else:
                    return data["HDI_kdiss"]
            ds.add_field(("gas", "HDI_kdiss_thin"), function=_HDI_kdiss_thin, sampling_type="cell", units="1/code_time")

            def _HDI_self_shielding_length1(field, data):
                return data["density"] / data["density_gradient_magnitude"]
            ds.add_field(("gas", "HDI_self_shielding_length1"), function=_HDI_self_shielding_length1, sampling_type="cell", units="code_length")
            def _HDI_column_density_corr1(field, data):
                return data["HDI_Density"] / (2.0*mp) * data["HDI_self_shielding_length1"]
            ds.add_field(("gas", "HDI_column_density_corr1"), function=_HDI_column_density_corr1, sampling_type="cell", units="1/code_length**2")
            def _HDI_kdiss_corr1(field, data):
                f_shield = f_shield_H2I_WH11(data["HDI_column_density_corr1"], data["temperature_corr"])
                kdiss_corr = f_shield * data["HDI_kdiss_thin"]
                kdiss_corr_min = ds.arr(1.0e-20, "1/code_time")
                kdiss_corr[kdiss_corr < kdiss_corr_min] = kdiss_corr_min
                return kdiss_corr
            ds.add_field(("gas", "HDI_kdiss_corr1"), function=_HDI_kdiss_corr1, sampling_type="cell", units="1/code_time")
 
            def _HDI_self_shielding_length3(field, data):
                return data["sound_speed_corr"] * np.sqrt( pi / (GravConst * data["density"]) )
            ds.add_field(("gas", "HDI_self_shielding_length3"), function=_HDI_self_shielding_length3, sampling_type="cell", units="code_length")
            def _HDI_column_density_corr3(field, data):
                return data["HDI_Density"] / (2.0*mp) * data["HDI_self_shielding_length3"]
            ds.add_field(("gas", "HDI_column_density_corr3"), function=_HDI_column_density_corr3, sampling_type="cell", units="1/code_length**2")
            def _HDI_kdiss_corr3(field, data):
                f_shield = f_shield_H2I_WH11(data["HDI_column_density_corr3"], data["temperature_corr"])
                kdiss_corr = f_shield * data["HDI_kdiss_thin"]
                kdiss_corr_min = ds.arr(1.0e-20, "1/code_time")
                kdiss_corr[kdiss_corr < kdiss_corr_min] = kdiss_corr_min
                return kdiss_corr
            ds.add_field(("gas", "HDI_kdiss_corr3"), function=_HDI_kdiss_corr3, sampling_type="cell", units="1/code_time")


    if inumber < 25:
        cp = [0.50, 0.50, 0.50]
        if CosmologySimulation:
            if NestedLevel < 2:
                width = (1, 'Mpccm/h')
            else:
                width = (0.2, 'Mpccm/h')
            axes_unit = "Mpccm/h"
        else:
            width = (1, 'code_length')
            axes_unit = "pc"
    elif inumber < 25:
        cp = posi_nHmax
        width = 100.0 * l_J
    elif inumber <= 9999:
  #     for iPopIII in range(nPopIII):
  #         if idPopIII[iPopIII] == 669229:
        iPopIII = 0
        cp = [xPopIII[iPopIII], yPopIII[iPopIII], zPopIII[iPopIII]]
  #     for iBH in range(nBH):
  #         if idBH[iBH] == 669229:
  #             cp = [xBH[iBH], yBH[iBH], zBH[iBH]]
  #     for iPopII in range(nPopII):
  #         if idPopII[iPopII] == 669229:
  #             cp = [xPopII[iPopII], yPopII[iPopII], zPopII[iPopII]]
  #     width = (100, 'pc')
 #      cp = [xPopIII[0], yPopIII[0], zPopIII[0]]
        if TopGridDimension == 64:
          width = (1, 'kpc')
          axes_unit = "kpc"
        if TopGridDimension == 512:
          width = (10, 'pc')
          axes_unit = "pc"
    else:
  #     cp = [0.53, 0.49, 0.53]
  #     width = (0.2, 'Mpccm/h')
        cp = posi_nHmax
        width = 50.0 * l_J
 #      cp = [xPopIII[0], yPopIII[0], zPopIII[0]]
 #      width = (1, 'pc')

#   cp = [0.53, 0.49, 0.53]
#   width = (0.2, 'Mpccm/h')

#   CosmologySimulationGridDimension      =                [     48,              56,              40]
#   CosmologySimulationGridLeftEdge       =                [0.40625,         0.40625,          0.4375]                
#   CosmologySimulationGridRightEdge      =                [0.59375,           0.625,         0.59375]                
#   cp = [(CosmologySimulationGridRightEdge[0] + CosmologySimulationGridLeftEdge[0])*0.5
#       , (CosmologySimulationGridRightEdge[1] + CosmologySimulationGridLeftEdge[1])*0.5
#       , (CosmologySimulationGridRightEdge[2] + CosmologySimulationGridLeftEdge[2])*0.5]
#   width = ((CosmologySimulationGridRightEdge[1] - CosmologySimulationGridLeftEdge[1])*0.5, 'code_length')

#   for iPopIII in range(nPopIII):
#       if idPopIII[iPopIII] == 390151:
#           cp = [xPopIII[iPopIII], yPopIII[iPopIII], zPopIII[iPopIII]]
#   for iBH in range(nBH):
#       if idBH[iBH] == 390151:
#           cp = [xBH[iBH], yBH[iBH], zBH[iBH]]
#   width = (1, 'kpc')

    if yt.is_root():
        print(cp   )
        print(width)
        print(datetime.datetime.now(), flush=True)


    if PROJ or SLICE:
   #    if StarParticleCreation:
   #        if inumber > 39:
   #            for iPopIII in range(nPopIII):
   #                if idPopIII[iPopIII] == 390139:
   #                    cp = [xPopIII[iPopIII], yPopIII[iPopIII], zPopIII[iPopIII]]
   #            for iBH in range(nBH):
   #                if idBH[iBH] == 390139:
   #                    cp = [xBH[iBH], yBH[iBH], zBH[iBH]]
   #            width = (1, 'kpc')
        iPopIII0 = np.argmin(cPopIII)
        if PROJ:
            cp = [0.5, 0.5, 0.5]
            width = (0.2, 'Mpccm/h')
            axes_unit = "Mpccm/h"
        if SLICE:
            cp = [xPopIII[iPopIII0], yPopIII[iPopIII0], zPopIII[iPopIII0]]
            if str(Threshold) == '1stP3' or str(Threshold) == '1stP3m':
                width = (2.0, 'pc')
            if str(Threshold) == '1stP3h':
                width = (0.2, 'pc')
            axes_unit = "pc"

        norm = 'z'
        axis_x = [1.0, 0.0, 0.0]
        axis_y = [0.0, 1.0, 0.0]
        axis_z = [0.0, 0.0, 1.0]

        if CosmologySimulation:
            if PROJ:
                plot = yt.ProjectionPlot(ds, norm, ("deposit", "all_cic"), weight_field=("deposit", "all_cic")
                                       , center=cp, width=width, axes_unit=axes_unit)
                plot.annotate_text((0.05, 0.05), 'z = %.2f' % ds.current_redshift, coord_system='axis', text_args={'color':'white'})
                plot.set_unit(("deposit", "all_cic"), 'g/cm**3')
                plot.set_cmap(field=("deposit", "all_cic"), cmap="bds_highcontrast")
                if HALO:
                    for ihalo in range(halo_masses.size):
                        if halo_masses[ihalo] > ds.arr(1.0e5, 'Msun'):
                            if ihalo == ihalo_max:
                                circlecolor = "black"
                            else:
                                circlecolor = "white"
                            plot.annotate_sphere(
                              [halo_x[ihalo].in_units("code_length")
                             , halo_y[ihalo].in_units("code_length")
                             , halo_z[ihalo].in_units("code_length")]
                             , radius = halo_radii[ihalo]
                             , circle_args={'color':circlecolor})
#               plot.set_log(("deposit", "all_cic"), False)
                plot.save('Projection_%s_all_cic_all_cic_%04d.png' % (norm, inumber))
               
                plot = yt.ProjectionPlot(ds, norm, "grid_level", weight_field="ones", proj_style="mip"
                                       , center=cp, width=width, axes_unit=axes_unit)
                plot.annotate_text((0.05, 0.05), 'z = %.2f' % ds.current_redshift, coord_system='axis', text_args={'color':'white'})
                plot.set_cmap(field="grid_level", cmap="bds_highcontrast")
                plot.set_log("grid_level", False)
                plot.save('Projection_%s_grid_level_all_cic_%04d.png' % (norm, inumber))

        if MultiSpecies > 0:
            fields = [
                      "Hydrogen_number_density"
                    , "temperature_corr"
                    , "y_HI"
                ]
            if MultiSpecies > 1:
                fields.extend( [
                      "y_H2I"
                ] )
            if MultiSpecies > 2:
                fields.extend( [
                      "y_HDI"
                ] )
#           if MetalChemistry > 0:
#               fields.extend( [
#                     "Zmet"
#               ] )
#           if MultiMetals:
#               fields.extend( [
#                     "Zmet1"
#                   , "Zmet2"
#                   , "Zmet3"
#                   , "Zmet4"
#                   , "Zmet5"
#                   , "Zmet6"
#                   , "Zmet7"
#                   , "Zmet8"
#                   , "Zmet9"
#                   , "Zmet10"
#               ] )
            if StarParticleCreation:
                if MultiSpecies > 0:
                    fields.extend( [
                          "HI_kph"
                    ] )
                if MultiSpecies > 1:
                    fields.extend( [
                          "H2I_kdiss"
                        , "H2I_kdiss_thin"
                    ] )
                    if RadiativeTransferUseH2Shielding == 0 and H2_self_shielding > 0:
                      if H2_self_shielding == 1:
                        fields.extend( [
                          "H2I_self_shielding_length1"
                        , "H2I_kdiss_corr1"
                        ] )
                      if H2_self_shielding == 3:
                        fields.extend( [
                          "H2I_self_shielding_length3"
                        , "H2I_kdiss_corr3"
                        ] )
                if MultiSpecies > 2:
                    fields.extend( [
                          "HDI_kdiss"
                        , "HDI_kdiss_thin"
                    ] )
                    if RadiativeTransferUseH2Shielding == 0 and H2_self_shielding > 0:
                      if H2_self_shielding == 1:
                        fields.extend( [
                          "HDI_self_shielding_length1"
                        , "HDI_kdiss_corr1"
                        ] )
                      if H2_self_shielding == 3:
                        fields.extend( [
                          "HDI_self_shielding_length3"
                        , "HDI_kdiss_corr3"
                        ] )

            if PROJ:
                frb = yt.off_axis_projection(ds, cp, axis_z, [width, width, width], 800, fields[0], 'density', north_vector=axis_y)
            if SLICE:
                plot = yt.OffAxisSlicePlot(ds, axis_z, fields[0], center=cp, width=width, axes_unit=axes_unit, north_vector=axis_y)
                frb = plot.data_source.to_frb(width, 800)

            for ivar in range(len(fields)):
                if PROJ:
                    outfile = (outdir + 'Projection_%s_%s_density_%04d.dat' % (norm, fields[ivar], inumber))
                if SLICE:
                    outfile = (outdir + "Slice_%s_%s_%04d.dat"              % (norm, fields[ivar], inumber))
#               outfile = ('%s/%04d_%s.dat' % (pdir, inumber, field))
                outfp = open(outfile, 'wb')
 #              print(frb[field])
                outfp.write(frb[fields[ivar]])
                outfp.close()

            if StarParticleCreation:
                outfile = (outdir + "Stars_PopIII_%04d.dat" % (inumber))
                print(xPopIII[iPopIII].in_units(axes_unit))
                print(tPopIII)
                outfp = open(outfile, 'wb')
                outfp.write(struct.pack('i', nPopIII))
                for iPopIII in range(nPopIII):
                    outfp.write(struct.pack('d', xPopIII[iPopIII].in_units(axes_unit)))
                    outfp.write(struct.pack('d', yPopIII[iPopIII].in_units(axes_unit)))
                    outfp.write(struct.pack('d', zPopIII[iPopIII].in_units(axes_unit)))
                    outfp.write(struct.pack('d', cPopIII[iPopIII]))
                    outfp.write(struct.pack('d', dPopIII[iPopIII]))
                    outfp.write(struct.pack('d', tPopIII[iPopIII]))
                    outfp.write(struct.pack('d', MPopIII[iPopIII]))
                outfp.close()
#               plot.annotate_text((0.05, 0.05), 't = %.3f kyr' % tPopIII[iPopIII0], coord_system='axis', text_args={'color':'white'})
#               for iPopIII in range(nPopIII):
#                   plot.annotate_marker((xPopIII[iPopIII], yPopIII[iPopIII], zPopIII[iPopIII]), coord_system='data',
#                                marker='+',
#                                plot_args={'color':'cyan','s':100})
#               for iBH in range(nBH):
#                   plot.annotate_marker((xBH[iBH], yBH[iBH], zBH[iBH]), coord_system='data',
#                                marker='x',
#                                plot_args={'color':'cyan','s':100})
#               for iPopII in range(nPopII):
#                   plot.annotate_marker((xPopII[iPopII], yPopII[iPopII], zPopII[iPopII]), coord_system='data',
#                                marker='+',
#                                plot_args={'color':'pink','s':100})


        if PROJ:
          if MetalChemistry and MultiMetals:
            reso = 800
           
            prj = ds.proj('Hydrogen_number_density', norm, weight_field='density')
            frb_dens = prj.to_frb(width, center=cp, resolution=reso)
            prj = ds.proj('Zmet', 'z', weight_field='density')
            frb_Zmet = prj.to_frb(width, center=cp, resolution=reso)
            prj = ds.proj('Zmet5' , 'z', weight_field='density')
            frb_Zmet5 = prj.to_frb(width, center=cp, resolution=reso)
            prj = ds.proj('Zmet6' , 'z', weight_field='density')
            frb_Zmet6 = prj.to_frb(width, center=cp, resolution=reso)
            prj = ds.proj('Zmet7' , 'z', weight_field='density')
            frb_Zmet7 = prj.to_frb(width, center=cp, resolution=reso)
            prj = ds.proj('Zmet8' , 'z', weight_field='density')
            frb_Zmet8 = prj.to_frb(width, center=cp, resolution=reso)
            prj = ds.proj('Zmet9' , 'z', weight_field='density')
            frb_Zmet9 = prj.to_frb(width, center=cp, resolution=reso)
            prj = ds.proj('Zmet10', 'z', weight_field='density')
            frb_Zmet10= prj.to_frb(width, center=cp, resolution=reso)
           
            Y, X = np.mgrid[0:reso, 0:reso]
            X = X / float(reso) * width[0]
            Y = Y / float(reso) * width[0]
            X = X - 0.5 * width[0]
            Y = Y - 0.5 * width[0]
           
            Z  = np.log10(frb_dens  ['Hydrogen_number_density'])
#           Z  = np.log10(frb_Zmet  ['Zmet'  ])
            Z5 = np.log10(frb_Zmet5 ['Zmet5' ])
            Z6 = np.log10(frb_Zmet6 ['Zmet6' ])
            Z7 = np.log10(frb_Zmet7 ['Zmet7' ])
            Z8 = np.log10(frb_Zmet8 ['Zmet8' ])
            Z9 = np.log10(frb_Zmet9 ['Zmet9' ])
            Z10= np.log10(frb_Zmet10['Zmet10'])
           
            levels = np.arange(-6, 0, 1)
           
            fig, ax = plt.subplots()
            ax.set_aspect('equal', adjustable='box')

            ### Density projection ###
            pcolor = ax.pcolormesh(X, Y, Z, cmap='B-W LINEAR')

            ### Metallicity Contors ###
            cont = ax.contour(X, Y, Z5 , levels, colors='red')
            cont = ax.contour(X, Y, Z6 , levels, colors='orange')
            cont = ax.contour(X, Y, Z7 , levels, colors='yellow')
            cont = ax.contour(X, Y, Z8 , levels, colors='green')
            cont = ax.contour(X, Y, Z9 , levels, colors='blue')
            cont = ax.contour(X, Y, Z10, levels, colors='purple')
#           for iy in range(reso):
#               for ix in range(reso):
#                   print("%13.5e %13.5e %13.5e" % (X[ix,iy], Y[ix,iy], Z10[ix,iy]))
#               print("")

            ### Star positions ###
            if StarParticleCreation:
                if nPopIII:
                    xstar = ( xPopIII - cp[0] ).in_units(axes_unit)
                    ystar = ( yPopIII - cp[1] ).in_units(axes_unit)
                    pscatt = ax.scatter(xstar, ystar, s=100.0, linewidth=1, color='cyan', marker='+')
                if nBH:
                    xstar = ( xBH     - cp[0] ).in_units(axes_unit)
                    ystar = ( yBH     - cp[1] ).in_units(axes_unit)
                    pscatt = ax.scatter(xstar, ystar, s=100.0, linewidth=1, color='cyan', marker='x')
                if nPopII:
                    xstar = ( xPopII  - cp[0] ).in_units(axes_unit)
                    ystar = ( yPopII  - cp[1] ).in_units(axes_unit)
                    pscatt = ax.scatter(xstar, ystar, s=100.0, linewidth=1, color='pink', marker='+')

            fig.savefig('Projection_%s_Metallicity_Density_%04d.png' % (norm, inumber))


    if PARTICLE:
        for idir in range(1):
          if idir == 0: dir_x = 'particle_position_x'; dir_y = 'particle_position_y'
          if idir == 1: dir_x = 'particle_position_y'; dir_y = 'particle_position_z'
          if idir == 2: dir_x = 'particle_position_z'; dir_y = 'particle_position_x'

          plot = yt.ParticlePlot(ds, dir_x, dir_y, 'particle_mass', weight_field=None, axes_unit='Mpccm/h')
          plot.annotate_text((0.05, 0.05), 'z = %.2f' % ds.current_redshift, coord_system='axis', text_args={'color':'white'})
          plot.set_unit('particle_mass', 'Msun')
          plot.set_cmap(field='particle_mass', cmap="bds_highcontrast")
          if HALO:
              for ihalo in range(halo_masses.size):
                  if halo_masses[ihalo] > ds.arr(1.0e5, 'Msun'):
                      if ihalo == ihalo_max:
                          circlecolor = "black"
                      else:
                          circlecolor = "white"
                      plot.annotate_sphere(
                        [halo_x[ihalo].in_units("code_length")
                       , halo_y[ihalo].in_units("code_length")
                       , halo_z[ihalo].in_units("code_length")]
                       , radius = halo_radii[ihalo]
                       , circle_args={'color':circlecolor})
          plot.save('Particle_%s_%s_particle_mass_%04d.png' % (dir_x, dir_y, inumber))

#       plot = yt.ParticlePlot(ds, 'particle_position_x', 'particle_position_y', 'particle_mass', weight_field='particle_ones', axes_unit='Mpccm/h')
#       plot.annotate_text((0.05, 0.05), 'z = %.2f' % ds.current_redshift, coord_system='axis', text_args={'color':'black'})
#       plot.set_unit('particle_mass', 'Msun')
#       plot.set_cmap(field='particle_mass', cmap="bds_highcontrast")
#       plot.save('Particle_particle_position_x_particle_opsition_y_particle_mass_particle_ones_%04d.png' % inumber)


    if PHASE or PROF:
        reso = 128

        if inumber < 26:
            cp = posi_nHmax
            width = 50.0 * l_J
            cloud = ds.sphere(cp, (1, 'kpc')).cut_region(["obj['Hydrogen_number_density'] > 0.1"])
        else:
#           iPopIII0 = 0
            iPopIII0 = np.argmin(cPopIII)
            cp = [xPopIII[iPopIII0], yPopIII[iPopIII0], zPopIII[iPopIII0]]
            width = (1, 'Mpccm/h')
            cloud = ds.sphere(cp, width)

  #     plot = yt.PhasePlot(cloud, "radius", "Hydrogen_number_density", "H2I_kdiss", weight_field="cell_mass")
  #     plot.set_unit("radius", 'pc')
  #     plot.set_unit("H2I_kdiss", '1/s')
# #     plot.set_unit("cell_mass", 'Msun')
  #     plot.save("Profile-2d_radius_00Hydrogen_number_density_H2I_kdiss_%04d.png" % (inumber))

        if MultiSpecies > 0:
            fields = [
                "Hydrogen_number_density"
              , "temperature_corr"
              , "y_elec"
              , "y_HI"
              , "y_HII"
              , "y_HeI"
              , "y_HeII"
              , "y_HeIII"
            ]
        if MultiSpecies > 1:
            fields.extend( [
                "y_HM"
              , "y_H2I"
              , "y_H2II"
            ] )
        if MultiSpecies > 2:
            fields.extend( [
                "y_DI"
              , "y_DII"
              , "y_HDI"
            ] )
        if MultiSpecies > 3:
            fields.extend( [
                "y_DM"
              , "y_HDII"
              , "y_HeHII"
            ] )
        if StarParticleCreation:
            if MultiSpecies > 0:
                fields.extend( [
                      "HI_kph"
                    , "HeI_kph"
                    , "HeII_kph"
                ] )
            if MultiSpecies > 1:
                fields.extend( [
                      "H2I_kdiss"
                    , "H2I_kdiss_thin"
                    , "H2I_self_shielding_length1"
                    , "H2I_kdiss_corr1"
                    , "H2I_self_shielding_length3"
                    , "H2I_kdiss_corr3"
                ] )
        #   if MultiSpecies > 2:
        #       fields.extend( [
        #             "HDI_kdiss"
        #           , "HDI_kdiss_thin"
        #       ] )
        #       if RadiativeTransferUseH2Shielding == 0 and H2_self_shielding > 0:
        #         if H2_self_shielding == 1:
        #           fields.extend( [
        #             "HDI_self_shielding_length1"
        #           , "HDI_kdiss_corr1"
        #           ] )
        #         if H2_self_shielding == 3:
        #           fields.extend( [
        #             "HDI_self_shielding_length3"
        #           , "HDI_kdiss_corr3"
        #           ] )

        if PHASE:
            for ivar in range(len(fields)):
  #             plot = yt.PhasePlot(cloud, "Hydrogen_number_density", fields[ivar], "cell_mass", weight_field=None)
  #             plot.set_unit("cell_mass", 'Msun')
  #             plot.save("Profile-2d_Hydrogen_number_density_%02d%s_cell_mass_%04d.png" % (ivar, fields[ivar], inumber))
            
  #             plot = yt.PhasePlot(cloud, "radius", fields[ivar], "cell_mass", weight_field=None)
  #             plot.set_unit("radius", 'pc')
  #             plot.set_unit("cell_mass", 'Msun')
  #             plot.set_xlim(1e0, 1e3)
  #             plot.save("Profile-2d_radius_%02d%s_cell_mass_%04d.png" % (ivar, fields[ivar], inumber))

                if ivar == 0 or ivar == len(fields) - 1:
                    if ivar == 0:
                        fn_nT = outdir + 'nT_%04d.dat' % (inumber)
                    if ivar == len(fields) - 1:
                        fn_nT = outdir + 'nG_%04d.dat' % (inumber)
                    fp_nT = open(fn_nT, mode='w')
                    plot = yt.create_profile(cloud, ["Hydrogen_number_density", fields[ivar]], "cell_mass", weight_field=None
                                           , n_bins=(reso, reso))
                    for iy in range(reso):
                        for ix in range(reso):
                            fp_nT.write("%13.5e %13.5e %13.5e\n" % (
                                plot.x[ix]
                              , plot.y[iy]
                              , plot["cell_mass"][ix, iy].in_units("Msun")
                            ) )
                        fp_nT.write("\n")
                    fp_nT.close()

        if PROF:
            # create profile
            plot = yt.create_profile(cloud, "radius", fields, weight_field="cell_mass"
                         , extrema=None, n_bins=reso)

            fn_prof = outdir + 'prof_%04d.dat' % (inumber)
            fp_prof = open(fn_prof, mode='w')
            for ix in range(reso):
                fp_prof.write("%13.5e " % (plot.x[ix].in_units('Mpccm/h')) )
                for ivar in range(len(fields)):
                    fp_prof.write("%13.5e " % (plot[fields[ivar]][ix]) )
                fp_prof.write("\n")
            fp_prof.close()
            

    if RAY:
        # index of most massive Pop III star
#       iPopIII = np.argmax(MPopIII)
        iPopIII0 = np.argmin(cPopIII)
#       print([xPopIII[iPopIII], yPopIII[iPopIII], zPopIII[iPopIII]])
#       print(posi_nHmax)

        # create arrays
        ray = ds.ray([xPopIII[iPopIII0], yPopIII[iPopIII0], zPopIII[iPopIII0]]
                   , posi_nHmax)
        ray_rad   = ray["radius"].in_units("pc")
        ray_nH    = ray["Hydrogen_number_density"]
        ray_Tg    = ray["temperature_corr"]

        ray_yHI   = ray["y_HI"]
        ray_yelec = ray["y_elec"]
        ray_yH2I  = ray["y_H2I"]
        ray_yHM   = ray["y_HM"]
        ray_yHDI  = ray["y_HDI"]

        ray_kHI   = ray["HI_kph"].in_units("1/s")

        ray_kH2I  = ray["H2I_kdiss"].in_units("1/s")
        ray_kH2It = ray["H2I_kdiss_thin"].in_units("1/s")
        ray_sH2I1 = ray["H2I_self_shielding_length1"].in_units("pc")
        ray_NH2I1 = ray["H2I_column_density_corr1"].in_units("1/cm**2")
        ray_kH2I1 = ray["H2I_kdiss_corr1"].in_units("1/s")
        ray_sH2I3 = ray["H2I_self_shielding_length3"].in_units("pc")
        ray_NH2I3 = ray["H2I_column_density_corr3"].in_units("1/cm**2")
        ray_kH2I3 = ray["H2I_kdiss_corr3"].in_units("1/s")

        ray_kHDI  = ray["HDI_kdiss"].in_units("1/s")
        ray_kHDIt = ray["HDI_kdiss_thin"].in_units("1/s")
        ray_sHDI1 = ray["HDI_self_shielding_length1"].in_units("pc")
        ray_NHDI1 = ray["HDI_column_density_corr1"].in_units("1/cm**2")
        ray_kHDI1 = ray["HDI_kdiss_corr1"].in_units("1/s")
        ray_sHDI3 = ray["HDI_self_shielding_length3"].in_units("pc")
        ray_NHDI3 = ray["HDI_column_density_corr3"].in_units("1/cm**2")
        ray_kHDI3 = ray["HDI_kdiss_corr3"].in_units("1/s")

        # calculate column density
        iradii = np.argsort(ray_rad)
        ray_nH2I  = ray_yH2I * ray_nH
        ray_NH2I  = ds.arr(np.zeros(ray_rad.size), "1/cm**2")
        ray_nHDI  = ray_yHDI * ray_nH
        ray_NHDI  = ds.arr(np.zeros(ray_rad.size), "1/cm**2")
        irad_pre = -1
        for irad in iradii:
            if irad_pre == -1:
                rad0 = 0.0
            else:
                rad0 = ray_rad[irad_pre]
            ray_NH2I[irad] = ray_NH2I[irad_pre] + ray_nH2I[irad] * (ray_rad[irad] - rad0)
            ray_NHDI[irad] = ray_NHDI[irad_pre] + ray_nHDI[irad] * (ray_rad[irad] - rad0)
            irad_pre = irad

        ray_sH2I = (ray_NH2I / ray_nH2I).in_units("pc")
        ray_kH2I0 = ray_kH2It * f_shield_H2I_WH11(ray_NH2I, ray_Tg)

        ray_sHDI = (ray_NHDI / ray_nHDI).in_units("pc")
        ray_kHDI0 = ray_kHDIt * f_shield_H2I_WH11(ray_NHDI, ray_Tg)

        # output star data
        fn_str = outdir + 'Profile-1d_%04d.log' % (inumber)
        fp_str = open(fn_str, mode='w')
        fp_str.write("%23.15e %23.15e\n" % (
            ds.current_redshift
          , ds.current_time
        ))
        fp_str.write("%23.15e %23.15e %23.15e\n" % (
            xPopIII[iPopIII].in_units("pc")
          , yPopIII[iPopIII].in_units("pc")
          , zPopIII[iPopIII].in_units("pc")
        ))
        fp_str.write("%23.15e %23.15e %23.15e\n" % (
            posi_nHmax[0].in_units("pc")
          , posi_nHmax[1].in_units("pc")
          , posi_nHmax[2].in_units("pc")
        ))
        fp_str.close()


        # output profile data
        fn_ray = outdir + 'Profile-1d_%04d.dat' % (inumber)
        fp_ray = open(fn_ray, mode='w')
        for irad in iradii:
            fp_ray.write("%13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e " % (
                ray_rad  [irad]
              , ray_nH   [irad]
              , ray_Tg   [irad]
              , ray_yHI  [irad]
              , ray_yelec[irad]
              , ray_yH2I [irad]
              , ray_yHM  [irad]
              , ray_kHI  [irad]
            ))
            fp_ray.write("%13.5e %13.5e " % (
                ray_kH2It[irad]
              , ray_kH2I0[irad]
            ))
            fp_ray.write("%13.5e %13.5e %13.5e " % (
                ray_sH2I [irad]
              , ray_NH2I [irad]
              , ray_kH2I [irad]
            ))
            fp_ray.write("%13.5e %13.5e %13.5e " % (
                ray_sH2I1[irad]
              , ray_NH2I1[irad]
              , ray_kH2I1[irad]
            ))
            fp_ray.write("%13.5e %13.5e %13.5e " % (
                ray_sH2I3[irad]
              , ray_NH2I3[irad]
              , ray_kH2I3[irad]
            ))
            fp_ray.write("%13.5e %13.5e " % (
                ray_kHDIt[irad]
              , ray_kHDI0[irad]
            ))
            fp_ray.write("%13.5e %13.5e %13.5e " % (
                ray_sHDI [irad]
              , ray_NHDI [irad]
              , ray_kHDI [irad]
            ))
            fp_ray.write("%13.5e %13.5e %13.5e " % (
                ray_sHDI1[irad]
              , ray_NHDI1[irad]
              , ray_kHDI1[irad]
            ))
            fp_ray.write("%13.5e %13.5e %13.5e " % (
                ray_sHDI3[irad]
              , ray_NHDI3[irad]
              , ray_kHDI3[irad]
            ))
            fp_ray.write("\n")
        fp_ray.close()

        '''
        # output png files
        plt.loglog(ray_rad[iradii[:]], ray_nH  [iradii[:]], label=r'$n_{\rm H}$')
        plt.loglog(ray_rad[iradii[:]], ray_nH2I[iradii[:]], label=r'$n ({\rm H}_2)$')
        plt.loglog(ray_rad[iradii[:]], ray_nHDI[iradii[:]], label=r'$n ({\rm HD})$')
        plt.xlabel(r"Distance from Pop III star [pc]")
        plt.ylabel(r"Density [cm$^{-3}$]")
        plt.legend()
        plt.savefig("Profile-1d_radius_nH_%04d.png" % (inumber))
        plt.close('all')

        # H2
        plt.loglog(ray_rad[iradii[:]], ray_NH2I [iradii[:]], label=r'Radiative Transfer')
        if RadiativeTransferUseH2Shielding == 0 and H2_self_shielding > 0:
            if H2_self_shielding == 1:
                xlabel = r'$n ({\rm H}_2) (\rho / \nabla \rho)$'
                plt.loglog(ray_rad[iradii[:]], ray_NH2I1[iradii[:]], label=xlabel)
            if H2_self_shielding == 3:
                xlabel = r'$n ({\rm H}_2) l_{\rm J}$'
                plt.loglog(ray_rad[iradii[:]], ray_NH2I3[iradii[:]], label=xlabel)
        plt.xlabel(r"Distance from Pop III star [pc]")
        plt.ylabel(r"H$_2$ column density [cm$^{-2}$]")
        plt.legend()
        plt.savefig("Profile-1d_radius_NH2I_%04d.png" % (inumber))
        plt.close('all')

        plt.loglog(ray_rad[iradii[:]], ray_kH2I [iradii[:]], label='Radiative Transfer')
        if RadiativeTransferUseH2Shielding == 1:
            plt.loglog(ray_rad[iradii[:]], ray_kH2I0[iradii[:]], label='post-process')
        if RadiativeTransferUseH2Shielding == 0 and H2_self_shielding > 0:
            if H2_self_shielding == 1:
                plt.loglog(ray_rad[iradii[:]], ray_kH2I1[iradii[:]], label='correction')
            if H2_self_shielding == 3:
                plt.loglog(ray_rad[iradii[:]], ray_kH2I3[iradii[:]], label='correction')
        plt.xlabel(r"Distance from Pop III star [pc]")
        plt.ylabel(r"H$_2$ photodissociation rate [s$^{-1}$]")
        plt.legend()
        plt.savefig("Profile-1d_radius_kH2I_%04d.png" % (inumber))
        plt.close('all')

        # HD
        plt.loglog(ray_rad[iradii[:]], ray_NHDI [iradii[:]], label=r'Radiative Transfer')
        if RadiativeTransferUseH2Shielding == 0 and H2_self_shielding > 0:
            if H2_self_shielding == 1:
                xlabel = r'$n ({\rm HD}) (\rho / \nabla \rho)$'
                plt.loglog(ray_rad[iradii[:]], ray_NHDI1[iradii[:]], label=xlabel)
            if H2_self_shielding == 3:
                xlabel = r'$n ({\rm HD}) l_{\rm J}$'
                plt.loglog(ray_rad[iradii[:]], ray_NHDI3[iradii[:]], label=xlabel)
        plt.xlabel(r"Distance from Pop III star [pc]")
        plt.ylabel(r"HD column density [cm$^{-2}$]")
        plt.legend()
        plt.savefig("Profile-1d_radius_NHDI_%04d.png" % (inumber))
        plt.close('all')

        plt.loglog(ray_rad[iradii[:]], ray_kHDI [iradii[:]], label='Radiative Transfer')
        if RadiativeTransferUseH2Shielding == 1:
            plt.loglog(ray_rad[iradii[:]], ray_kHDI0[iradii[:]], label='Radiative Transfer')
        if RadiativeTransferUseH2Shielding == 0 and H2_self_shielding > 0:
            if H2_self_shielding == 1:
                plt.loglog(ray_rad[iradii[:]], ray_kHDI1[iradii[:]], label='correction')
            if H2_self_shielding == 3:
                plt.loglog(ray_rad[iradii[:]], ray_kHDI3[iradii[:]], label='correction')
        plt.xlabel(r"Distance from Pop III star [pc]")
        plt.ylabel(r"HD photodissociation rate [s$^{-1}$]")
        plt.legend()
        plt.savefig("Profile-1d_radius_kHDI_%04d.png" % (inumber))
        plt.close('all')
        '''


    if TRACE_ABUNDANCE:

        ad_dens   = ad.mean("Density"         , weight="cell_volume").in_units("g/cm**3")
        ad_temp   = ad.mean("temperature"     , weight="cell_mass")  .in_units("K")
        ad_metal  = ad.mean("Metal_Density"   , weight="cell_volume").in_units("g/cm**3")
##                + ad.mean("SN_Colour"       , weight="cell_volume").in_units("g/cm**3"))
        if MultiSpecies > 0:
            ad_elec   = ad.mean("Electron_Density", weight="cell_volume").in_units("g/cm**3")
            ad_HI     = ad.mean(      "HI_Density", weight="cell_volume").in_units("g/cm**3")
            ad_HII    = ad.mean(     "HII_Density", weight="cell_volume").in_units("g/cm**3")
            ad_HeI    = ad.mean(     "HeI_Density", weight="cell_volume").in_units("g/cm**3")
            ad_HeII   = ad.mean(    "HeII_Density", weight="cell_volume").in_units("g/cm**3")
            ad_HeIII  = ad.mean(   "HeIII_Density", weight="cell_volume").in_units("g/cm**3")
        if MultiSpecies > 1:
            ad_HM     = ad.mean(      "HM_Density", weight="cell_volume").in_units("g/cm**3")
            ad_H2I    = ad.mean(     "H2I_Density", weight="cell_volume").in_units("g/cm**3")
            ad_H2II   = ad.mean(    "H2II_Density", weight="cell_volume").in_units("g/cm**3")
        if MultiSpecies > 2:
            ad_DI     = ad.mean(      "DI_Density", weight="cell_volume").in_units("g/cm**3")
            ad_DII    = ad.mean(     "DII_Density", weight="cell_volume").in_units("g/cm**3")
            ad_HDI    = ad.mean(     "HDI_Density", weight="cell_volume").in_units("g/cm**3")
        if MultiSpecies > 3:
            ad_DM     = ad.mean(      "DM_Density", weight="cell_volume").in_units("g/cm**3")
            ad_HDII   = ad.mean(    "HDII_Density", weight="cell_volume").in_units("g/cm**3")
            ad_HeHII  = ad.mean(   "HeHII_Density", weight="cell_volume").in_units("g/cm**3")
        if MetalChemistry > 0:
            ad_CII    = ad.mean(     "CII_Density", weight="cell_volume").in_units("g/cm**3")
            ad_CI     = ad.mean(      "CI_Density", weight="cell_volume").in_units("g/cm**3")
            ad_CO     = ad.mean(      "CO_Density", weight="cell_volume").in_units("g/cm**3")
            ad_CO2    = ad.mean(     "CO2_Density", weight="cell_volume").in_units("g/cm**3")
            ad_CH     = ad.mean(      "CH_Density", weight="cell_volume").in_units("g/cm**3")
            ad_CH2    = ad.mean(     "CH2_Density", weight="cell_volume").in_units("g/cm**3")
            ad_COII   = ad.mean(    "COII_Density", weight="cell_volume").in_units("g/cm**3")
            ad_OI     = ad.mean(      "OI_Density", weight="cell_volume").in_units("g/cm**3")
            ad_OH     = ad.mean(      "OH_Density", weight="cell_volume").in_units("g/cm**3")
            ad_H2O    = ad.mean(     "H2O_Density", weight="cell_volume").in_units("g/cm**3")
            ad_O2     = ad.mean(      "O2_Density", weight="cell_volume").in_units("g/cm**3")
            ad_OII    = ad.mean(     "OII_Density", weight="cell_volume").in_units("g/cm**3")
            ad_OHII   = ad.mean(    "OHII_Density", weight="cell_volume").in_units("g/cm**3")
            ad_H2OII  = ad.mean(   "H2OII_Density", weight="cell_volume").in_units("g/cm**3")
            ad_H3OII  = ad.mean(   "H3OII_Density", weight="cell_volume").in_units("g/cm**3")
            ad_O2II   = ad.mean(    "O2II_Density", weight="cell_volume").in_units("g/cm**3")
            ad_SiI    = ad.mean(     "SiI_Density", weight="cell_volume").in_units("g/cm**3")
            ad_SiOI   = ad.mean(    "SiOI_Density", weight="cell_volume").in_units("g/cm**3")
            ad_SiO2I  = ad.mean(   "SiO2I_Density", weight="cell_volume").in_units("g/cm**3")
            ad_Mg     = ad.mean(      "Mg_Density", weight="cell_volume").in_units("g/cm**3")
            ad_Al     = ad.mean(      "Al_Density", weight="cell_volume").in_units("g/cm**3")
            ad_S      = ad.mean(       "S_Density", weight="cell_volume").in_units("g/cm**3")
            ad_Fe     = ad.mean(      "Fe_Density", weight="cell_volume").in_units("g/cm**3")
        if GrainGrowth > 0:
            ad_SiM    = ad.mean(     "SiM_Density", weight="cell_volume").in_units("g/cm**3")
            ad_FeM    = ad.mean(     "FeM_Density", weight="cell_volume").in_units("g/cm**3")
            ad_Mg2SiO4= ad.mean( "Mg2SiO4_Density", weight="cell_volume").in_units("g/cm**3")
            ad_MgSiO3 = ad.mean(  "MgSiO3_Density", weight="cell_volume").in_units("g/cm**3")
            ad_Fe3O4  = ad.mean(   "Fe3O4_Density", weight="cell_volume").in_units("g/cm**3")
            ad_AC     = ad.mean(      "AC_Density", weight="cell_volume").in_units("g/cm**3")
            ad_SiO2D  = ad.mean(   "SiO2D_Density", weight="cell_volume").in_units("g/cm**3")
            ad_MgO    = ad.mean(     "MgO_Density", weight="cell_volume").in_units("g/cm**3")
            ad_FeS    = ad.mean(     "FeS_Density", weight="cell_volume").in_units("g/cm**3")
            ad_Al2O3  = ad.mean(   "Al2O3_Density", weight="cell_volume").in_units("g/cm**3")


        fp_trace_ab.write("%23.15e " % ds.current_redshift)
        fp_trace_ab.write("%13.5e %13.5e %13.5e "                   % ( ad_dens   , ad_temp   , ad_metal                         ) )
        if MultiSpecies > 0:
            fp_trace_ab.write("%13.5e %13.5e %13.5e "               % ( ad_elec   , ad_HI     , ad_HII                           ) )
            fp_trace_ab.write("%13.5e %13.5e %13.5e "               % ( ad_HeI    , ad_HeII   , ad_HeIII                         ) )
        if MultiSpecies > 1:
            fp_trace_ab.write("%13.5e %13.5e %13.5e "               % ( ad_HM     , ad_H2I    , ad_H2II                          ) )
        if MultiSpecies > 2:
            fp_trace_ab.write("%13.5e %13.5e %13.5e "               % ( ad_DI     , ad_DII    , ad_HDI                           ) )
        if MultiSpecies > 3:
            fp_trace_ab.write("%13.5e %13.5e %13.5e "               % ( ad_DM     , ad_HDII   , ad_HeHII                         ) )
        if MetalChemistry > 0:
            fp_trace_ab.write("%13.5e %13.5e %13.5e %13.5e "        % ( ad_CI     , ad_CII    , ad_CO     , ad_CO2               ) )
            fp_trace_ab.write("%13.5e %13.5e %13.5e %13.5e "        % ( ad_OI     , ad_OH     , ad_H2O    , ad_O2                ) )
            fp_trace_ab.write("%13.5e %13.5e %13.5e "               % ( ad_SiI    , ad_SiOI   , ad_SiO2I                         ) )
            fp_trace_ab.write("%13.5e %13.5e %13.5e "               % ( ad_CH     , ad_CH2    , ad_COII                          ) )
            fp_trace_ab.write("%13.5e %13.5e %13.5e %13.5e %13.5e " % ( ad_OII    , ad_OHII   , ad_H2OII  , ad_H3OII  , ad_O2II  ) )
        if GrainGrowth > 0:
            fp_trace_ab.write("%13.5e %13.5e %13.5e %13.5e "        % ( ad_Mg     , ad_Al     , ad_S      , ad_Fe                ) )
            fp_trace_ab.write("%13.5e %13.5e %13.5e %13.5e %13.5e " % ( ad_SiM    , ad_FeM    , ad_Mg2SiO4, ad_MgSiO3 , ad_Fe3O4 ) )
            fp_trace_ab.write("%13.5e %13.5e %13.5e %13.5e %13.5e " % ( ad_AC     , ad_SiO2D  , ad_MgO    , ad_FeS    , ad_Al2O3 ) )
        fp_trace_ab.write("\n")                                          


if TRACE_MASS:
    fp_trace_mass.close()
if TRACE_ABUNDANCE:
    fp_trace_ab.close()
