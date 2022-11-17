from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate as inte
# from hoki import load
# from constants import cm_pc, g_Msun, L_sun_conversion, h, c, G, sigma_SB, p_mass, pion_lifetime
# from astropy import constants as con
from astropy import units as u

# cm per pc
cm_pc = 3.086*10**18

# g per solar mass
g_Msun = 1.988*10**33

# (ergs/s) / L_Sun
L_sun_conversion = 3.826*10**33

h = 6.6261*(10**(-27))

c = 2.99792458*(10**10)

G = 6.67259*(10**-8)

sigmaSB = 5.67037441918442945397*10**-5

protonMass = 1.672621911*10**-24

# Pion lifetime in years
pion_lifetime = 5*10**7

meanFreePath = 0.1 * u.pc
rShellInitial = 10 * u.pc

# 30 Dor
# Edot wind 2.2 × 1039 erg s−1
# vshell 25 km s−1 

###############################################################################
## Code for calculating the cosmic ray pressure on regions of a galaxy.
##
## Case: pascaleCase
## 
##
##
##
###############################################################################


### Define a class for models.
###############################################################################
class model:
    # All units are cgs unless specified
    def __init__(self, name, meanFreePath, gasColumnHeight, windToCREnergyFraction, coverageFraction):
        self.name = name
        self.meanFreePath = meanFreePath * cm_pc # * u.pc# mean free path in parsecs
        self.gasColumnHeight = gasColumnHeight * cm_pc # *u.pc # Initial column height in parsecs
        self.windToCREnergyFraction = windToCREnergyFraction # parameter for what fraction of wind energy is converted to CR energy
        self.coverageFraction = coverageFraction

    def __str__(self):
        return f"Model: {self.name}"

### Define a class for the region of interest.
### Maybe delete?
###############################################################################
# class currentRegion:
#     # All units are cgs unless specified
#     def __init__(self, galaxyData, regionID):
#         self.massGas = galaxyData["Mass_g"][galaxyData.regionID == regionID].values[0]
#         self.massNewStars = galaxyData["Mass_new"][galaxyData.regionID == regionID].values[0]
#         self.massOldStars = galaxyData["Mass_old"][galaxyData.regionID == regionID].values[0]
#         self.sigmaOldStars = galaxyData["Sigma_star"][galaxyData.regionID == regionID].values[0]
#         self.sigmaGas = galaxyData["Sigma_g"][galaxyData.regionID == regionID].values[0]
#         self.sigmaNewStars = galaxyData["Sigma_new"][galaxyData.regionID == regionID].values[0]
#         self.tauInitial = galaxyData["tau"][galaxyData.regionID == regionID].values[0]
#         self.bolometricLuminosity = galaxyData["L_Bol"][galaxyData.regionID == regionID].values[0]
#         self.heightOldStars = galaxyData["H_old"][galaxyData.regionID == regionID].values[0]
#         self.momentum = galaxyData["Momentum"][galaxyData.regionID == regionID].values[0]
#         self.bolometricFlux = galaxyData["F_Bol"][galaxyData.regionID == regionID].values[0]
#         self.eddingtonFlux = galaxyData["F_Edd"][galaxyData.regionID == regionID].values[0]
#         self.coverageFraction = 0.5
#         self.radius = galaxyData["Dist_to_center"][galaxyData.regionID == regionID].values[0]


class regionData:
    # All units are cgs unless specified
    def __init__(self, name, age, luminosity, energyDotWind, radius, massShell, massNewStars, gasDensity):
        self.name = name
        self.age = age # Age in Myr
        self.luminosity = luminosity * L_sun_conversion # Luminosity in solar luminosities
        self.energyDotWind = energyDotWind # Power in erg/s
        self.radius = radius * cm_pc # radius in pc
        self.massShell = massShell * g_Msun # Mass in the gas shell in solar masses
        self.massNewStars = massNewStars * g_Msun # Mass in the star cluster in solar masses
        self.gasDensity = gasDensity # Gas density outside the shell, in g/cm^3
        self.electronDensity = self.massShell / (4/3*np.pi*(self.radius)**3) / protonMass # electron density in n/cm^3
        self.pionTime = pion_lifetime / self.electronDensity
        
    def __str__(self):
        return f"Region: {self.name}"

### Define models
###############################################################################
fiducial = model("fiducial", meanFreePath = 0.01, gasColumnHeight = 10, windToCREnergyFraction = 0.1, coverageFraction = 1)

###############################################################################

### Define regions.
###############################################################################
testRegion = regionData("Test Region", age = 1, luminosity = 2.3*10**8, energyDotWind = 9.565*10**37, radius = 100, massShell = 10**4, massNewStars = 10**4, gasDensity = 0)


### Define timescale functions
###############################################################################

def getMinimumTime(rShell, vShell, meanFreePath, pionTime):
    tDiff = 3*rShell**2 / (c * meanFreePath)
    tAdv = rShell/vShell
    pionTime = np.inf
    
    return min(tDiff, tAdv, pionTime)

def ODE(rShell, X, currentRegion, currentModel, sweepUpMass = False):
    
    vShell, t = X
    
    dvdr =  currentModel.windToCREnergyFraction*currentRegion.energyDotWind*currentModel.coverageFraction / rShell /(currentRegion.massShell*vShell) * getMinimumTime(rShell, vShell, currentModel.meanFreePath, currentRegion.pionTime) - G*(currentRegion.massShell + currentRegion.massNewStars)/(vShell*rShell**2)
    
    if sweepUpMass:
        dvdr -= vShell*4*np.pi*currentRegion.gasDensity*(rShell - rShellInitial)**2 / currentRegion.massShell

    dtdr = 1/(vShell*31556952)
    
    return dvdr, dtdr

### Define quantity functions
###############################################################################

def getLGamma(Edot_cr, t_diff, t_pion):
    
    L_gamma = Edot_cr/3 * np.minimum(1,t_diff/t_pion)
    
    return L_gamma

def getPCR(Edot_cr, t, V):
    
    P_cr = Edot_cr * t / (3*V)
    
    return P_cr

### Fill out current model and region
###############################################################################
currentModel = fiducial
currentRegion = testRegion

v0 = 1
t0 = 10**6

sweepUpMass = False

rSpan = [currentModel.gasColumnHeight, 100*currentRegion.radius]
r = np.linspace(rSpan[0], rSpan[1], 2000)

print(str(currentModel))
print(str(currentRegion))
ODESolve = inte.solve_ivp(ODE, rSpan, [v0, t0], args = [currentRegion, currentModel, sweepUpMass], max_step = cm_pc, rtol = 1e20)


### Plots
###############################################################################

#### Plot ODE
#################################################

fig, ax = plt.subplots(dpi = 200)

plt.plot(ODESolve.t/cm_pc, ODESolve.y[0]/10**5)

plt.xscale('log')
plt.yscale('log')

plt.ylim(1)

# plt.text(0.01, 0.9, '{}'.format(str(currentModel)), transform = ax.transAxes)
plt.text(0.7, 0.17, r'$M_{\rm Star}\, 10^4\,M_\odot$', transform = ax.transAxes)
plt.text(0.7, 0.1, r'$M_{\rm shell}\, 10^4\,M_\odot$', transform = ax.transAxes)
plt.text(0.7, 0.03, r'$\dot{E}_{\rm CR}\, 2.5\times 10^3\, L_\odot$', transform = ax.transAxes)

plt.xlabel("Distance (pc)")
plt.ylabel("Velocity (km/s)")




