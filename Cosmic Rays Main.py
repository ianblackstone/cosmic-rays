# %%
from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate as inte
from scipy.optimize import fsolve
# from hoki import load
from astropy import constants as con
from astropy import units as u

# %%
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

# %%
# Set fiducial values
###############################################################################
meanFreePathFiducial = 0.01  # pc
gasColumnHeightFiducial = 10  # pc
windToCREnergyFractionFiducial = 0.1  # Fraction
coverageFractionFiducial = 1  # Fraction

# Turn on or off various pressures in the model.
energyInjectionFiducial = True
advectionPressureFiducial = True
diffusionPressureFiducial = True
pionPressureFiducial = True
streamPressureFiducial = True
sweepUpMassFiducial = False
###############################################################################

# 30 Dor
# Edot wind 2.2 × 1039 erg s−1
# vshell 25 km s−1

###############################################################################
# Code for calculating the cosmic ray pressure on regions of a galaxy.
##
## Case: pascaleCase
##
###############################################################################

# %%
# Define a class for models.
###############################################################################


class model:
    # All units read from a model are cgs unless specified. Inputs are in more convenient units.
    def __init__(self, name, meanFreePath=meanFreePathFiducial, gasColumnHeight=gasColumnHeightFiducial, windToCREnergyFraction=windToCREnergyFractionFiducial,
                 coverageFraction=coverageFractionFiducial, energyInjection=energyInjectionFiducial, advectionPressure=advectionPressureFiducial,
                 diffusionPressure=diffusionPressureFiducial, pionPressure=pionPressureFiducial, streamPressure=streamPressureFiducial, sweepUpMass=sweepUpMassFiducial):
        """_summary_

        Args:
            name (string): The name of the model
            meanFreePath (number, optional): mean free path. Input in parsecs, will be converted to cm and a unit label added. Defaults to meanFreePathFiducial.
            gasColumnHeight (number, optional): Initial column height in parsecs. Defaults to gasColumnHeightFiducial.
            windToCREnergyFraction (number, optional): Parameter for what fraction of wind energy is converted to CR energy. Defaults to windToCREnergyFractionFiducial.
            coverageFraction (number, 0 to 1, optional): Shell coverage fraction. Defaults to coverageFractionFiducial.
            energyInjection (boolean, optional): Boolean to enable energy injection in the model. Defaults to energyInjectionFiducial.
            advectionPressure (boolean, optional): Boolean to enable advection in the model. Defaults to advectionPressureFiducial.
            diffusionPressure (boolean, optional): Boolean to enable diffusion in the model. Defaults to diffusionPressureFiducial.
            pionPressure (boolean, optional): Boolean to enable pion decay in the model. Defaults to pionPressureFiducial.
            streamPressure (boolean, optional): Boolean to enable streaming in the model, not currently implemented. Defaults to streamPressureFiducial.
            sweepUpMass (boolean, optional): Boolean to enable the sweeping up of additional mass in the model. Defaults to sweepUpMassFiducial.
        """
        self.name = name
        self.meanFreePath = meanFreePath * cm_pc
        self.gasColumnHeight = gasColumnHeight * cm_pc
        self.windToCREnergyFraction = windToCREnergyFraction
        self.coverageFraction = coverageFraction
        self.energyInjection = energyInjection
        self.advectionPressure = advectionPressure
        self.diffusionPressure = diffusionPressure
        self.pionPressure = pionPressure
        self.streamPressure = streamPressure
        self.sweepUpMass = sweepUpMass

    def __str__(self):
        return f"Model: {self.name}"


class regionData:
    # All units are cgs unless specified
    def __init__(self, name, age, luminosity, energyDotWind, radius, radiusOldStars, massShell, massNewStars, massOldStars, gasDensity):
        """_summary_

        Args:
            name (string): The region name or reference.
            age (number): Region age. Input in Myr, unit will be assigned to it.
            luminosity (number): Stellar luminosity of the region. Input in solar luminosities, unit will be assigned to it.
            energyDotWind (number): The energy input to the region from stellar wind. Input in solar luminosities, unit will be assigned to it.
            radius (number): The radius of the region. Input in pc, unit will be assigned to it. Currently not used in favor of the model's gasColumnHeight.
            radiusOldStars (number): The radius of the old stellar population. Input in pc, unit will be assigned to it.
            massShell (number): The mass of the gas shell. Input in solar masses, unit will be assigned to it.
            massNewStars (number): The mass of new stars in the cluster. Input in solar masses, unit will be assigned to it.
            massOldStars (number): The mass of old stars in the cluster. Input in solar masses, unit will be assigned to it. This is not the enclosed old stellar mass, but the total.
            gasDensity (number): The density of cold gas outside the gas shell, which provides the material to be swept up. Input in ____. Not currently used.

        Additional parameters (automatically calculated):
            massTotal (number): The total region mass in grams.
            electronDensity (number): The electron number density in n/cm^3. Not currently used but important for pion losses.
            pionTime (number): The pion timescale in s. Not currently used.
        """
        self.name = name
        self.age = age
        self.luminosity = luminosity * L_sun_conversion
        self.energyDotWind = energyDotWind
        self.radius = radius * cm_pc
        self.radiusOldStars = radiusOldStars * cm_pc
        self.massShell = massShell * g_Msun
        self.massNewStars = massNewStars * g_Msun
        self.massOldStars = massOldStars * g_Msun
        self.massTotal = self.massNewStars + self.massShell
        self.gasDensity = gasDensity
        self.electronDensity = self.massShell / \
            (4/3*np.pi*(self.radius)**3) / \
            protonMass
        self.pionTime = pion_lifetime / self.electronDensity

    def __str__(self):
        return f"Region: {self.name}"


# Define models
###############################################################################
# Models are created using the model class, a model only requires a name, all other parameters are set using the fiducial values
fiducial = model("fiducial")

###############################################################################

# Define regions.
###############################################################################
testRegion = regionData("Test Region", age=1, luminosity=2.3*10**8, energyDotWind=9.565*10**37, radius=10,
                        radiusOldStars=10**4, massShell=10**4, massNewStars=10**4, massOldStars=10**4, gasDensity=0)


# %%
# Define timescale functions
###############################################################################
def getMinimumTime(rShell, vShell, model):
    """Function that returns the minimum timescale for cosmic ray momentum deposition

    Args:
        rShell (number): The radius of the shell in cm
        vShell (number): The velocity of the shell in cm/s
        model (model): The current model

    Returns:
        number: The minimum time scale.
    """
    tDiff = 3*rShell**2 / (c * model.meanFreePath)
    tAdv = rShell/vShell
    pionTime = np.inf  # Not currently accounted for

    return min(tDiff, tAdv, pionTime)


def getDVDR(rShell, X, region, model):
    """Set of coupled ODEs giving dv/dr, dp/dr, and dt/dr

    Args:
        rShell (number): The radius of the shell in cm
        X (array of numbers): An array with the current value of [vShell, pCR, t].
        region (region): The current region
        model (model): The current model

    Returns:
        array of numbers: Returns [dv/dr, dp/dr, dt/dr]
    """

    vShell, pCR, t = X

    dpdr = 0

    if model.energyInjection:
        dpdr += model.windToCREnergyFraction * \
            region.energyDotWind / (4 * np.pi * rShell**3 * vShell)

    if model.advectionPressure:
        dpdr -= 4 * pCR / rShell

    if model.diffusionPressure:
        dpdr -= c * model.meanFreePath * pCR / (vShell * rShell**2)

    if model.streamPressure:
        dpdr -= 0  # To-Do

    dpdr = dpdr / vShell

    dvdr = pCR * 4 * np.pi * rShell**2/(region.massShell*vShell) - G*(region.massShell + region.massNewStars +
                                                                      region.massOldStars*(min(rShell, region.radiusOldStars)/region.radiusOldStars)**3)/(vShell*rShell**2)

    # Old dvdr that uses P ~ Edot * t
    # dvdr =  model.windToCREnergyFraction*region.energyDotWind*model.coverageFraction / rShell /(region.massShell*vShell) * getMinimumTime(rShell, vShell, model.meanFreePath, region.pionTime) - G*(region.massShell + region.massNewStars)/(vShell*rShell**2)

    if model.sweepUpMass:
        dvdr -= vShell*4*np.pi*region.gasDensity * \
            (rShell - model.gasColumnHeight)**2 / region.massShell

    dtdr = 1/vShell

    return dvdr, dpdr, dtdr

# %%
# Define quantity functions - Not currently used
###############################################################################

# def getLGamma(Edot_cr, t_diff, t_pion):

#     L_gamma = Edot_cr/3 * np.minimum(1,t_diff/t_pion)

#     return L_gamma

# def getPCR(Edot_cr, t, V):
#     P_cr = Edot_cr * t / (3*V)
#     return P_cr

# Find the critical , not giving the right answer probably a math error.
# def getCritRadius(region, model):

#     func = lambda R: 3 * region.energyDotWind * model.windToCREnergyFraction / (region.massShell * c * model.meanFreePath) * R**4 * (1 - region.radius**2 / R**2) - G * region.massTotal / region.radius * R**2 * (1 - region.radius / R) - c**2 * model.meanFreePath**2 / 9

#     criticalRadius = fsolve(func, 5*region.radius)

    # criticalRadius = np.sqrt(region.massShell * model.meanFreePath * c * G * region.massTotal/(region.energyDotWind * model.windToCREnergyFraction * region.radius) * (1 + np.sqrt(1 + (4*region.energyDotWind * region.radius**2)/(region.massShell * G**2 * region.massTotal**2))))/np.sqrt(6)

    # return criticalRadius


# %%
# Fill out current model and region
###############################################################################
model = fiducial
region = testRegion

v0 = 10**-4
t0 = 10**6
p0 = 10**-11

X0 = [v0, p0, t0]

rSpan = [model.gasColumnHeight, 1000*region.radius]
r = np.linspace(rSpan[0], rSpan[1], 2000)

print(str(model))
print(str(region))
ODESolve = inte.solve_ivp(getDVDR, rSpan, X0, args=[
                          region, model], max_step=cm_pc, rtol=1)


# Plots
###############################################################################

# %%
# Plot ODE
#################################################

fig, ax = plt.subplots(dpi=200)

plt.plot(ODESolve.t/cm_pc, ODESolve.y[0]/10**5, label=r"$v$")
# plt.plot(ODESolve.t/cm_pc, c/10**5 * model.meanFreePath / (3*ODESolve.t), label = r"$v_{\rm crit}$")

plt.xscale('log')
plt.yscale('log')

plt.ylim(1)

# plt.text(0.01, 0.9, '{}'.format(str(model)), transform = ax.transAxes)
plt.text(0.7, 0.17, r'$M_{\rm Star}\, 10^4\,M_\odot$', transform=ax.transAxes)
plt.text(0.7, 0.1, r'$M_{\rm shell}\, 10^4\,M_\odot$', transform=ax.transAxes)
plt.text(0.7, 0.03,
         r'$\dot{E}_{\rm CR}\, 2.5\times 10^3\, L_\odot$', transform=ax.transAxes)

plt.xlabel("Distance (pc)")
plt.ylabel("Velocity (km/s)")

# plt.legend()


# %%
