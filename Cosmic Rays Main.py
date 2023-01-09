# %%
from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy import integrate as inte
from scipy.optimize import fsolve
# from hoki import load
from astropy import constants as con
from astropy import units as u

# Pion lifetime in years
pion_lifetime = 5*10**7 * u.yr

# %%
# Set fiducial values
###############################################################################
meanFreePathFiducial = 0.01  # pc
gasColumnHeightFiducial = 10  # pc
windToCREnergyFractionFiducial = 0.1  # Fraction
coverageFractionFiducial = 1  # Fraction
vInitialFiducial = 10 # km/s
tInitialFiducial = 0 # yr
eddRatioFiducial = 1

ageFiducial = 1 # Myr
luminosityFiducial = 10**8 # LSun
energyDotWindFiducial = 2500 # LSun
radiusFiducial = 10 # pc
radiusOldStarsFiducial = 10**4 # pc
massShellFiducial = 10**4 # MSun
massNewStarsFiducial = 10**4 # MSun
massOldStarsFiducial = 0 # MSun
gasDensityFiducial = 0 # MSun/pc^2

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
    def __init__(self, name,
                meanFreePath = meanFreePathFiducial,
                gasColumnHeight = gasColumnHeightFiducial,
                windToCREnergyFraction = windToCREnergyFractionFiducial,
                vInitial = vInitialFiducial,
                tInitial = tInitialFiducial,
                coverageFraction = coverageFractionFiducial,
                eddRatio = eddRatioFiducial,
                energyInjection = energyInjectionFiducial,
                advectionPressure = advectionPressureFiducial,
                diffusionPressure = diffusionPressureFiducial,
                pionPressure = pionPressureFiducial,
                streamPressure = streamPressureFiducial,
                sweepUpMass = sweepUpMassFiducial):
        """A model object contains the base data and parameters not related to region data for calculations.

        Args:
            name (string): The name of the model
            meanFreePath (Float, optional): mean free path. Input in parsecs, will be converted to cm and a unit label added. Defaults to meanFreePathFiducial.
            gasColumnHeight (Float, optional): Initial column height. Input in parsecs, will be converted to cm and a unit label added. Defaults to gasColumnHeightFiducial.
            vInitial (Float, optional): Initial velocity. Input in km/s, will be converted to cm/s and a unit label added. Defaults to vInitialFiducial.
            tInitial (Float, optional): Initial age of the cluster. Input in yr, will be converted to s and a unit label added. Defaults to tInitialFiducial. Not currently used.
            windToCREnergyFraction (Float, optional): Parameter for what fraction of wind energy is converted to CR energy. Defaults to windToCREnergyFractionFiducial.
            coverageFraction (Float, optional): Shell coverage fraction. Defaults to coverageFractionFiducial.
            eddRatio (Float, optional): The initial Eddington ratio. Defaults to eddRatioFiducial.
            energyInjection (Boolean, optional): Boolean to enable energy injection in the model. Defaults to energyInjectionFiducial.
            advectionPressure (Boolean, optional): Boolean to enable advection in the model. Defaults to advectionPressureFiducial.
            diffusionPressure (Boolean, optional): Boolean to enable diffusion in the model. Defaults to diffusionPressureFiducial.
            pionPressure (Boolean, optional): Boolean to enable pion decay in the model. Defaults to pionPressureFiducial.
            streamPressure (Boolean, optional): Boolean to enable streaming in the model, not currently implemented. Defaults to streamPressureFiducial.
            sweepUpMass (Boolean, optional): Boolean to enable the sweeping up of additional mass in the model. Defaults to sweepUpMassFiducial.
        """
        self.name = name
        self.meanFreePath = (meanFreePath * u.pc).cgs
        self.gasColumnHeight = (gasColumnHeight * u.pc).cgs
        self.windToCREnergyFraction = windToCREnergyFraction
        self.coverageFraction = coverageFraction
        self.eddRatio = eddRatio
        self.vInitial = (vInitial * u.km / u.s).cgs
        self.tInitial = (tInitial * u.yr).cgs
        self.energyInjection = energyInjection
        self.advectionPressure = advectionPressure
        self.diffusionPressure = diffusionPressure
        self.pionPressure = pionPressure
        self.streamPressure = streamPressure
        self.sweepUpMass = sweepUpMass

    def __str__(self):
        return f"Model: {self.name}"

class region:
    # All units are cgs unless specified
    def __init__(self, name,
            age = ageFiducial,
            luminosity = luminosityFiducial,
            energyDotWind = energyDotWindFiducial,
            radius = radiusFiducial,
            radiusOldStars = radiusOldStarsFiducial,
            massShell = massShellFiducial,
            massNewStars = massNewStarsFiducial,
            massOldStars = massOldStarsFiducial,
            gasDensity = gasDensityFiducial):
        """_summary_

        Args:
            name (String): The region name or reference.
            age (Float , optional): Region age. Input in Myr, unit will be assigned to it. Defaults to ageFiducial
            luminosity (Float , optional): Stellar luminosity of the region. Input in solar luminosities, unit will be assigned to it. Defaults to luminosityFiducial
            energyDotWind (Float , optional): The energy input to the region from stellar wind. Input in solar luminosities, unit will be assigned to it. Defaults to energyDotWindFiducial
            radius (Float , optional): The radius of the region. Input in pc, unit will be assigned to it. Currently not used in favor of the model's gasColumnHeight. Defaults to radiusFiducial
            radiusOldStars (Float , optional): The radius of the old stellar population. Input in pc, unit will be assigned to it. Defaults to radiusOldStarsFiducial
            massShell (Float , optional): The mass of the gas shell. Input in solar masses, unit will be assigned to it. Defaults to massShellFiducial
            massNewStars (Float , optional): The mass of new stars in the cluster. Input in solar masses, unit will be assigned to it. Defaults to massNewStarsFiducial
            massOldStars (Float , optional): The mass of old stars in the cluster. Input in solar masses, unit will be assigned to it. This is not the enclosed old stellar mass, but the total. Defaults to massOldStarsFiducial
            gasDensity (Float , optional): The density of cold gas outside the gas shell, which provides the material to be swept up. Input in ____. Not currently used. Defaults to gasDensityFiducial

        Additional parameters (automatically calculated):
            massTotal (Float): The total region mass in grams.
            electronDensity (Float): The electron number density in n/cm^3. Not currently used but important for pion losses.
            pionTime (Float): The pion timescale in s. Not currently used.
            eddPressure (Float): The Eddington pressure for the initial region conditions, in Bayres.
        """
        self.name = name
        self.age = age
        self.luminosity = (luminosity * u.solLum).cgs
        self.energyDotWind = (energyDotWind * u.solLum).cgs
        self.radius = (radius * u.pc).cgs
        self.radiusOldStars = (radiusOldStars * u.pc).cgs
        self.massShell = (massShell * u.solMass).cgs
        self.massNewStars = (massNewStars * u.solMass).cgs
        self.massOldStars = (massOldStars * u.solMass).cgs
        self.massTotal = self.massNewStars + self.massShell
        self.gasDensity = gasDensity
        self.electronDensity = self.massShell / \
            (4/3*np.pi*(self.radius)**3) / \
            con.m_p.cgs
        self.pionTime = pion_lifetime / self.electronDensity
        self.eddPressure = (con.G * self.massTotal * self.massShell / (4 * np.pi * self.radius**4)).to(u.Ba)

    def __str__(self):
        return f"Region: {self.name}"

class results:
    def __init__(self, model = None, region = None, file = None):
        """Holds the results and methods for printing, saving, and loading them.
         
         Args:
            model (model): A model object
            region (region): A region object
            file (string): A filename to read from
        """
        if file is None:
            if model is None or region is None:
                raise Exception("Results must either have models and regions, or a filepath")
            self.name = model.name + " " + region.name
            self.model = model
            self.region = region

            self.calculate()
        else:
            self.load(file)
    
    def __getItem__(self, key):
        return self.item[key]

    def calculate(self):
        """Calculates the results for a given model and region
        
        Args:
            model (model): A model object
            region (region): A region object
        """
        ODESolve = solveODE(self.model, self.region)

        r = (ODESolve.t * u.cm).to(u.pc)
        v = (ODESolve.y[0] * u.cm/u.s).to(u.km/u.s)
        p = ODESolve.y[1] * u.Ba
        t = ODESolve.y[2] * u.yr

        self.radius = (ODESolve.t * u.cm).to(u.pc)
        self.velocity = (ODESolve.y[0] * u.cm/u.s).to(u.km/u.s)
        self.pressure = ODESolve.y[1] * u.Ba
        self.time = (ODESolve.y[2] * u.s).to(u.yr)

    def save(self):
        with open(self.name + ".result", 'ab') as file:
            pickle.dump(self.__dict__, file)

    def load(self, fileName):
        with open(fileName, 'rb') as file:
            self.__dict__ = pickle.load(file)

    def plot(self, x, y, *z, scale = None):
        """_summary_

        Args:
            x (string): _description_
            y (string): _description_
            z (string): 
            scale (_type_, optional): _description_. Defaults to None.
        """
        multiplot = len(z) != 0

        X = getattr(self, x)
        Y = getattr(self, y)

        if multiplot:
            

            fig, ax1 = plt.subplots(dpi = 200)
            ax2 = plt.twinx(ax1)

            ax1.plot(X.value, Y.value, label = y)
            
            for i, arg in enumerate(z):
                Z = getattr(self, arg)
                ax2.plot(X.value, Z.value, linestyle = "--", label = f"{z[i]}")

            if scale != None:
                ax1.set_xscale(scale)
                ax1.set_yscale(scale)
                ax2.set_yscale(scale)


            ax1.set_xlabel(f"{x} ({X.unit})")
            ax1.set_ylabel(f"{y} ({Y.unit})")
            ax2.set_ylabel(f"{z[0]} ({Z[0].cgs.unit})")

            ax1.legend()
            ax2.legend()

        else:
            plt.figure(dpi = 200)
            plt.title(self.name)
            plt.plot(getattr(self, x), getattr(self, y), label = y)

            plt.xlabel(f"{x} ({getattr(self, x).unit})")
            plt.ylabel(f"{y} ({getattr(self, y).unit})")

            if scale != None:
                plt.xscale(scale)
                plt.yscale(scale)


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
    tDiff = 3*rShell**2 / (con.c * model.meanFreePath)
    tAdv = rShell/vShell
    pionTime = np.inf  # Not currently accounted for

    return min(tDiff, tAdv, pionTime)


def getDVDR(rShell, X, region, model):
    """Set of coupled ODEs giving dv/dr, dp/dr, and dt/dr

    Args:
        rShell (number): The radius of the shell in cm
        X (array): An array with the current value of [vShell, pCR, t].
        region (region): The current region
        model (model): The current model

    Returns:
        array of numbers: Returns [dv/dr, dp/dr, dt/dr]
    """
    vShell, pCR, t = X

    dpdr = 0

    if model.energyInjection:
        dpdr += model.windToCREnergyFraction * \
            region.energyDotWind.value / (4 * np.pi * rShell**3 * vShell)

    if model.advectionPressure:
        dpdr -= 4 * pCR / rShell

    if model.diffusionPressure:
        dpdr -= con.c.cgs.value * model.meanFreePath.value * pCR / (vShell * rShell**2)

    if model.streamPressure:
        dpdr -= 0  # To-Do

    dvdr = pCR * 4 * np.pi * rShell**2/(region.massShell.value*vShell) - \
        con.G.cgs.value*(region.massShell.value + region.massNewStars.value)/(vShell*rShell**2)

    # Old dvdr that uses P ~ Edot * t
    # dvdr =  model.windToCREnergyFraction*region.energyDotWind*model.coverageFraction / rShell /(region.massShell*vShell) * getMinimumTime(rShell, vShell, model.meanFreePath, region.pionTime) - G*(region.massShell + region.massNewStars)/(vShell*rShell**2)

    if model.sweepUpMass:
        dvdr -= vShell*4*np.pi*region.gasDensity.value * \
            (rShell - model.gasColumnHeight.value)**2 / region.massShell.value

    dtdr = 1/abs(vShell)

    return dvdr, dpdr, dtdr


# %%
# Define a function to return the ODE result
def solveODE(model, region, verbose = True):
    """Returns the ODEs for a given model and region

    Args:
        model (_type_): A model object
        region (_type_): a region object.
        verbose (bool, optional): Whether to be verbose during run. Defaults to True.

    Returns:
        _type_: _description_
    """
    X0 = [model.vInitial.value, model.eddRatio* 2 * region.eddPressure.value, model.tInitial.value]

    rSpan = [model.gasColumnHeight.value, 1000*model.gasColumnHeight.value]

    if verbose:
        print("Calculating ODE for " + str(model) + " " + str(region))

    ODESolve = inte.solve_ivp(getDVDR, rSpan, X0, args=[
                            region, model], max_step=(1 * u.pc).cgs.value, rtol=1)

    return ODESolve


# Define  basic models
###############################################################################
# Models are created using the model class, a model only requires a name, all other parameters are set using the fiducial values
fiducial = model("fiducial")
fiducialOnlyAdvection = model("Only advection", energyInjection=False, diffusionPressure=False, pionPressure=False, streamPressure=False)
fiducialNoDiffusion = model("No diffusion", diffusionPressure=False)
fiducialOnlyInput = model("Only energy injection", advectionPressure=False, diffusionPressure=False, pionPressure=False, streamPressure=False)
fiducialNoAdvection = model("No advection", advectionPressure=False)
###############################################################################

# Define regions.
###############################################################################
testRegion = region("Test Region")


# %%
# Fill out current models and regions
###############################################################################

modelOne = model("lambda CR: 0.01 pc, R0: 10 pc")
modelTwo = model("lambda CR: 0.1 pc, R0: 10 pc", meanFreePath = 0.1)
modelThree = model("lambda CR: 0.01 pc, R0: 50 pc", gasColumnHeight = 50)
modelFour = model("lambda CR: 0.1 pc, R0: 50 pc", meanFreePath = 0.1, gasColumnHeight = 50)

regionOne = region(r"MShell: $10^4$ $M_\odot$", age=1, luminosity=10**8, energyDotWind=2500, radius=10,
                        radiusOldStars=10**4, massShell=10**4, massNewStars=10**4, massOldStars=0, gasDensity=0)
regionTwo = region(r"MShell: $10^5$ $M_\odot$", age=1, luminosity=10**8, energyDotWind=2500, radius=10,
                        radiusOldStars=10**4, massShell=10**5, massNewStars=10**4, massOldStars=0, gasDensity=0)
regionThree = region(r"MShell: $10^3$ $M_\odot$", age=1, luminosity=10**8, energyDotWind=2500, radius=10,
                        radiusOldStars=10**4, massShell=10**3, massNewStars=10**4, massOldStars=0, gasDensity=0)

modelList = [modelOne, modelTwo, modelThree, modelFour]
regionList = [regionOne, regionTwo, regionThree]

for currentModel in modelList:
    for currentRegion in regionList:
        currentResult = results(currentModel, currentRegion)
        currentResult.plot("radius", "velocity", scale = "symlog")

## %%
## Define quantity functions - Not currently used
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

# Plots
###############################################################################

# # %%
# # Plot ODE
# #################################################

# fig, ax = plt.subplots(dpi=200)

# initialForce = (region.eddPressure * 4 * np.pi * region.radius**2).to(u.N)

# analyticVelocity = np.sqrt(initialForce * 4 * np.pi * r / region.massShell - con.G * (region.massNewStars + region.massShell)/r + 10**8 * u.m**2 / u.s**2).to(u.km/u.s)

# plt.plot(r, v, label=r"$v$")
# plt.plot(r, analyticVelocity, label = r"Analytic velocity")
# # plt.plot(ODESolve.t/cm_pc, c/10**5 * model.meanFreePath / (3*ODESolve.t), label = r"$v_{\rm crit}$")

# plt.xscale('log')
# # plt.yscale('log')

# # plt.ylim(1)

# # plt.text(0.01, 0.9, '{}'.format(str(model)), transform = ax.transAxes)
# # plt.text(0.7, 0.17, r'$M_{\rm Star}\, 10^4\,M_\odot$', transform=ax.transAxes)
# # plt.text(0.7, 0.1, r'$M_{\rm shell}\, 10^4\,M_\odot$', transform=ax.transAxes)
# # plt.text(0.7, 0.03,
# #          r'$\dot{E}_{\rm CR}\, 2.5\times 10^3\, L_\odot$', transform=ax.transAxes)

# plt.xlabel(f"Distance ({r.unit})")
# plt.ylabel(f"Velocity ({v.unit})")

# plt.legend()

# # %%

## Diagnostic plot of the ODEs
################################################

# fig, ax1 = plt.subplots(dpi = 200)

# r = ODESolve.t * u.cm
# v = ODESolve.y[0] * u.cm / u.s
# p = ODESolve.y[1] * u.Ba
# t = ODESolve.y[2] * u.yr

# dvdr, dpdr, dtdr = getDVDR(r.value, [v.value, p.value, t.value], region, model)

# energyInjection = model.windToCREnergyFraction * \
##             region.energyDotWind / (4 * np.pi * r**3 * v)

# advectionPressure = -4 * p / r

# diffusionPressure = -con.c.cgs * model.meanFreePath * p / (v * r**2)

# ax2 = plt.twinx(ax1)

# ax1.plot(r.to(u.pc).value, p.value, label = "Pressure")
# ax2.plot(r.to(u.pc).value, dpdr, 'k', label = "dp/dr")
# ax2.plot(r.to(u.pc).value, energyInjection.cgs + advectionPressure.cgs + diffusionPressure.cgs, 'k--', label = "all pressures")
# ax2.plot(r.to(u.pc).value, energyInjection.cgs, 'r--', label = "Energy Injection")
# ax2.plot(r.to(u.pc).value, advectionPressure.cgs, 'b--', label = "Advection")
# ax2.plot(r.to(u.pc).value, diffusionPressure.cgs, 'g--', label = "Diffusion")

# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax2.set_yscale('symlog')

# # ax1.set_ylim(5*10**-12, 2*10**-11)
# ax2.set_ylim(-10**-30, 10**-30)

# ax1.set_xlabel(f"Distance ({u.pc})")
# ax1.set_ylabel(f"Pressure ({p.unit})")
# ax2.set_ylabel(f"dP/dr ({diffusionPressure.cgs.unit})")

# ax1.legend(loc = 8)
# ax2.legend()

# # %%
# fig, ax1 = plt.subplots(dpi = 200)

# r = ODESolve.t * u.cm
# v = ODESolve.y[0] * u.cm / u.s
# p = ODESolve.y[1] * u.Ba
# t = ODESolve.y[2] * u.yr

# initialForce = (region.eddPressure * 4 * np.pi * region.radius**2).to(u.N)

# analyticPressure =  initialForce / (4*np.pi*r**2)

# force = (p * 4 * np.pi * r**2).cgs
# gravity = (con.G * (region.massNewStars + region.massShell) * region.massShell / r**2).cgs

# ax2 = plt.twinx(ax1)

# ax1.plot(r.to(u.pc).value, p.value, label = "Pressure")
# ax1.plot(r.to(u.pc).value, analyticPressure.cgs.value, label = "Analytic pressure")
# ax2.plot(r.to(u.pc).value, force.value, 'k', label = r"p4$\pi r^2$")
# ax2.plot(r.to(u.pc).value, gravity.value, 'r', label = "Gravity")

# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax2.set_yscale('log')

# # ax1.set_ylim(5*10**-12, 2*10**-11)
# # ax2.set_ylim(-10**-30, 10**-30)

# ax1.set_xlabel(f"Distance ({u.pc})")
# ax1.set_ylabel(f"Pressure ({p.unit})")
# ax2.set_ylabel(f"Force ({force.cgs.unit})")

# ax1.legend(loc = 8)
# ax2.legend()

# %%
