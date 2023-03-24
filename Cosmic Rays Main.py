# %%
from matplotlib import pyplot as plt
import numpy as np
import pickle
import math
from scipy import integrate as inte
from scipy.optimize import fsolve
from astropy import constants as con
from astropy import units as u
from matplotlib.collections import LineCollection


# %%
# Set fiducial values
###############################################################################
meanFreePathFiducial = 0.01  # pc
gasColumnHeightFiducial = 10  # pc
windToCREnergyFractionFiducial = 0.1  # Fraction
vWindFiducial = 500 # km/s
coverageFractionFiducial = 1  # Fraction
vInitialFiducial = 1 # km/s
tInitialFiducial = 0 # yr
eddRatioFiducial = 2 # We assume that all regions are 2x Eddington by default
pionLifetimeFiducial = 5*10**7 # yr
vAlfvenFiducial = 10 # km/s
externalMassScaleFiducial = 0 # External masss density will scale as (r0/r)^x where this is x
luminosityPerMassFiducial = 1500 # LSun/MSun The solar luminosities per mass for a star cluster.

ageFiducial = 1 # Myr
# luminosityFiducial = 10**8 # LSun
radiusFiducial = 100 # pc
radiusOldStarsFiducial = 10**4 # pc
massShellFiducial = 10**5 # MSun
massNewStarsFiducial = 10**4 # MSun
energyDotWindFiducial = 2500 * massNewStarsFiducial/10**4 # LSun
massOldStarsFiducial = 10**4 # MSun
externalGasDensityFiducial = 1 # protons/cm^3

# Turn on or off various pressures in the model.
energyInjectionFiducial = True
advectionPressureFiducial = True
diffusionPressureFiducial = True
pionPressureFiducial = True
streamPressureFiducial = False
sweepUpMassFiducial = False
radiationPressureFiducial = False

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
                vWind = vWindFiducial,
                vInitial = vInitialFiducial,
                tInitial = tInitialFiducial,
                coverageFraction = coverageFractionFiducial,
                eddRatio = eddRatioFiducial,
                pionLifetime = pionLifetimeFiducial,
                vAlfven = vAlfvenFiducial,
                energyInjection = energyInjectionFiducial,
                advectionPressure = advectionPressureFiducial,
                diffusionPressure = diffusionPressureFiducial,
                pionPressure = pionPressureFiducial,
                streamPressure = streamPressureFiducial,
                sweepUpMass = sweepUpMassFiducial,
                radiationPressure = radiationPressureFiducial,
                externalMassScale = externalMassScaleFiducial):
        """A model object contains the base data and parameters not related to region data for calculations.

        Args:
            name (string): The name of the model
            meanFreePath (Float, optional): mean free path. Input in parsecs, will be converted to cm and a unit label added. Defaults to meanFreePathFiducial.
            gasColumnHeight (Float, optional): Initial column height. Input in parsecs, will be converted to cm and a unit label added. Defaults to gasColumnHeightFiducial.
            vInitial (Float, optional): Initial velocity. Input in km/s, will be converted to cm/s and a unit label added. Defaults to vInitialFiducial.
            tInitial (Float, optional): Initial age of the cluster. Input in yr, will be converted to s and a unit label added. Defaults to tInitialFiducial. Not currently used.
            windToCREnergyFraction (Float, optional): Parameter for what fraction of wind energy is converted to CR energy. Defaults to windToCREnergyFractionFiducial.
            vWind (number): The velocity of the wind. Input in km/s, will be converted to cgs.
            coverageFraction (Float, optional): Shell coverage fraction. Defaults to coverageFractionFiducial.
            eddRatio (Float, optional): The initial Eddington ratio. Defaults to eddRatioFiducial.
            pionLifetime (Float, optional): The pion lifetime scale. Input in yr, will be converted to s and a unit label added. Defaults to pionLifetimeFiducial
            vAlphane (Floar, optional): The Alphane velocity of the model. Given in km/s, will be converted to cgs. Defaults to vAlphaneFiducial.
            energyInjection (Boolean, optional): Boolean to enable energy injection in the model. Defaults to energyInjectionFiducial.
            advectionPressure (Boolean, optional): Boolean to enable advection in the model. Defaults to advectionPressureFiducial.
            diffusionPressure (Boolean, optional): Boolean to enable diffusion in the model. Defaults to diffusionPressureFiducial.
            pionPressure (Boolean, optional): Boolean to enable pion decay in the model. Defaults to pionPressureFiducial.
            streamPressure (Boolean, optional): Boolean to enable streaming in the model. Defaults to streamPressureFiducial.
            sweepUpMass (Boolean, optional): Boolean to enable the sweeping up of additional mass in the model. Defaults to sweepUpMassFiducial.
            radiationPressure (Boolean, optional): Boolean to enable radiation pressure in the model. Defaults to radiationPressureFiducial.
            externalMassScale (Float, optional): The scale factor for externally swept up mass. Defaults to externalMassScaleFiducial
        """
        self.name = name
        self.meanFreePath = (meanFreePath * u.pc).cgs
        self.gasColumnHeight = (gasColumnHeight * u.pc).cgs
        self.windToCREnergyFraction = windToCREnergyFraction
        self.vWind = (vWind * u.km / u.s).cgs
        self.coverageFraction = coverageFraction
        self.eddRatio = eddRatio
        self.pionLifetime = (pionLifetime * u.yr).cgs
        self.vAlphane = (vAlfven * u.km/u.s).cgs
        self.vInitial = (vInitial * u.km / u.s).cgs
        self.tInitial = (tInitial * u.yr).cgs
        self.energyInjection = energyInjection
        self.advectionPressure = advectionPressure
        self.diffusionPressure = diffusionPressure
        self.pionPressure = pionPressure
        self.streamPressure = streamPressure
        self.sweepUpMass = sweepUpMass
        self.externalMassScale = externalMassScale
        self.radiationPressure = radiationPressure

    def __str__(self):
        return f"Model: {self.name}"

class region:
    # All units are cgs unless specified
    def __init__(self, name,
            age = ageFiducial,
            luminosity = None,
            energyDotWind = energyDotWindFiducial,
            radius = radiusFiducial,
            radiusOldStars = radiusOldStarsFiducial,
            massShell = massShellFiducial,
            massNewStars = massNewStarsFiducial,
            massOldStars = massOldStarsFiducial,
            externalGasDensity = externalGasDensityFiducial,
            luminosityPerMass = luminosityPerMassFiducial):
        """_summary_

        Args:
            name (String): The region name or reference.
            age (Float , optional): Region age. Input in Myr, unit will be assigned to it. Defaults to ageFiducial
            luminosity (Float , optional): Stellar luminosity of the region. Input in solar luminosities, unit will be assigned to it. Will be calculated automatically if not set.
            energyDotWind (Float , optional): The energy input to the region from stellar wind. Input in solar luminosities, unit will be assigned to it. Defaults to energyDotWindFiducial
            radius (Float , optional): The radius of the region. Input in pc, unit will be assigned to it. Currently not used in favor of the model's gasColumnHeight. Defaults to radiusFiducial
            radiusOldStars (Float , optional): The radius of the old stellar population. Input in pc, unit will be assigned to it. Defaults to radiusOldStarsFiducial
            massShell (Float , optional): The mass of the gas shell. Input in solar masses, unit will be assigned to it. Defaults to massShellFiducial
            massNewStars (Float , optional): The mass of new stars in the cluster. Input in solar masses, unit will be assigned to it. Defaults to massNewStarsFiducial
            massOldStars (Float , optional): The mass of old stars in the cluster. Input in solar masses, unit will be assigned to it. This is not the enclosed old stellar mass, but the total. Defaults to massOldStarsFiducial
            externalGasDensity (Float , optional): The density of cold gas outside the gas shell, which provides the material to be swept up. Input in protons/cm^3, will be converted to g/cm^3. Defaults to externalGasDensityFiducial
            luminosityPerMass (Float, optional): The luminosity per mass of the stellar cluster. Input in solar luminosities per solar mass, will be converted to cgs. Defaults to luminosityPerMassFiducial

        Additional parameters (automatically calculated):
            massTotal (Float): The total region mass in grams.
            electronDensity (Float): The electron number density in n/cm^3. Not currently used but important for pion losses.
            pionTime (Float): The pion timescale in s. Not currently used.
            eddPressure (Float): The Eddington pressure for the initial region conditions, in Bayres.
        """
        self.name = name
        self.age = (age * u.Myr).cgs
        self.energyDotWind = (energyDotWind * u.solLum).cgs
        self.radius = (radius * u.pc).cgs
        self.radiusOldStars = (radiusOldStars * u.pc).cgs
        self.massShell = (massShell * u.solMass).cgs
        self.massNewStars = (massNewStars * u.solMass).cgs
        self.massOldStars = (massOldStars * u.solMass).cgs
        self.massTotal = self.massNewStars + self.massShell
        self.externalGasDensity = (externalGasDensity * con.m_p / u.cm**3).cgs
        self.electronDensity = self.massShell / (4/3*np.pi*(self.radius)**3) / con.m_p.cgs
        self.eddPressure = (con.G * self.massTotal * self.massShell / (4 * np.pi * self.radius**4)).to(u.Ba)
        self.luminosityPerMass = (luminosityPerMass * u.solLum / u.solMass).cgs
        if luminosity:
            self.luminosity = (luminosity * u.solLum).cgs
        else:
            self.luminosity = self.luminosityPerMass * self.massNewStars

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

    def calculate(self):
        """Calculates the results for a given model and region
        
        Args:
            model (model): A model object
            region (region): A region object
        """
        ODESolve = solveODE(self.model, self.region)

        self.radius = (ODESolve.t * u.cm).to(u.pc)
        self.velocity = (ODESolve.y[0] * u.cm/u.s).to(u.km/u.s)
        self.pressure = ODESolve.y[1] * u.Ba
        self.time = (ODESolve.y[2] * u.s).to(u.yr)
        self.massShell = ODESolve.y[3] * u.g
        self.mDotWind = (2 * self.region.energyDotWind / self.model.vWind**2).cgs
        self.innerShockGasDensity = (self.mDotWind / (4 * np.pi * self.radius**2 * self.model.vWind)).cgs
        self.innerShockNumberDensity = 4 * self.innerShockGasDensity / con.m_p.cgs
        self.tPion = self.model.pionLifetime / self.innerShockNumberDensity / u.cm**3
        self.tDiff = (3*self.radius**2 / (con.c * self.model.meanFreePath)).cgs
        self.tAdv = (self.radius/self.velocity).cgs
        self.tStream = (self.radius/self.model.vAlphane).cgs
        self.energyCR = (self.pressure * 4 * np.pi * self.radius**3).cgs
        self.gammaLuminosity = (1 / 3 * self.energyCR / self.tPion).cgs

        self.dvdr, self.dpdr, _, self.dmdr = getDVDR(self.time.value, [self.velocity.value, self.pressure.value, self.radius.value, self.massShell.value], self.region, self.model)

        self.tauPi = (self.radius * (self.region.externalGasDensity / con.m_p * u.cm**3)/(con.c * self.model.pionLifetime)).cgs
        self.tauScatt = (self.radius/self.model.meanFreePath).cgs

        self.effectiveExternalOpticalDepth = np.sqrt(self.tauPi*(self.tauPi + self.tauScatt)).cgs

        self.advectionLosses = self.energyCR*self.velocity/self.radius/3

        self.externalGammaLuminosity = self.energyCR/3/self.tDiff*(1-np.exp(-self.effectiveExternalOpticalDepth.cgs))

    def save(self):
        with open("Results/" + self.name, 'ab') as file:
            pickle.dump(self.__dict__, file)

    def load(self, fileName):
        with open("Results/" + fileName, 'rb') as file:
            self.__dict__ = pickle.load(file)

    def plot(self, x, y, *z, scale = None):
        """_summary_

        Args:
            x (string): A string giving the attribute to use as the x axis.
            y (string): A string giving the attribute to use as the y axis.
            z (string, optional): A string giving additional attributes to plot on a new axis.
            scale (string, optional): The scale to use for both axes. "log" or "symlog" preferred. Defaults to None.
        """

        X = getattr(self, x)
        Y = getattr(self, y)

        if len(z) != 0:
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
            plt.figure(dpi = 200, facecolor = "white")
            plt.title(self.name)
            plt.plot(X.value, Y.value, label = y)

            plt.xlabel(f"{x} ({getattr(self, x).unit})")
            plt.ylabel(f"{y} ({getattr(self, y).unit})")

            if scale != None:
                plt.xscale(scale)
                plt.yscale(scale)
        
    def multiPlot(self, x, y, result, scale = None):

        """Plots the data from multiple results objects

        Args:
            x (string): A string giving the attribute to use as the x axis.
            y (string): A string giving the attribute to use as the y axis.
            y2 (array): An array containing any number of other results objects to also plot the same y attributes from.
            scale (string, optional): The scale to use for both axes. "log" or "symlog" preferred. Defaults to None.
        """
        plt.figure(dpi = 200, facecolor = "white")
        plt.title(y + " vs. " + x)

        plt.plot(getattr(self, x), getattr(self, y), label = self.name)
        for res in result:
            plt.plot(getattr(res, x), getattr(res, y), label = res.name)
        
        plt.legend(bbox_to_anchor=(1, 1))
        if scale != None:
            plt.xscale(scale)
            plt.yscale(scale)

        plt.xlabel(f"{x} ({getattr(self, x).unit})")
        plt.ylabel(f"{y} ({getattr(self, y).unit})")

    def verify(self):
        """
        Calculates the analytic solution v = sqrt( F0 * 4pi * r / Msh - GMtot / r + C) where C is the integration constant, solved for by solving for v0.
        """

        # Define the initial pressure as the eddington pressure
        # initialPressure = self.region.eddPressure

        # Define the initial pressure as the diffusion pressure
        initialPressure = ((self.model.windToCREnergyFraction * self.region.energyDotWind) * self.model.gasColumnHeight / (con.c * self.model.meanFreePath))

        initialForce = (self.model.eddRatio *  initialPressure * 4 * np.pi * self.model.gasColumnHeight**2).cgs

        integrationConstantDiffusion = np.power(self.model.vInitial,2).cgs - (initialForce * 4 * np.pi * self.model.gasColumnHeight / self.region.massShell).cgs + (con.G * (self.region.massNewStars + self.region.massShell)/self.model.gasColumnHeight).cgs

        analyticVelocityDiffusion = np.sqrt(initialForce * 4 * np.pi * self.radius / self.region.massShell - con.G * (self.region.massNewStars + self.region.massShell)/self.radius + integrationConstantDiffusion).to(u.km/u.s)

        advVelocity = (np.cbrt((3 * self.model.windToCREnergyFraction * self.region.energyDotWind * self.radius/(4 * self.region.massShell)))).to(u.km/u.s)

        fig, ax = plt.subplots(dpi = 200)
        ax2 = plt.twinx(ax)
        ax.plot(self.radius, analyticVelocityDiffusion.to(u.km/u.s), 'k', label = "Velocity (Analytic)")
        ax.plot(self.radius, self.velocity.to(u.km/u.s), 'c', label = "velocity (ODE)")
        ax.plot(self.radius, advVelocity, 'g', label = r"$(\frac{3\dot{E}_{\rm cr}R}{4M_{\rm sh}})^{1/3}$")
        ax2.plot(self.radius, self.tDiff.to(u.yr), 'r--', label = "Diffusion Time")
        ax2.plot(self.radius, self.tAdv.to(u.yr), 'b--', label = "Advection Time")
        
        ax.set_title(self.name)

        ax.legend(loc = 2)
        ax2.legend(loc = 4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax2.set_yscale('log')

        ax.set_xlabel(f"Radius ({self.radius.unit})")
        ax.set_ylabel(f"Velocity ({self.velocity.unit})")
        ax2.set_ylabel(f"Time ({u.yr})")


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
        X (array): An array with the current value of [vShell, pCR, t, mShell].
        region (region): The current region
        model (model): The current model

    Returns:
        array of numbers: Returns [dv/dr, dp/dr, dt/dr]
    """
    vShell, pCR, t, mShell = X

    dpdr = 0

    if model.energyInjection:
        dpdr += model.windToCREnergyFraction * region.energyDotWind.value / (4 * np.pi * rShell**3 * vShell)

    if model.advectionPressure:
        dpdr -= 4 * pCR / rShell

    if model.diffusionPressure:
        dpdr -= con.c.cgs.value * model.meanFreePath.value * pCR / (vShell * rShell**2)

    if model.streamPressure:
        dpdr -= (3*vShell + model.vAlphane.value) * pCR / ( rShell * vShell)

    dvdr = pCR * 4 * np.pi * rShell**2/(mShell*vShell) - con.G.cgs.value*(mShell + region.massNewStars.value)/(vShell*rShell**2)

    dmdr = 0

    if model.radiationPressure:
        dvdr += region.luminosity.value / con.c.cgs.value / vShell / mShell

    if model.sweepUpMass:
        dmdr = 4 * np.pi * region.externalGasDensity.value * (model.gasColumnHeight.value/rShell)**model.externalMassScale * rShell**2
        dvdr -= vShell * dmdr / mShell

    dtdr = 1/abs(vShell)

    return [dvdr, dpdr, dtdr, dmdr]

def solveODE(model, region, verbose = True):
    """Returns the ODEs for a given model and region

    Args:
        model (model): A model object
        region (region): a region object.
        verbose (bool, optional): Whether to be verbose during run. Defaults to True.

    Returns:
        ODESolve: A solve_ivp object.
    """
    initialPressure = ((model.windToCREnergyFraction * region.energyDotWind) * model.gasColumnHeight / (con.c * model.meanFreePath) / (4 * math.pi * model.gasColumnHeight**2)).cgs.value
    print(initialPressure)

    X0 = [model.vInitial.value, initialPressure, region.age.value, region.massShell.value]

    rSpan = (model.gasColumnHeight.value, 1000*model.gasColumnHeight.value)

    if verbose:
        print("Calculating ODE for " + str(model) + " " + str(region))

    ODESolve = inte.solve_ivp(getDVDR, rSpan, X0, method = "Radau", args=[region, model], max_step=(1 * u.pc).cgs.value, rtol = 10e-6, atol = 10e-9)

    return ODESolve

def findNearest(array, value):
    """Finds the nearest value in an array and returns the index

    Args:
        array (array): Array of numbers to search.
        value (array): Array of the values to search for.

    Returns:
        index (int): The index of the nearest value.
    """
    array = np.asarray(array)
    value = np.asarray(value)
    index = np.zeros_like(value)
    for i, val in enumerate(value):
        index[i] = (np.abs(array - val)).argmin()
    return index

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

modelOne = model(r"$\lambda_{\rm CR}$: 0.01 pc", radiationPressure = True)
modelTwo = model(r"$\lambda_{\rm CR}$: 0.03 pc", meanFreePath = 0.03, radiationPressure = True)
modelThree = model(r"$\lambda_{\rm CR}$: 0.007 pc", meanFreePath = 0.007, radiationPressure = True)
modelFour = model(r"$\lambda_{\rm CR}$: 0.01 pc", sweepUpMass = True, radiationPressure = True)
modelFive = model(r"$\lambda_{\rm CR}$: 0.03 pc", meanFreePath = 0.03, sweepUpMass = True, radiationPressure = True)
modelSix = model(r"$\lambda_{\rm CR}$: 0.007 pc", meanFreePath = 0.007, sweepUpMass = True, radiationPressure = True)
modelSeven = model(r"No radiation, $\lambda_{\rm CR}$: 0.01 pc", radiationPressure = False)
modelEight = model(r"No radiation, $\lambda_{\rm CR}$: 0.03 pc", meanFreePath = 0.03, radiationPressure = False)
modelNine = model(r"No radiation, $\lambda_{\rm CR}$: 0.007 pc", meanFreePath = 0.007, radiationPressure = False)
modelTen = model(r"No radiation, $\lambda_{\rm CR}$: 0.01 pc", sweepUpMass = True, radiationPressure = False)
modelEleven = model(r"No radiation, $\lambda_{\rm CR}$: 0.03 pc", meanFreePath = 0.03, sweepUpMass = True, radiationPressure = False)
modelTwelve = model(r"No radiation, $\lambda_{\rm CR}$: 0.007 pc", meanFreePath = 0.007, sweepUpMass = True, radiationPressure = False)
# modelFour = model("lambda CR: 0.07 pc", meanFreePath = 0.07)
# modelFive = model("lambda CR: 0.1 pc", meanFreePath = 0.1)

regionOne = region("")
# regionTwo = region(r"MShell: $10^5$ $M_\odot$", massShell=10**5)
# regionThree = region(r"MShell: $10^3$ $M_\odot$", massShell=10**3)

modelList = [modelOne, modelTwo, modelThree, modelFour, modelFive, modelSix, modelSeven, modelEight, modelNine, modelTen, modelEleven, modelTwelve]
regionList = [regionOne]

# modelList = [modelOne]
# regionList = [regionOne]

resultList = []

for currentModel in modelList:
    for currentRegion in regionList:
        currentResult = results(currentModel, currentRegion)
        resultList.append(currentResult)

# resultList[0].multiPlot("radius", "velocity", resultList[1:-1], scale = "log")
# resultList[0].multiPlot("time", "velocity", resultList[1:-1], scale = "log")

# for res in resultList:
#     res.verify()


# # %%
# # Plot results
# for res in resultList:
#     res.plot("radius", "velocity", scale = "symlog")


## energyInjection = (res.model.windToCREnergyFraction * res.region.energyDotWind / (4 * np.pi * res.radius**3 * res.velocity)).cgs
## advectionPressure = (4 * res.pressure / res.radius).cgs
## streamPressure = ((3*res.velocity + res.model.vAlphane) * res.pressure / ( res.radius * res.velocity)).cgs
## diffusionPressure = (con.c.cgs * res.model.meanFreePath * res.pressure / (res.velocity * res.radius**2)).cgs

# Plots
###############################################################################

# %%
# Plot radius and time dependent velocity
###################################################

fig, ax = plt.subplots(1, 2, dpi = 200, figsize = (10,4), facecolor = "white")

ax[0].plot(resultList[0].radius, resultList[0].velocity, 'k', label = resultList[0].name)
ax[0].plot(resultList[1].radius, resultList[1].velocity, 'b', label = resultList[1].name)
ax[0].plot(resultList[2].radius, resultList[2].velocity, 'g', label = resultList[2].name)
ax[0].plot(resultList[3].radius, resultList[3].velocity, 'k--')
ax[0].plot(resultList[4].radius, resultList[4].velocity, 'b--')
ax[0].plot(resultList[5].radius, resultList[5].velocity, 'g--')
ax[0].plot(resultList[6].radius, resultList[6].velocity, 'r', label = resultList[6].name)
ax[0].plot(resultList[7].radius, resultList[7].velocity, c = 'orange', label = resultList[7].name)
ax[0].plot(resultList[8].radius, resultList[8].velocity, 'c', label = resultList[8].name)
ax[0].plot(resultList[9].radius, resultList[9].velocity, 'r--')
ax[0].plot(resultList[10].radius, resultList[10].velocity, c = 'orange', linestyle = "dashed")
ax[0].plot(resultList[11].radius, resultList[11].velocity, 'c--')

ax[1].plot(resultList[0].time, resultList[0].velocity, 'k', label = "No swept up mass")
ax[1].plot(resultList[1].time, resultList[1].velocity, 'b')
ax[1].plot(resultList[2].time, resultList[2].velocity, 'g')
ax[1].plot(resultList[3].time, resultList[3].velocity, 'k--', label = "Swept up mass")
ax[1].plot(resultList[4].time, resultList[4].velocity, 'b--')
ax[1].plot(resultList[5].time, resultList[5].velocity, 'g--')
ax[1].plot(resultList[6].time, resultList[6].velocity, 'r')
ax[1].plot(resultList[7].time, resultList[7].velocity, c = 'orange')
ax[1].plot(resultList[8].time, resultList[8].velocity, 'c')
ax[1].plot(resultList[9].time, resultList[9].velocity, 'r--')
ax[1].plot(resultList[10].time, resultList[10].velocity, c = 'orange', linestyle = "dashed")
ax[1].plot(resultList[11].time, resultList[11].velocity, 'c--')

ax[0].set_ylim(0.1, 200)
ax[1].set_ylim(0.1, 200)
ax[0].set_xlim(10,100)
ax[1].set_xlim(10**6,5*10**6)

ax[0].legend()
ax[1].legend()

ax[0].set_xlabel(f'Radius ({resultList[0].radius.unit})')
ax[1].set_xlabel(f'Time ({resultList[0].time.unit})')
ax[0].set_ylabel(f'Velocity ({resultList[0].velocity.unit})')

ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_yscale('log')

# %%
# Compare results with and without streaming
###################################################

# noStreaming = model("No Streaming", streamPressure = False)
# streamingModel10 = model(r"$v_{\rm A} = 10 \, km/s$")
# streamingModel30 = model(r"$v_{\rm A} = 30 \, km/s$", vAlfven = 30)
# streamingModel100 = model(r"$v_{\rm A} = 100 \, km/s$", vAlfven = 100)
# streamingModel500 = model(r"$v_{\rm A} = 500 \, km/s$", vAlfven = 500)

# lowMassRegion = region(r"$M_{\rm sh} = 10^3 \, M_\odot$", massShell = 10**3)
# mediumMassRegion = region(r"$M_{\rm sh} = 10^4 \, M_\odot$", massShell = 10**4)
# highMassRegion = region(r"$M_{\rm sh} = 10^5 \, M_\odot$", massShell = 10**5)

# modelList = [noStreaming, streamingModel10, streamingModel30, streamingModel100, streamingModel500]
# regionList = [lowMassRegion, mediumMassRegion, highMassRegion]

# resultList = []

# for currentModel in modelList:
#     for currentRegion in regionList:
#         currentResult = results(currentModel, currentRegion)
#         resultList.append(currentResult)

# # resultList[0].multiPlot("radius", "velocity", resultList[1:-1], scale = "log")
# # resultList[0].multiPlot("time", "velocity", resultList[1:-1], scale = "log")

# plt.figure(dpi = 200, facecolor = "White")

# plt.plot(resultList[0].time, resultList[0].velocity, 'k', label = resultList[0].name)
# plt.plot(resultList[1].time, resultList[1].velocity, 'b', label = resultList[1].name)
# plt.plot(resultList[2].time, resultList[2].velocity, 'r', label = resultList[2].name)

# plt.plot(resultList[3].time, resultList[3].velocity, 'k--', label = resultList[3].name)
# plt.plot(resultList[4].time, resultList[4].velocity, 'b--', label = resultList[4].name)
# plt.plot(resultList[5].time, resultList[5].velocity, 'r--', label = resultList[5].name)

# plt.plot(resultList[6].time, resultList[6].velocity, 'k:', label = resultList[6].name)
# plt.plot(resultList[7].time, resultList[7].velocity, 'b:', label = resultList[7].name)
# plt.plot(resultList[8].time, resultList[8].velocity, 'r:', label = resultList[8].name)

# plt.plot(resultList[9].time, resultList[9].velocity, 'k-.', label = resultList[9].name)
# plt.plot(resultList[10].time, resultList[10].velocity, 'b-.', label = resultList[10].name)
# plt.plot(resultList[11].time, resultList[11].velocity, 'r-.', label = resultList[11].name)

# plt.plot(resultList[12].time, resultList[12].velocity, c = 'k', linewidth = 0.5, label = resultList[12].name)
# plt.plot(resultList[13].time, resultList[13].velocity, c = 'b', linewidth = 0.5, label = resultList[13].name)
# plt.plot(resultList[14].time, resultList[14].velocity, c = 'r', linewidth = 0.5, label = resultList[14].name)

# plt.xscale('log')
# plt.yscale('log')

# plt.ylim(10**-1)

# plt.legend(bbox_to_anchor=(1, 1))

# plt.xlabel("Time (yr)")
# plt.ylabel("Velocity (km/s)")


# %%
# Plot v_infinity for resultsList
###################################################

# from itertools import cycle

# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = cycle(prop_cycle.by_key()['color'])

# plt.figure(dpi = 200, facecolor = "white")

# for res in resultList:
#     advVelocity = (np.cbrt((3 * res.model.windToCREnergyFraction * res.region.energyDotWind * res.radius/(4 * res.region.massShell)))).to(u.km/u.s)

#     color = color=next(colors)

#     plt.plot(res.radius.to(u.pc), res.velocity.to(u.km/u.s), label = res.name, color = color)
#     plt.plot(res.radius.to(u.pc), advVelocity, linestyle = "--", color = color)

# plt.xscale('log')
# plt.yscale('log')

# plt.xlabel(f"Radius ({u.pc})")
# plt.ylabel(f"Velocity ({u.km}/{u.s})")

# plt.ylim(1)


# %%
# Proposal Plot results
###################################################

# proposalRegion = region("proposal region", massShell = 10**3)
# proposalModelOne = model("Constant", meanFreePath = 0.01, sweepUpMass= True)
# proposalModelTwo = model(r"$R^{-1}$", meanFreePath = 0.01, sweepUpMass= True, externalMassScale = 1)
# proposalModelThree = model(r"$R^{-2}$", meanFreePath = 0.01, sweepUpMass= True, externalMassScale = 2)

# proposalResultsOne = results(proposalModelOne, proposalRegion)
# proposalResultsTwo = results(proposalModelTwo, proposalRegion)
# proposalResultsThree = results(proposalModelThree, proposalRegion)

# %%
# Proposal plots
###################################################

# fig, ax = plt.subplots(1,2, dpi = 200, figsize = (10,4), facecolor = "white")

# ax[0].plot(proposalResultsOne.radius.to(u.pc), proposalResultsOne.velocity, label = "Constant")
# ax[0].plot(proposalResultsTwo.radius.to(u.pc), proposalResultsTwo.velocity, label = r"$R^{-1}$")
# ax[0].plot(proposalResultsThree.radius.to(u.pc), proposalResultsThree.velocity, label = r"$R^{-2}$")

# # ax[1].plot(proposalResultsOne.radius.to(u.pc), proposalResultsOne.gammaLuminosity, label = r"$\lambda_{\rm CR} = 0.01\,$ pc")
# # ax[1].plot(proposalResultsTwo.radius.to(u.pc), proposalResultsTwo.gammaLuminosity, label = r"$\lambda_{\rm CR} = 0.03\,$ pc")
# # ax[1].plot(proposalResultsThree.radius.to(u.pc), proposalResultsThree.gammaLuminosity, label = r"$\lambda_{\rm CR} = 0.1\,$ pc")

# # ax[1].plot(proposalResultsOne.radius.to(u.pc), proposalResultsOne.externalGammaLuminosity, linestyle = 'dashed', c = "C0")
# # ax[1].plot(proposalResultsTwo.radius.to(u.pc), proposalResultsTwo.externalGammaLuminosity, linestyle = 'dashed', c = "C1")
# # ax[1].plot(proposalResultsThree.radius.to(u.pc), proposalResultsThree.externalGammaLuminosity, linestyle = 'dashed', c = "C2")

# # ax[1].plot(proposalResultsOne.radius.to(u.pc), proposalResultsOne.advectionLosses.cgs, linestyle = 'dotted', c = "C0")
# # ax[1].plot(proposalResultsTwo.radius.to(u.pc), proposalResultsTwo.advectionLosses.cgs, linestyle = 'dotted', c = "C1")
# # ax[1].plot(proposalResultsThree.radius.to(u.pc), proposalResultsThree.advectionLosses.cgs, linestyle = 'dotted', c = "C2")

# # energyInput = proposalResultsOne.region.energyDotWind.cgs/3 * proposalResultsOne.model.windToCREnergyFraction

# # ax[1].plot(proposalResultsOne.radius.to(u.pc), energyInput*np.ones_like(proposalResultsOne.radius.value), 'k')

# ax[1].plot(proposalResultsOne.radius.to(u.pc), proposalResultsOne.externalGammaLuminosity, c = "C0")
# ax[1].plot(proposalResultsTwo.radius.to(u.pc), proposalResultsTwo.externalGammaLuminosity, c = "C1")
# ax[1].plot(proposalResultsThree.radius.to(u.pc), proposalResultsThree.externalGammaLuminosity, c = "C2")

# # Find these times for putting on the plot
# times = [10**6, 2*10**6, 4*10**6]

# proposalResultsOneTimes = findNearest(proposalResultsOne.time.value - 10**6, times)
# proposalResultsTwoTimes = findNearest(proposalResultsTwo.time.value - 10**6, times)
# proposalResultsThreeTimes = findNearest(proposalResultsThree.time.value - 10**6, times)

# ax[0].scatter(proposalResultsOne.radius.to(u.pc)[proposalResultsOneTimes], proposalResultsOne.velocity[proposalResultsOneTimes], c = "C0")
# ax[0].scatter(proposalResultsTwo.radius.to(u.pc)[proposalResultsTwoTimes], proposalResultsTwo.velocity[proposalResultsTwoTimes], c = "C1")
# ax[0].scatter(proposalResultsThree.radius.to(u.pc)[proposalResultsThreeTimes], proposalResultsThree.velocity[proposalResultsThreeTimes], c = "C2")

# ax[1].scatter(proposalResultsOne.radius.to(u.pc)[proposalResultsOneTimes], (proposalResultsOne.externalGammaLuminosity)[proposalResultsOneTimes], c = "C0")
# ax[1].scatter(proposalResultsTwo.radius.to(u.pc)[proposalResultsTwoTimes], (proposalResultsTwo.externalGammaLuminosity)[proposalResultsTwoTimes], c = "C1")
# ax[1].scatter(proposalResultsThree.radius.to(u.pc)[proposalResultsThreeTimes], (proposalResultsThree.externalGammaLuminosity)[proposalResultsThreeTimes], c = "C2")

# # ax[1].fill_between([10,50], 0.7*10**34, 1.5*10**34, facecolor = "grey", alpha = 0.2)


# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')

# ax[0].set_xlim(10,500)
# ax[1].set_xlim(10,500)

# ax[1].set_ylim(float(5*10**32))
# ax[0].set_ylim(1,100)

# ax[0].set_xlabel('Radius (pc)')
# ax[0].set_ylabel('Velocity (km/s)')
# ax[1].set_xlabel('Radius (pc)')
# ax[1].set_ylabel(r'$\gamma$-ray Luminosity (ergs/s)')

# ax[0].legend()

#  %%

# %%
# Plot verification

# plt.figure(dpi = 200)

# energyInjection = (proposalResultsOne.model.windToCREnergyFraction * proposalResultsOne.region.energyDotWind / (4 * np.pi * proposalResultsOne.radius**3 * proposalResultsOne.velocity)).cgs

# diffTerm = (con.c  * proposalResultsOne.model.meanFreePath * proposalResultsOne.pressure / (proposalResultsOne.velocity * proposalResultsOne.radius**2)).cgs

# advTerm = (4* proposalResultsOne.pressure / proposalResultsOne.radius).cgs

# plt.plot(proposalResultsOne.radius, energyInjection, label = "Energy Injection")
# plt.plot(proposalResultsOne.radius, advTerm, label = "Advection")
# plt.plot(proposalResultsOne.radius, diffTerm, label = "Diffusion")

# plt.xscale("log")
# plt.yscale("log")

# plt.xlabel("Radius")
# plt.ylabel(f"Pressure {energyInjection.unit}")

# plt.legend()


# %%


# # %%
# # Plot ODE
# #################################################

# fig, ax = plt.subplots(dpi=200)

# initialForce = (testRegion.eddPressure * 4 * np.pi * testRegion.radius**2).to(u.N)

# analyticVelocity = np.sqrt(initialForce * 4 * np.pi * r / testRegion.massShell - con.G * (testRegion.massNewStars + testRegion.massShell)/r + 10**8 * u.m**2 / u.s**2).to(u.km/u.s)

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
