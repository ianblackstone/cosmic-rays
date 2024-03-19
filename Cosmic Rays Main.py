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
from hoki import load

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
tauScaleFiducial =  0.1 # pc^2 / MSun

ageFiducial = 1 # Myr
# luminosityFiducial = 10**8 # LSun
radiusFiducial = 10 # pc
radiusOldStarsFiducial = 10**4 # pc
massShellFiducial = 10**4 # MSun
massNewStarsFiducial = 10**4 # MSun
energyDotWindFiducial = 2500 * massNewStarsFiducial/10**4 # LSun ## DEPRECATED
massOldStarsFiducial = 10**4 # MSun
externalGasDensityFiducial = 1 # protons/cm^3

# Turn on or off various pressures in the model.
energyInjectionFiducial = True
advectionPressureFiducial = True
diffusionPressureFiducial = True
pionPressureFiducial = True
streamPressureFiducial = True
sweepUpMassFiducial = False
radiationPressureFiducial = False
windPressureFiducial = False
ionPressureFiducial = False
gravityFiducial = True

# alphaB recombination constant, since this is not included in astropy.
alphaB = 2.56 * 10**-13 * u.cm**3 / u.s

# BPASS file locations
ionFileFiducial = 'bpass/ionizing-bin-imf135_300.a+02.z020.dat'
colorsFileFiducial = 'bpass/colours-bin-imf135_300.a+02.z020.dat'
spectraFileFiducial = 'bpass/spectra-bin-imf135_300.a+02.z020.dat'
yieldsFileFiducial = 'bpass/yields-bin-imf135_300.z020.dat'

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
                windPressure = windPressureFiducial,
                ionPressure = ionPressureFiducial,
                externalMassScale = externalMassScaleFiducial,
                tauScale = tauScaleFiducial,
                ionFile = ionFileFiducial,
                # colorsFile = colorsFileFiducial,
                spectraFile = spectraFileFiducial,
                yieldsFile = yieldsFileFiducial,
                gravity = gravityFiducial):
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
            tauScale (Float, optional): The mean free path for photons in pc. Defaults to tauScaleFiducial.
            vAlfven (Float, optional): The Alfven velocity of the model. Given in km/s, will be converted to cgs. Defaults to vAlfvenFiducial.
            energyInjection (Boolean, optional): Boolean to enable energy injection in the model. Defaults to energyInjectionFiducial.
            advectionPressure (Boolean, optional): Boolean to enable advection in the model. Defaults to advectionPressureFiducial.
            diffusionPressure (Boolean, optional): Boolean to enable diffusion in the model. Defaults to diffusionPressureFiducial.
            pionPressure (Boolean, optional): Boolean to enable pion decay in the model. Defaults to pionPressureFiducial.
            streamPressure (Boolean, optional): Boolean to enable streaming in the model. Defaults to streamPressureFiducial.
            sweepUpMass (Boolean, optional): Boolean to enable the sweeping up of additional mass in the model. Defaults to sweepUpMassFiducial.
            externalMassScale (Float, optional): The scale factor for externally swept up mass. Defaults to externalMassScaleFiducial.
            radiationPressure (Boolean, optional): Boolean to enable radiation pressure in the model. Defaults to radiationPressureFiducial.
            windPressure (Boolean, optional): Boolean to enable wind pressure in the model. Defaults to windPressureFiducial.
            ionPressure (Boolean, optional): Boolean to enable ionized gas pressure in the model. Defaults to ionPressureFiducial.
            gravity (Boolean, optional): Boolean to enable gravity in the model. Defaults to gravityFiducial.
        """
        self.name = name
        self.meanFreePath = (meanFreePath * u.pc).cgs
        self.gasColumnHeight = (gasColumnHeight * u.pc).cgs
        self.windToCREnergyFraction = windToCREnergyFraction
        self.vWind = (vWind * u.km / u.s).cgs
        self.coverageFraction = coverageFraction
        self.eddRatio = eddRatio
        self.pionLifetime = (pionLifetime * u.yr).cgs
        self.tauScale = (tauScale * u.pc**2 / u.Msun).cgs
        self.vAlfven = (vAlfven * u.km/u.s).cgs
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
        self.windPressure = windPressure
        self.ionPressure = ionPressure
        self.gravity = gravity
        
        
        ionData = load.model_output(ionFile)
        # colorsData = load.model_output(colorsFile)
        spectraData = load.model_output(spectraFile)
        yieldsData = load.model_output(yieldsFile)
        self.BPASSData = BPASSDataSet(ionData, yieldsData, spectraData)
        
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
            tauInitial (float): The initial optical depth, calculated as (0.1 pc)^2  * massShell / r^2. This is calculated when given a model.
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
        self.electronDensity = self.massShell / (4/3*math.pi*(self.radius)**3) / con.m_p.cgs
        self.eddPressure = (con.G * self.massTotal * self.massShell / (4 * math.pi * self.radius**4)).to(u.Ba)
        self.luminosityPerMass = (luminosityPerMass * u.solLum / u.solMass).cgs
        if luminosity:
            self.luminosity = (luminosity * u.solLum).cgs
        else:
            self.luminosity = self.luminosityPerMass * self.massNewStars

        # self.luminosity = (con.c * con.G * self.massShell * (self.massShell + self.massNewStars) / (10 * u.pc)**2).cgs

    def calculateTau(self, model, r = None):
        """Calculate the initialTau for a given model or radius

        Args:
            model (model): a model object
            r (number or array, optional): The radius to use if overriding the model value. Can be an array. Defaults to None.
        """
        
        radius = model.gasColumnHeight if r == None else r * u.pc
        self.tauInitial = (model.tauScale * self.massShell / radius**2).cgs

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
        self.innerShockGasDensity = (self.mDotWind / (4 * math.pi * self.radius**2 * self.model.vWind)).cgs
        self.innerShockNumberDensity = 4 * self.innerShockGasDensity / con.m_p.cgs
        self.tPion = self.model.pionLifetime / self.innerShockNumberDensity / u.cm**3
        self.tDiff = (3*self.radius**2 / (con.c * self.model.meanFreePath)).cgs
        self.tAdv = (self.radius/self.velocity).cgs
        self.tStream = (self.radius/self.model.vAlfven).cgs
        self.energyCR = (self.pressure * 4 * math.pi * self.radius**3).cgs
        self.gammaLuminosity = (1 / 3 * self.energyCR / self.tPion).cgs

        self.dvdr, self.dpdr, _, self.dmdr = getDVDR(self.radius.cgs.value, [self.velocity.cgs.value, self.pressure.cgs.value, self.time.cgs.value, self.massShell.cgs.value], self.region, self.model, verbose = True)

        dpdrConversion = 4 * math.pi * self.radius**2/(self.massShell*self.velocity) * u.pc

        if (self.model.diffusionPressure):
            self.dvdr.diffusion = (np.cumsum(self.dpdr.diffusion) * dpdrConversion).cgs
        if (self.model.advectionPressure):
            self.dvdr.advection = (np.cumsum(self.dpdr.advection) * dpdrConversion).cgs
        if (self.model.streamPressure):
            self.dvdr.streaming = (np.cumsum(self.dpdr.streaming) * dpdrConversion).cgs
        if (self.model.energyInjection):
            self.dvdr.energyInjection = (np.cumsum(self.dpdr.energyInjection) * dpdrConversion).cgs

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

        initialForce = (self.model.eddRatio *  initialPressure * 4 * math.pi * self.model.gasColumnHeight**2).cgs

        integrationConstantDiffusion = np.power(self.model.vInitial,2).cgs - (initialForce * 4 * math.pi * self.model.gasColumnHeight / self.region.massShell).cgs + (con.G * (self.region.massNewStars + self.region.massShell)/self.model.gasColumnHeight).cgs

        analyticVelocityDiffusion = np.sqrt(initialForce * 4 * math.pi * self.radius / self.region.massShell - con.G * (self.region.massNewStars + self.region.massShell)/self.radius + integrationConstantDiffusion).to(u.km/u.s)

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

class BPASSDataSet:
    def __init__(self, ionData, yieldsData, spectraData):
        """ We need a container for BPASS data that supports units, so we have to make our own since neither hoki nor pandas support explicit units.

        Args:
            ionData (BPASS ion data): The result of hoki.model_output on a BPASS ion data file
            yieldsData (BPASS yields data): The result of hoki.model_output on a BPASS yields data file
            spectraData (BPASS Spectra data):

        Returns:
            age (float): The age in years.
            ionRate (float): The ion production rate in ions/second.
            mDotWind (float): The wind mass loss rate for the cluster in solar masses / year.
            eDotWind (float): The energy in winds in erg / second.
            eDotSN (float): The energy in supernova in erg / second.
            vWind (float): The wind velocity in cm / second, from 1/2 mv^2.
            luminosity (float): The bolometric luminosity in erg / secons
        """
        self.age = np.power(10, ionData.log_age).values * u.yr
        self.ionRate = np.power(10, ionData.prod_rate).values / u.s
        self.mDotWind = (yieldsData.H_wind + yieldsData.He_wind + yieldsData.Z_wind).values * u.Msun / u.yr
        self.eDotWind = (yieldsData.E_wind.values * u.Msun * u.m**2 / u.s**2 / u.yr).cgs
        self.eDotSN = (yieldsData.E_sn.values * u.J/u.yr).cgs
        self.vWind = np.sqrt(2 * self.eDotWind / self.mDotWind).cgs
        self.luminosity = (spectraData.drop('WL', axis = 1).sum(axis = 0).values * u.Lsun).cgs
        self.ionLuminosity = (spectraData.loc[(con.h * con.c / (spectraData.WL.values * u.angstrom)).to(u.eV) > (13.6 * u.eV)].drop('WL', axis = 1).sum(axis = 0).values * u.Lsun).cgs

    # To-Do:
    # Add method for outputting as a nice looking table.

class genericData:
    def __init__(self, name):
        """This is a generic class for any dataset.

        Args:
            name (string): The name of the dataset
        """


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


# %%
# Define ODE functions
###############################################################################
def getDVDR(rShell, X, region, model, verbose = False):
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

    if verbose:
        dpdr = type('DPDR', (), {})()
    else:
        dpdr = 0

    if (model.energyInjection or model.windPressure):
        eDotWind = (np.interp(t * u.s, model.BPASSData.age, model.BPASSData.eDotWind) * region.massNewStars / (10**6 * u.Msun)).cgs.value
        # eDotWind = (8500 * u.Lsun).cgs.value

    if model.energyInjection:
        energy = model.windToCREnergyFraction * eDotWind / (4 * math.pi * rShell**3 * vShell)

        if verbose:
            dpdr.energyInjection = energy * u.Ba / u.cm
        else:
            dpdr += energy
        
    if model.advectionPressure:
        advection = 4 * pCR / rShell

        if verbose:
            dpdr.advection = advection * u.Ba / u.cm
        else:
            dpdr -= advection

    if model.diffusionPressure:
        diffusion = con.c.cgs.value * model.meanFreePath.value * pCR / (vShell * rShell**2)
        if verbose:
            dpdr.diffusion = diffusion * u.Ba / u.cm
        else:
            dpdr -= diffusion

    if model.streamPressure:
        streaming = (3*vShell + model.vAlfven.value) * pCR / ( rShell * vShell)
        if verbose:
            dpdr.streaming = streaming * u.Ba / u.cm
        else:
            dpdr -= streaming

    if verbose:
        dvdr = type('DVDR', (), {})()
    else:
        dvdr = 0

    dvdrCR = pCR * 4 * math.pi * rShell**2/(mShell*vShell)

    if verbose:
        dvdr.CR = dvdrCR / u.s
    else:
        dvdr = dvdrCR

    if model.gravity:
        dvdrGrav = con.G.cgs.value*(mShell + region.massNewStars.value)/(vShell*rShell**2)
        if verbose:
            dvdr.CR = dvdrCR / u.s
            dvdr.gravity = dvdrGrav / u.s
        else:
            dvdr -= dvdrGrav

    dmdr = 0

    if model.radiationPressure:
        luminosity = np.interp(t * u.s, model.BPASSData.age, model.BPASSData.luminosity) * region.massNewStars / (10**6 * u.Msun)
        radiation = luminosity.cgs.value / con.c.cgs.value / vShell / mShell * (1 - np.exp(-region.tauInitial * (model.gasColumnHeight.cgs.value / rShell)**2))

        if verbose:
            dvdr.radiation = radiation / u.s
        else:
            dvdr += radiation

    if model.windPressure:
        mDotWind = np.interp(t * u.s, model.BPASSData.age, model.BPASSData.mDotWind) * region.massNewStars / (10**6 * u.Msun)
        vWind = np.sqrt(2 * eDotWind / mDotWind)

        wind = (2 * eDotWind / vWind).cgs.value / vShell / mShell
        if verbose:
            dvdr.wind = wind / u.s
        else:
            dvdr += wind

    if model.sweepUpMass:
        dmdr = 4 * math.pi * region.externalGasDensity.value * (model.gasColumnHeight.value/rShell)**model.externalMassScale * rShell**2

        mass = vShell * dmdr / mShell

        if verbose:
            dvdr.mass = mass / u.s
        else:
            dvdr -= mass

    if model.ionPressure:
        ion = (-np.sqrt(3 * np.interp(t * u.s, model.BPASSData.age, model.BPASSData.ionRate) * (region.massNewStars / (10**6 * u.Msun)) / alphaB) * con.k_B * 10**4 * u.K / (rShell * u.cm)**(3/2) / mShell / vShell).to(u.Ba).value * 4 * math.pi * rShell **2

        if verbose:
            dvdr.ion = ion / u.s
        else:
            dvdr += ion

    dtdr = 1/abs(vShell)

    return [dvdr, dpdr, dtdr, dmdr]

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

def calculateForce(model, region, rSpan = None):
    """Calculates the force balance for a given model and region's initial conditions

    Args:
        model (_type_): A model object.
        region (_type_): A region object.
        rSpan (optional, number or array of numbers): An initial column height or array of column heights to use instead of the model values. Input in parsecs.

    Returns:
        forces (genericData): An object containing the pressures present in the model.
        pressures (genericData): An object containing the pressures present in the model.
    """

    if rSpan == None:
        rSpan = [model.gasColumnHeight]
    elif type(rSpan) != list:
        rSpan = [rSpan]

    if hasattr(rSpan, 'unit') == False:
        rSpan *= u.pc

    luminosity = model.BPASSData.luminosity[0] * region.massNewStars / (10**6 * u.Msun)
    radiationForce = (luminosity.cgs / con.c.cgs * (1 - np.exp(-region.tauInitial))).cgs
    radiationPressure = (radiationForce / (4 * math.pi * np.power(rSpan,2))).cgs

    eDotWind = model.BPASSData.eDotWind[0] * region.massNewStars / (10**6 * u.Msun)
    mDotWind = model.BPASSData.mDotWind[0] * region.massNewStars / (10**6 * u.Msun)
    vWind = np.sqrt(2 * eDotWind / mDotWind)

    windForce = (2 * eDotWind / vWind).cgs
    windPressure = (windForce / (4 * math.pi * np.power(rSpan,2))).cgs

    ionRate = model.BPASSData.ionRate[0] * region.massNewStars / (10**6 * u.Msun)
    ionizedPressure = (np.sqrt(3 * ionRate / alphaB) * con.k_B * 10**4 * u.K / np.power(rSpan,3/2)).cgs
    ionizedForce = (ionizedPressure * 4 * math.pi * np.power(rSpan,2)).cgs

    diffusionPressure = (eDotWind * model.windToCREnergyFraction / (4 * math.pi * rSpan * con.c * model.meanFreePath)).cgs
    diffusionForce = (diffusionPressure * (4 * math.pi * np.power(rSpan,2))).to(u.dyn)

    gravity = (con.G * region.massTotal * region.massShell / (np.power(rSpan,2))).cgs

    totalForce = np.copy(diffusionForce)

    if model.radiationPressure:
        totalForce += radiationForce
    if model.windPressure:
        totalForce += windForce
    if model.ionPressure:
        totalForce += ionizedForce
    if model.gravity:
        totalForce -= gravity

    forces = genericData("Forces")
    forces.radiation = radiationForce
    forces.wind = windForce
    forces.ionizedGas = ionizedForce
    forces.gravity = gravity
    forces.diffusion = diffusionForce
    forces.total = totalForce

    pressures = genericData("Pressures")
    pressures.radiation = radiationPressure
    pressures.wind = windPressure
    pressures.ionizedGas = ionizedPressure
    pressures.diffusion = diffusionPressure

    return forces, pressures


def solveODE(model, region, verbose = True):
    """Returns the ODEs for a given model and region

    Args:
        model (model): A model object
        region (region): a region object.
        verbose (bool, optional): Whether to be verbose during run. Defaults to True.

    Returns:
        ODESolve: A solve_ivp object.
    """
    region.calculateTau(model)

    forces, pressures = calculateForce(model, region)
    # initialPressure = ((model.windToCREnergyFraction * region.energyDotWind) * model.gasColumnHeight / (con.c * model.meanFreePath) / (4 * math.pi * model.gasColumnHeight**2)).cgs.value
    initialPressure = pressures.diffusion[0]
    # Need to set the initial tau to allow for proper radiation pressure calculation.
    

    # if model.radiationPressure:
    #     luminosity = model.BPASSData.luminosity[0] * region.massNewStars / (10**6 * u.Msun)
    #     initialPressure -= luminosity.cgs / con.c.cgs * (1 - np.exp(-region.tauInitial)) / (4 * math.pi * model.gasColumnHeight**2)

    # if model.windPressure:
    #     eDotWind = model.BPASSData.eDotWind[0] * region.massNewStars / (10**6 * u.Msun)
    #     mDotWind = model.BPASSData.mDotWind[0] * region.massNewStars / (10**6 * u.Msun)
    #     vWind = np.sqrt(2 * eDotWind / mDotWind)

    #     initialPressure -= (2 * eDotWind / vWind / (4 * math.pi * model.gasColumnHeight**2)).cgs

    # if model.ionPressure:
    #     ionRate = model.BPASSData.ionRate[0] * region.massNewStars / (10**6 * u.Msun)
    #     initialPressure -= np.sqrt(3 * ionRate / alphaB) * con.k_B * 10**4 * u.K / (model.gasColumnHeight)**(3/2)

    # print(initialPressure)
    # initialPressure = max(initialPressure.value, 0)

    X0 = [model.vInitial.value, initialPressure.value, region.age.value, region.massShell.value]

    rSpan = (model.gasColumnHeight.value, 1000*model.gasColumnHeight.value)

    if verbose:
        print("Calculating ODE for " + str(model) + " " + str(region))

    ODESolve = inte.solve_ivp(getDVDR, rSpan, X0, method = "Radau", args=[region, model], max_step=(1 * u.pc).cgs.value, rtol = 10e-6, atol = 10e-9)

    return ODESolve

# def getTotalForce(mass, massType, radius):
#     """Returns the total force for a given mass of new stars and radius as a function of the gas mass

#     Args:
#         massStars (float): The mass of the stellar cluster to return, in log10(Msun). Do not attach units.
#         radius (float): The radius to return in pc. Do not attach units.
#     """

#     rIndex = rSpan.index(radius)

#     if massType == "stars":
#         forceList = list(filter(lambda f: f.massNewStars == mass, forces))
#     elif massType == "shell":
#         forceList = list(filter(lambda f: f.massShell == mass, forces))
#     else:
#         print("massType must be either 'stars' or 'shell'.")
#         return False

#     totalForce = np.zeros_like(forceList)

#     for i, force in enumerate(forceList):
#         totalForce[i] = force.total[rIndex].value

#     return totalForce

def getEpsSigma(reg, rSpan):
    """Returns the ratio of new stellar mass to shell mass and the gas surface density of the shell

    Args:
        reg (region): A region object

    Returns:
        eps, sigma (array): Arrays containing epsilon and sigma.
    """

    sigma = (reg.massShell / (4 * (rSpan * u.pc)**2)).cgs
    eps = (reg.massNewStars / reg.massShell).cgs * np.ones_like(rSpan)

    return eps, sigma

def scatterPlotData(mod, radii, shellMasses, clusterMasses):
    
    epsList = []
    sigmaList = []
    forceList = []

    for shMass in shellMasses:
        for clMass in clusterMasses:
            reg = region("temp", massShell=10**shMass, massNewStars=10**clMass)
            reg.calculateTau(mod, radii)

            eps, sigma = getEpsSigma(reg, radii)
            force, _ = calculateForce(mod, reg, radii)

            epsList = np.append(epsList, eps)
            sigmaList = np.append(sigmaList, sigma.value)
            forceList = np.append(forceList, force.total.value)

    return epsList, sigmaList, forceList

# %%
# Function to return the dominant CR force.

# def returnCRForce(rSpan, lambdaModel, lambdaRegion):
#     """Returns the CR force, taking the maximum of the diffusion and streaming force.

#     Args:
#         rSpan (array): Array of radii. Input in pc.
#         lambdaModel (model): The model to use.
#         lambdaRegion (region): The region to use.

#     Returns:
#         CRForce: The maximum CR force, in dyns.
#         diffusion: The diffusion force given by 3Edot_CR / c / lambda.
#         streaming: The streaming force given by Edot_CR/v_Alvfen.
#         criticalRadius: The critical radius where streaming takes over from diffusion
#     """

#     eDotCR = lambdaRegion.energyDotWind * lambdaModel.windToCREnergyFraction

#     diffusion = (3 * eDotCR * rSpan / (con.c * lambdaModel.meanFreePath)).cgs
#     streaming = (eDotCR / lambdaModel.vAlfven).cgs

#     diffusionDominatedMask = diffusion > streaming

#     CRForce = np.copy(diffusion)
#     CRForce[diffusionDominatedMask] = streaming

#     rCritIndex = np.searchsorted(diffusion, streaming, side="right")
#     criticalRadius = rSpan[rCritIndex]

#     return CRForce, criticalRadius, diffusion, streaming

def returnCRForce(radii, eDotWind, meanFreePath):

    eDotCR = eDotWind * 0.1

    diffusion = (3 * eDotCR * radii / (con.c * meanFreePath)).cgs

    return diffusion
    

def returnIonForce(radii, S):
    T = 10**4 * u.K

    if  not hasattr(S, "unit") or not hasattr(radii, "unit"):
        raise Exception("Drop your units?")
    if S.unit != u.s**-1:
        raise Exception("Ion rate has wrong units")
    if radii.unit != u.pc:
        raise Exception("Radius has wrong units")

    fIon = (np.sqrt(12 * math.pi * S * phi / alphaB * radii) * con.k_B * T).cgs

    return fIon

def returnWindForce(eDotWind, mDotWind):

    vWind = np.sqrt(2 * eDotWind / mDotWind)
    fWind = (2 * eDotWind / vWind).cgs

    return fWind

def returnRPForce(radii, lBol):

    # if len(radii) > 1:
    #     tauThin = 0.1 * radii[0]**2/radii**2
    #     tauThick = 10 * radii[0]**2/radii**2
    # else:
    tauThin = 0.1
    tauThick = 10

    fRPThin = ((1 - np.exp(-tauThin)) * lBol / con.c).cgs
    fRPThick = ((1 - np.exp(-tauThick)) * lBol / con.c).cgs

    return fRPThin, fRPThick



# %%
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

# modelOne = model(r"$\lambda_{\rm CR}$: 0.1 pc", meanFreePath = 0.1, radiationPressure = True, windPressure = True)
# modelTwo = model(r"$\lambda_{\rm CR}$: 0.1 pc", meanFreePath = 0.1, radiationPressure = True, windPressure = True, ionPressure = True)
# modelThree = model(r"$\lambda_{\rm CR}$: 0.1 pc", meanFreePath = 0.1, radiationPressure = True, windPressure = True, ionPressure = True, sweepUpMass = True)
# modelFour = model(r"$\lambda_{\rm CR}$: 0.01 pc", sweepUpMass = True, radiationPressure = True, windPressure = True)
# modelFive = model(r"$\lambda_{\rm CR}$: 0.03 pc", meanFreePath = 0.03, sweepUpMass = True, radiationPressure = True, windPressure = True)
# modelSix = model(r"$\lambda_{\rm CR}$: 0.007 pc", meanFreePath = 0.007, sweepUpMass = True, radiationPressure = True, windPressure = True)
# # modelSeven = model(r"No radiation, $\lambda_{\rm CR}$: 0.01 pc", radiationPressure = False)
# # modelEight = model(r"No radiation, $\lambda_{\rm CR}$: 0.03 pc", meanFreePath = 0.03, radiationPressure = False)
# # modelNine = model(r"No radiation, $\lambda_{\rm CR}$: 0.007 pc", meanFreePath = 0.007, radiationPressure = False)
# # modelTen = model(r"No radiation, $\lambda_{\rm CR}$: 0.01 pc", sweepUpMass = True, radiationPressure = False)
# # modelEleven = model(r"No radiation, $\lambda_{\rm CR}$: 0.03 pc", meanFreePath = 0.03, sweepUpMass = True, radiationPressure = False)
# # modelTwelve = model(r"No radiation, $\lambda_{\rm CR}$: 0.007 pc", meanFreePath = 0.007, sweepUpMass = True, radiationPressure = False)
# # modelFour = model("lambda CR: 0.07 pc", meanFreePath = 0.07)
# # modelFive = model("lambda CR: 0.1 pc", meanFreePath = 0.1)

# regionOne = region(r"MShell: $10^5$ $M_\odot$", massShell = 10**5, massNewStars = 10**4)
# regionTwo = region(r"MShell: $10^5$ $M_\odot$", massShell = 10**4)
# regionThree = region(r"MShell: $10^6$ $M_\odot$", massShell = 10**4)

# modelList = [modelOne, modelTwo, modelThree, modelFour, modelFive, modelSix]
# regionList = [regionOne,regionTwo,regionThree]

# modelList = [modelOne]
# regionList = [regionOne]

# resultList = []

# for currentModel in modelList:
#     for currentRegion in regionList:
#         currentResult = results(currentModel, currentRegion)
#         resultList.append(currentResult)

# resultList[0].multiPlot("radius", "velocity", resultList[1:-1], scale = "log")
# resultList[0].multiPlot("time", "velocity", resultList[1:-1], scale = "log")

# # for res in resultList:
# #     res.verify()


# # # %%
# # # Plot results
# # for res in resultList:
# #     res.plot("radius", "velocity", scale = "symlog")


## energyInjection = (res.model.windToCREnergyFraction * res.region.energyDotWind / (4 * math.pi * res.radius**3 * res.velocity)).cgs
## advectionPressure = (4 * res.pressure / res.radius).cgs
## streamPressure = ((3*res.velocity + res.model.vAlfven) * res.pressure / ( res.radius * res.velocity)).cgs
## diffusionPressure = (con.c.cgs * res.model.meanFreePath * res.pressure / (res.velocity * res.radius**2)).cgs

# Plots
###############################################################################

# %%
# Plot forces
###################################################

import matplotlib.colors as colors

modelAllForces = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True)

# rad = [2, 3, 5, 10, 30, 50, 100]
# clusterMasses = [3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6]
# shellMasses = [3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6]

# for r in rad:
#     epsList, sigmaList, forceList = scatterPlotData(modelAllForces, r, shellMasses, clusterMasses)

#     mask = forceList > 0

#     plt.figure(dpi = 200, facecolor = "white")
#     ax = plt.axes()

#     plt.scatter(sigmaList[mask], 1/(1+1/epsList[mask]), color="r")
#     plt.scatter(sigmaList[~mask], 1/(1+1/epsList[~mask]), color="b")

#     ax.set_facecolor("white")

#     plt.yscale('log')
#     plt.xscale('log')

#     plt.title(r)

#     plt.xlabel(r"$\Sigma_{\rm gas} (g/cm^2)$")
#     plt.ylabel(r"$1/(1+M_{\rm sh}/M_{\rm cl})$")

# %%
# Find the ratio of forces
###################################################

phi = 0.73

def forceRatio(model, radius, massNewStars, massShell = 10**4 * u.Msun, ageIndex = 0, forceNumerator = "diffusion", forceDenominator = "ionized"):
    acceptableValues = ["ionized", "diffusion", "wind", "radiation", "gravity", "streaming", "lAlpha", "ISM", "total"]

    if ~(forceNumerator in acceptableValues) or ~(forceDenominator in acceptableValues):
        ValueError("numerator or denominator not an acceptable value.")

    if forceDenominator == forceNumerator:
        ValueError("The answer is 1 by definition, do better.")

    if forceNumerator == "total" or forceDenominator == "total":
        forces = acceptableValues
    else:
        forces = [forceNumerator, forceDenominator]

    if "diffusion" in forces:
        eDotCR = model.BPASSData.eDotWind[ageIndex] * massNewStars / (10**6 * u.Msun) * model.windToCREnergyFraction

        diffusion = eDotCR * radius / (con.c * model.meanFreePath)

        if forceNumerator == "diffusion":
            numerator = diffusion
        elif forceDenominator == "diffusion":
            denominator = diffusion

    if "streaming" in forces:
        eDotCR = model.BPASSData.eDotWind[ageIndex] * massNewStars / (10**6 * u.Msun) * model.windToCREnergyFraction

        streaming = eDotCR / (10 * u.km / u.s)

        if forceNumerator == "streaming":
            numerator = streaming
        elif forceDenominator == "streaming":
            denominator = streaming

    if "lAlpha" in forces:

        lAlpha = model.BPASSData.ionLuminosity[ageIndex] * massNewStars / (10**6 * u.Msun) * 35 * 0.68 / con.c

        if forceNumerator == "lAlpha":
            numerator = lAlpha
        elif forceDenominator == "lAlpha":
            denominator = lAlpha
    
    if "ISM" in forces:
        ISM = 10**-12 * u.erg / u.cm**3 * 4 * math.pi * radius**2

        if forceNumerator == "ISM":
            numerator = ISM
        elif forceDenominator == "ISM":
            denominator == ISM

    if "ionized" in forces:
        ionRate = model.BPASSData.ionRate[ageIndex] * massNewStars / (10**6 * u.Msun)
        T = 10**4 * u.K

        ionized = con.k_B * T * np.sqrt(12 * math.pi * ionRate * phi * radius / alphaB)

        if forceNumerator == "ionized":
            numerator = ionized
        elif forceDenominator == "ionized":
            denominator = ionized

    if "wind" in forces:
        eDotWind = model.BPASSData.eDotWind[ageIndex] * massNewStars / (10**6 * u.Msun)
        mDotWind = model.BPASSData.mDotWind[ageIndex] * massNewStars / (10**6 * u.Msun)
        vWind = np.sqrt(2 * eDotWind / mDotWind)

        wind = (2 * eDotWind / vWind)

        if forceNumerator == "wind":
            numerator = wind
        elif forceDenominator == "wind":
            denominator = wind

    if "radiation" in forces:
        tau = (model.tauScale * massShell / np.power(radius,2)).cgs
        kappaIR = 5 * u.cm**2 / u.g
        tauIR = (kappaIR * massShell / (4 * math.pi * np.power(radius,2))).cgs

        luminosity = model.BPASSData.luminosity[ageIndex] * massNewStars / (10**6 * u.Msun)
        radiation = (luminosity / con.c * (1 - np.exp(-tau) + tauIR))

        if forceNumerator == "radiation":
            numerator = radiation
        elif forceDenominator == "radiation":
            denominator = radiation
    
    if "gravity" in forces:
        gravity = con.G * (massShell + massNewStars) * massShell / (np.power(radius,2))

        if forceNumerator == "gravity":
            numerator = gravity
        elif forceDenominator == "gravity":
            denominator = gravity

    if "total" in forces:
        total = diffusion + radiation + ionized + wind - gravity

        if forceNumerator == "total":
            numerator = total
        elif forceDenominator == "total":
            denominator = total

    ratio = numerator / denominator

    return ratio.cgs


# # %%
# # Plot ratios
# ###################################################

# dr = 0.1
# dm = 10**4

# rMin = 1
# rMax = 1000

# mMin = 10**4
# mMax = 10**8

# rSpan, mSpan = np.mgrid[slice(rMin,rMax + dr, dr),
#                         slice(mMin,mMax + dm, dm)]

# forceNumerator = "diffusion"
# forceDenominator = "ionized"

# mShell = 10**5 * u.Msun
# mCluster = mSpan * u.Msun

# ageIndex = 0

# ratios = forceRatio(modelAllForces, rSpan * u.pc, mCluster, mShell, ageIndex, forceNumerator, forceDenominator)

# class MidPointLogNorm(colors.LogNorm):
#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#         colors.LogNorm.__init__(self,vmin=vmin, vmax=vmax, clip=clip)
#         self.midpoint=midpoint
#     def __call__(self, value, clip=None):
#         x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
#         return np.ma.masked_array(np.interp(np.log(value), x, y))

# plt.figure(dpi = 200, facecolor = "white")

# divnorm = MidPointLogNorm(vmin=ratios.min(), midpoint=1, vmax=ratios.max())

# c = plt.pcolormesh(rSpan, mSpan, ratios, cmap = "seismic", shading = "auto", norm = divnorm)
# plt.colorbar(c)

# plt.yscale('log')
# plt.xscale('log')

# plt.xlabel("Radius (pc)")
# plt.ylabel(r"Cluster mass ($M_\odot$)")

# plt.title(f"{forceNumerator} over {forceDenominator}")


# # %% Plot forces over ionized pressure
# ###################################################

# rSpan = np.logspace(0, 3, 1000) * u.pc
# cutoffAge = 10**7 * u.yr
# ages = modelAllForces.BPASSData.age[modelAllForces.BPASSData.age < cutoffAge]

# for i, age in enumerate(ages):
#     ionRate = modelAllForces.BPASSData.ionRate[i]
#     ionLuminosity = modelAllForces.BPASSData.ionLuminosity[i]
#     eDotWind = modelAllForces.BPASSData.eDotWind[i]
#     eDotCR = modelAllForces.windToCREnergyFraction * eDotWind
#     mDotWind = modelAllForces.BPASSData.mDotWind[i]
#     luminosity = modelAllForces.BPASSData.luminosity[i]
#     T = 10**4 * u.K
#     mStarNorm = 10**6 * u.Msun
#     kappaIR = 5 * u.cm**2 / u.g
#     tau4 = (modelAllForces.tauScale * 10**4 * u.Msun / rSpan**2).cgs
#     tau5 = (modelAllForces.tauScale * 10**5 * u.Msun / rSpan**2).cgs
#     tau6 = (modelAllForces.tauScale * 10**6 * u.Msun / rSpan**2).cgs
#     tauIR4 = (kappaIR * 10**4 * u.Msun / (4 * math.pi * rSpan**2)).cgs
#     tauIR5 = (kappaIR * 10**5 * u.Msun / (4 * math.pi * rSpan**2)).cgs
#     tauIR6 = (kappaIR * 10**6 * u.Msun / (4 * math.pi * rSpan**2)).cgs
#     vAlfven = 10 * u.km / u.s
#     phi = 0.73
#     ISMPressure = 10**-12 * u.erg / u.cm**3

#     tauLyEff = 158 ##* (NHI/10^21 / u.cm**2)**(1/3) (10**4 * u.K / T)**(1/3)
#     tauLyEffMax = 35 ##* (fdg/fdg_solar)**(1/4) * (10**4 * u.K / T)**(1/4)

#     mStarCR = ((4 * math.pi * con.c * modelAllForces.meanFreePath * con.k_B * T / eDotCR)**2 * mStarNorm * 3 * ionRate * phi / alphaB * 1/rSpan).to(u.Msun)
#     mStarWind = (24 * ionRate * phi * rSpan * (math.pi * con.k_B * T)**2 * mStarNorm / (eDotWind * mDotWind * alphaB)).to(u.Msun)
#     mStarRP4 = (3 * ionRate * phi * rSpan / alphaB * (4 * math.pi * con.c * con.k_B * T / (luminosity * (1 - np.exp(-tau4) + tauIR4)))**2 * mStarNorm).to(u.Msun)
#     mStarRP5 = (3 * ionRate * phi * rSpan / alphaB * (4 * math.pi * con.c * con.k_B * T / (luminosity * (1 - np.exp(-tau5) + tauIR5)))**2 * mStarNorm).to(u.Msun)
#     mStarRP6 = (3 * ionRate * phi * rSpan / alphaB * (4 * math.pi * con.c * con.k_B * T / (luminosity * (1 - np.exp(-tau6) + tauIR6)))**2 * mStarNorm).to(u.Msun)
#     mStarStream = ((vAlfven * 4 * math.pi * con.k_B * T / eDotCR)**2 * 3 * ionRate * phi * rSpan / alphaB * mStarNorm).to(u.Msun)
#     mStarISM = ((ISMPressure/con.k_B / T)**2 * 4 * math.pi * alphaB / 3 / ionRate / phi * rSpan**3 * mStarNorm).to(u.Msun)
#     mStarLy = ((4 * math.pi * con.k_B * T * con.c / (min(tauLyEff,tauLyEffMax) * 0.68 * ionLuminosity))**2 * 3 * ionRate * rSpan / alphaB * mStarNorm).to(u.Msun)
#     mStarSelfGravity = ((3 * ionRate * phi / alphaB / con.G**2 * (con.k_B * T)**2 * rSpan**5 / mStarNorm)**(1/3)).to(u.Msun)
#     mStarGravity4 = (3 * ionRate * phi * (con.k_B * T)**2 * rSpan**5 /(alphaB * con.G**2 * (10**4 * u.Msun)**2 * mStarNorm)).to(u.Msun)
#     mStarGravity5 = (3 * ionRate * phi * (con.k_B * T)**2 * rSpan**5 /(alphaB * con.G**2 * (10**5 * u.Msun)**2 * mStarNorm)).to(u.Msun)
#     mStarGravity6 = (3 * ionRate * phi * (con.k_B * T)**2 * rSpan**5 /(alphaB * con.G**2 * (10**6 * u.Msun)**2 * mStarNorm)).to(u.Msun)

#     mStarSelfGravityShell4 = ((con.G * (10**4 * u.Msun)**2 / (con.k_B * T))**2 * alphaB / (12 * math.pi * ionRate * phi) / rSpan**5 * mStarNorm).to(u.Msun)
#     mStarSelfGravityShell5 = ((con.G * (10**5 * u.Msun)**2 / (con.k_B * T))**2 * alphaB / (12 * math.pi * ionRate * phi) / rSpan**5 * mStarNorm).to(u.Msun)
#     mStarSelfGravityShell6 = ((con.G * (10**6 * u.Msun)**2 / (con.k_B * T))**2 * alphaB / (12 * math.pi * ionRate * phi) / rSpan**5 * mStarNorm).to(u.Msun)

#     ageLabel = age / (10**6 * u.yr) * u.Myr

#     plt.figure(dpi = 200, facecolor = "white")

#     plt.title(f"Forces over ionized gas for {ageLabel:.1f}")

#     plt.plot(rSpan, mStarCR, label = "Diffusion")
#     plt.plot(rSpan, mStarWind, label = "Wind")
#     plt.plot(rSpan, mStarRP4, label = r"RP, $M_{\rm sh}=10^4$")
#     plt.plot(rSpan, mStarRP5, label = r"RP, $M_{\rm sh}=10^5$")
#     plt.plot(rSpan, mStarRP6, label = r"RP, $M_{\rm sh}=10^6$")
#     plt.plot(rSpan, mStarStream, label = "Streaming")
#     plt.plot(rSpan, mStarISM, label = "ISM")
#     plt.plot(rSpan, mStarLy, label = r"$Ly_\alpha$")
#     plt.plot(rSpan, mStarSelfGravity, label = "Self Gravity")
#     plt.plot(rSpan, mStarGravity4, 'k', label = r"Gravity, $M_{\rm sh}=10^4$")
#     plt.plot(rSpan, mStarGravity5, 'k', linestyle = "--", label = r"Gravity, $M_{\rm sh}=10^5$")
#     plt.plot(rSpan, mStarGravity6, 'k', linestyle = ":", label = r"Gravity, $M_{\rm sh}=10^6$")

#     plt.plot(rSpan, mStarSelfGravityShell4, 'grey', label = r"Self Gravity, shell, $M_{\rm sh}=10^4$")
#     plt.plot(rSpan, mStarSelfGravityShell5, 'grey', linestyle="--", label = r"Self Gravity, shell, $M_{\rm sh}=10^5$")
#     plt.plot(rSpan, mStarSelfGravityShell6, 'grey', linestyle=":", label = r"Self Gravity, shell, $M_{\rm sh}=10^6$")

#     plt.xlabel("Radius (pc)")
#     plt.ylabel(r"$M_{\rm cl} (M_\odot)$")

#     plt.xscale('log')
#     plt.yscale('log')

#     plt.xlim(rSpan[0].value, rSpan[-1].value)
#     plt.ylim(10**2, 10**9)

#     plt.legend(bbox_to_anchor=(1, 1))

# %%
# Plot radius and time dependent velocity
###################################################

# fig, ax = plt.subplots(1, 2, dpi = 200, figsize = (10,4), facecolor = "white")

# ax[0].plot(resultList[0].radius, resultList[0].velocity, c = 'blue', label = resultList[0].name)
# ax[0].plot(resultList[1].radius, resultList[1].velocity, c = 'orange', label = resultList[1].name)
# ax[0].plot(resultList[2].radius, resultList[2].velocity, c = 'green', label = resultList[2].name)
# ax[0].plot(resultList[3].radius, resultList[3].velocity, c = 'red', label = resultList[3].name)
# ax[0].plot(resultList[4].radius, resultList[4].velocity, c = 'purple', label = resultList[4].name)
# ax[0].plot(resultList[5].radius, resultList[5].velocity, c = 'brown', label = resultList[5].name)
# ax[0].plot(resultList[6].radius, resultList[6].velocity, c = 'pink', label = resultList[6].name)
# ax[0].plot(resultList[7].radius, resultList[7].velocity, c = 'olive', label = resultList[7].name)
# ax[0].plot(resultList[8].radius, resultList[8].velocity, c = 'cyan', label = resultList[8].name)
# # ax[0].plot(resultList[9].radius, resultList[9].velocity, c = 'blue', linestyle = "dashed")
# # ax[0].plot(resultList[10].radius, resultList[10].velocity, c = 'orange', linestyle = "dashed")
# # ax[0].plot(resultList[11].radius, resultList[11].velocity, c = 'green', linestyle = "dashed")
# # ax[0].plot(resultList[12].radius, resultList[12].velocity, c = 'red', linestyle = "dashed")
# # ax[0].plot(resultList[13].radius, resultList[13].velocity, c = 'purple', linestyle = "dashed")
# # ax[0].plot(resultList[14].radius, resultList[14].velocity, c = 'brown', linestyle = "dashed")
# # ax[0].plot(resultList[15].radius, resultList[15].velocity, c = 'pink', linestyle = "dashed")
# # ax[0].plot(resultList[16].radius, resultList[16].velocity, c = 'olive', linestyle = "dashed")
# # ax[0].plot(resultList[17].radius, resultList[17].velocity, c = 'cyan', linestyle = "dashed")

# ax[1].plot(resultList[0].time, resultList[0].velocity, c = 'blue', label = resultList[0].name)
# ax[1].plot(resultList[1].time, resultList[1].velocity, c = 'orange', label = resultList[1].name)
# ax[1].plot(resultList[2].time, resultList[2].velocity, c = 'green', label = resultList[2].name)
# ax[1].plot(resultList[3].time, resultList[3].velocity, c = 'red', label = resultList[3].name)
# ax[1].plot(resultList[4].time, resultList[4].velocity, c = 'purple', label = resultList[4].name)
# ax[1].plot(resultList[5].time, resultList[5].velocity, c = 'brown', label = resultList[5].name)
# ax[1].plot(resultList[6].time, resultList[6].velocity, c = 'pink', label = resultList[6].name)
# ax[1].plot(resultList[7].time, resultList[7].velocity, c = 'olive', label = resultList[7].name)
# ax[1].plot(resultList[8].time, resultList[8].velocity, c = 'cyan', label = resultList[8].name)
# # ax[1].plot(resultList[9].time, resultList[9].velocity, c = 'blue', linestyle = "dashed")
# # ax[1].plot(resultList[10].time, resultList[10].velocity, c = 'orange', linestyle = "dashed")
# # ax[1].plot(resultList[11].time, resultList[11].velocity, c = 'green', linestyle = "dashed")
# # ax[1].plot(resultList[12].time, resultList[12].velocity, c = 'red', linestyle = "dashed")
# # ax[1].plot(resultList[13].time, resultList[13].velocity, c = 'purple', linestyle = "dashed")
# # ax[1].plot(resultList[14].time, resultList[14].velocity, c = 'brown', linestyle = "dashed")
# # ax[1].plot(resultList[15].time, resultList[15].velocity, c = 'pink', linestyle = "dashed")
# # ax[1].plot(resultList[16].time, resultList[16].velocity, c = 'olive', linestyle = "dashed")
# # ax[1].plot(resultList[17].time, resultList[17].velocity, c = 'cyan', linestyle = "dashed")

# ax[0].set_ylim(0.1, 200)
# ax[1].set_ylim(0.1, 200)
# # ax[0].set_xlim(10,100)
# # ax[1].set_xlim(10**6,5*10**6)

# ax[0].legend()
# # ax[1].legend()

# ax[0].set_xlabel(f'Radius ({resultList[0].radius.unit})')
# ax[1].set_xlabel(f'Time ({resultList[0].time.unit})')
# ax[0].set_ylabel(f'Velocity ({resultList[0].velocity.unit})')

# ax[0].set_xscale('log')
# ax[1].set_xscale('log')
# ax[0].set_yscale('log')
# ax[1].set_yscale('log')

# %%
# Rough code for comparing pressure magnitudes
###################################################
# r = np.logspace(0,3,500) * u.pc

# lam001 = 0.01 * u.pc
# lam01 = 0.1 * u.pc

# mshell2 = 10**2 * u.Msun
# mshell3 = 10**3 * u.Msun

# tau2 = 0.1 * u.pc**2 * mshell2 / u.Msun / r**2
# tau3 = 0.1 * u.pc**2 * mshell3 / u.Msun / r**2

# mstar = 10**2 * u.Msun

# lum = 1500 * mstar * u.Lsun / u.Msun

# grav2 = con.G * mshell2 * (mshell2 + mstar) / r**2
# grav3 = con.G * mshell3 * (mshell3 + mstar) / r**2

# eps = 2*10**-5

# wind =  lum/con.c

# CR001 = 3 * eps * r/ lam001 * wind
# CR01  = 3 * eps * r/ lam01 * wind

# rad2 = (1-np.exp(-tau2)) * wind
# rad3 = (1-np.exp(-tau3)) * wind

# plt.figure(dpi = 200, facecolor = 'white')

# plt.plot(r, rad2/grav2, 'k', label = "Radiation")
# plt.plot(r, CR001/grav2, 'b', label = r"CR ($\lambda = 0.01$ pc)")
# plt.plot(r, CR01/grav2, 'b--', label = r"CR ($\lambda = 0.1$ pc)")
# plt.plot(r, wind/grav2, 'r', label = "Wind")

# plt.xscale('log')
# plt.yscale('log')

# plt.xlabel('Radius (pc)')
# plt.ylabel(r'Force / $GMM_{\rm tot}/r^2$')

# plt.title(r'Force / Gravity for $M_{\rm sh} = 10^2\,M_\odot$')
# plt.legend()

# plt.show()

# plt.figure(dpi = 200, facecolor = 'white')

# plt.plot(r, rad3/grav3, 'k', label = "Radiation")
# plt.plot(r, CR001/grav3, 'b', label = r"CR ($\lambda = 0.01$ pc)")
# plt.plot(r, CR01/grav3, 'b--', label = r"CR ($\lambda = 0.1$ pc)")
# plt.plot(r, wind/grav3, 'r', label = "Wind")

# plt.xscale('log')
# plt.yscale('log')

# plt.xlabel('Radius (pc)')
# plt.ylabel(r'Force / $GMM_{\rm tot}/r^2$')

# plt.title(r'Force / Gravity for $M_{\rm sh} = 10^3\,M_\odot$')
# plt.legend()

# plt.show()
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

# energyInjection = (proposalResultsOne.model.windToCREnergyFraction * proposalResultsOne.region.energyDotWind / (4 * math.pi * proposalResultsOne.radius**3 * proposalResultsOne.velocity)).cgs

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

# initialForce = (testRegion.eddPressure * 4 * math.pi * testRegion.radius**2).to(u.N)

# analyticVelocity = np.sqrt(initialForce * 4 * math.pi * r / testRegion.massShell - con.G * (testRegion.massNewStars + testRegion.massShell)/r + 10**8 * u.m**2 / u.s**2).to(u.km/u.s)

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
##             region.energyDotWind / (4 * math.pi * r**3 * v)

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

# initialForce = (region.eddPressure * 4 * math.pi * region.radius**2).to(u.N)

# analyticPressure =  initialForce / (4*math.pi*r**2)

# force = (p * 4 * math.pi * r**2).cgs
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

# # %%
# ## Figure 1 - timescale comparison
# modelOne = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True)
# modelTwo = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=0.3)

# regionOne = region("10^4 Mcl 10^5 Mshell", massNewStars=10**4, massShell=10**5)

# rSpan = np.logspace(1,3,50000) * u.pc

# mDotWind = (2 * regionOne.energyDotWind / modelOne.vWind**2).cgs

# innerShockGasDensity = (mDotWind / (4 * math.pi * rSpan**2 * modelOne.vWind)).cgs
# innerShockNumberDensity = 4 * innerShockGasDensity / con.m_p.cgs

# vEsc = np.sqrt(2 * con.G * (regionOne.massShell + regionOne.massNewStars)/ rSpan).cgs

# tPion = modelOne.pionLifetime / innerShockNumberDensity / u.cm**3
# tDiff001 = (3*rSpan**2 / (con.c * modelOne.meanFreePath)).cgs
# tDiff03 = (3*rSpan**2 / (con.c * modelTwo.meanFreePath)).cgs
# tAdv = (rSpan/vEsc).cgs
# tStream = (rSpan/modelOne.vAlfven).cgs

# plt.figure(dpi = 200, facecolor="white")

# plt.plot(rSpan, tDiff001.to(u.Myr), label=r"Diff $\lambda = 0.01$ pc")
# plt.plot(rSpan, tDiff03.to(u.Myr), label=r"Diff $\lambda = 0.3$ pc")
# plt.plot(rSpan, tAdv.to(u.Myr), label=r"Advection ($v = v_{\rm esc}$)")
# plt.plot(rSpan, tPion.to(u.Myr), label="Pion losses")
# plt.plot(rSpan, tStream.to(u.Myr), label="Streaming")

# plt.xscale('log')
# plt.yscale('log')

# plt.xlabel("Radius (pc)")
# plt.ylabel("Time (Myr)")

# plt.legend()
    
# # %%
# # Figure 3 - CR to RP force ratio

# modelOne = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True)
# modelTwo = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=0.3)

# regionOne = region("10^4 Mcl 10^5 Mshell", massNewStars=10**4, massShell=10**5)
  
# rSpan = np.logspace(1,3,50000) * u.pc

# clusterRatio = regionOne.massNewStars/(10**6 * u.Msun)

# eDotWind = modelOne.BPASSData.eDotWind[0] * clusterRatio
# eDotCR = modelOne.windToCREnergyFraction * eDotWind

# tauThinConst = 0.1
# tauThickConst = 10
# tauThin = 0.1 * rSpan[0]**2/rSpan**2
# tauThick = 10 * rSpan[0]**2/rSpan**2

# lBol = regionOne.luminosity

# fDiff001 = (eDotCR * rSpan/(con.c * modelOne.meanFreePath)).cgs
# fDiff03 = (eDotCR * rSpan/(con.c * modelTwo.meanFreePath)).cgs

# fRPThinConstant = ((1 - np.exp(-tauThinConst)) * lBol / con.c).cgs
# fRPThickConstant = ((1 - np.exp(-tauThickConst)) * lBol / con.c).cgs
# fRPThin = ((1 - np.exp(-tauThin)) * lBol / con.c).cgs
# fRPThick = ((1 - np.exp(-tauThick)) * lBol / con.c).cgs

# fig, (ax1) = plt.subplots(1, 1, dpi = 200, facecolor="white")

# ax1.plot(rSpan, fDiff001/fRPThinConstant, 'k', label = r"Constant $\tau_{\rm RP}$, optically thin")
# ax1.plot(rSpan, fDiff001/fRPThickConstant, 'k--', label = r"Constant $\tau_{\rm RP}$, optically thick")
# ax1.plot(rSpan, fDiff001/fRPThin, 'b', label = r"Decreasing $\tau_{\rm RP}$, optically thin")
# ax1.plot(rSpan, fDiff001/fRPThick, 'b--', label = r"Decreasing $\tau_{\rm RP}$, optically thick")

# # ax2.plot(rSpan, fDiff03/fRPThinConstant, 'k', label = r"Constant $\tau_{\rm RP}$, optically thin")
# # ax2.plot(rSpan, fDiff03/fRPThickConstant, 'k--', label = r"Constant $\tau_{\rm RP}$, optically thick")
# # ax2.plot(rSpan, fDiff03/fRPThin, 'b', label = r"Decreasing $\tau_{\rm RP}$, optically thin")
# # ax2.plot(rSpan, fDiff03/fRPThick, 'b--', label = r"Decreasing $\tau_{\rm RP}$, optically thick")

# ax1.legend()

# ax1.set_xscale('log')
# ax1.set_yscale('log')

# # ax2.set_xscale('log')
# # ax2.set_yscale('log')

# # ax2.set_yticks([])

# ax1.set_ylim(0.001,10**6)
# # ax2.set_ylim(0.001,10**6)

# ax1.set_xlabel("Radius (pc)")
# ax1.set_ylabel(r"Ratio of forces $F_{\rm CR}/F_{\rm RP}$")

# # ax2.set_xlabel("Radius (pc)")
# # %%
# # CR to ionized force ratio

# modelOne = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True)
# modelTwo = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=0.3)
# modelThree = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=0.001)
# modelFour = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=0.1)
# modelFive = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=1)

# regionOne = region("10^4 Mcl 10^5 Mshell", massNewStars=10**6, massShell=10**5)

# rSpan = np.logspace(1,4,50000) * u.pc

# clusterRatio = regionOne.massNewStars/(10**6 * u.Msun)

# eDotWind = modelOne.BPASSData.eDotWind[0] * clusterRatio
# eDotCR = modelOne.windToCREnergyFraction * eDotWind

# fDiff0001 = (eDotCR * rSpan/(con.c * modelThree.meanFreePath)).cgs
# fDiff001 = (eDotCR * rSpan/(con.c * modelOne.meanFreePath)).cgs
# fDiff01 = (eDotCR * rSpan/(con.c * modelFour.meanFreePath)).cgs
# fDiff03 = (eDotCR * rSpan/(con.c * modelTwo.meanFreePath)).cgs
# fDiff1 = (eDotCR * rSpan/(con.c * modelFive.meanFreePath)).cgs

# fIon = (np.sqrt(12 * math.pi * modelOne.BPASSData.ionRate[0] * clusterRatio * phi * rSpan / alphaB) * con.k_B * T).cgs

# plt.figure(dpi = 200, facecolor="white")

# plt.plot(rSpan, fDiff0001/fIon, 'k', label = r"$\lambda = 0.001$ pc")
# plt.plot(rSpan, fDiff001/fIon, 'b', label = r"$\lambda = 0.01$ pc")
# plt.plot(rSpan, fDiff01/fIon, 'r', label = r"$\lambda = 0.1$ pc")
# plt.plot(rSpan, fDiff03/fIon, 'g', label = r"$\lambda = 0.3$ pc")
# plt.plot(rSpan, fDiff1/fIon, 'c', label = r"$\lambda = 1$ pc")

# plt.legend()

# plt.xscale('log')
# plt.yscale('log')

# plt.xlabel("Radius (pc)")
# plt.ylabel(r"Ratio of forces $F_{\rm CR}/F_{\rm ion}$")
# # %%
# ## Plot force ratios using the better of two 

# modelOne = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True)
# modelTwo = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=0.3)
# modelThree = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=0.001)
# modelFour = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=0.1)
# modelFive = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=1)

# regionOne = region("10^4 Mcl 10^5 Mshell", massNewStars=10**6, massShell=10**5)

# rSpan = np.logspace(1,4,50000) * u.pc

# clusterRatio = (regionOne.massNewStars/(10**6 * u.Msun)).cgs

# tauThin = 0.1 * rSpan[0]**2/rSpan**2
# tauThick = 10 * rSpan[0]**2/rSpan**2

# lBol = regionOne.luminosity.cgs
# eDotWind = (modelOne.BPASSData.eDotWind[0] * clusterRatio).cgs

# fCR0001, rCritical0001, diffusion0001, _  = returnCRForce(rSpan, modelThree, regionOne)
# fCR001, rCritical001, diffusion001, _  = returnCRForce(rSpan, modelOne, regionOne)
# fCR01, rCritical01, diffusion01, _  = returnCRForce(rSpan, modelFour, regionOne)
# fCR03, rCritical03, diffusion03, _  = returnCRForce(rSpan, modelTwo, regionOne)

# fRPThin = ((1 - np.exp(-tauThin)) * lBol / con.c).cgs
# fRPThick = ((1 - np.exp(-tauThick)) * lBol / con.c).cgs

# fWind = np.sqrt(2 * eDotWind * modelOne.BPASSData.mDotWind[0]).cgs

# fIon = (np.sqrt(12 * math.pi * modelOne.BPASSData.ionRate[0] * clusterRatio * phi * rSpan / alphaB) * con.k_B * T).cgs

# one = np.ones_like(rSpan)

# fig, ax = plt.subplots(2,2, dpi = 200, facecolor="white")

# ax[0,0].plot(rSpan, fCR0001/fIon)
# ax[0,0].plot(rSpan, fCR0001/fWind)
# ax[0,0].plot(rSpan, fCR0001/fRPThin)
# ax[0,0].plot(rSpan, fCR0001/fRPThick)

# ax[0,1].plot(rSpan, fCR001/fIon)
# ax[0,1].plot(rSpan, fCR001/fWind)
# ax[0,1].plot(rSpan, fCR001/fRPThin)
# ax[0,1].plot(rSpan, fCR001/fRPThick)

# ax[1,0].plot(rSpan, fCR01/fIon)
# ax[1,0].plot(rSpan, fCR01/fWind)
# ax[1,0].plot(rSpan, fCR01/fRPThin)
# ax[1,0].plot(rSpan, fCR01/fRPThick)

# ax[1,1].plot(rSpan, fCR03/fIon)
# ax[1,1].plot(rSpan, fCR03/fWind)
# ax[1,1].plot(rSpan, fCR03/fRPThin)
# ax[1,1].plot(rSpan, fCR03/fRPThick)

# ax[0,0].plot(rSpan, one, 'k--', alpha = 0.5)
# ax[0,1].plot(rSpan, one, 'k--', alpha = 0.5)
# ax[1,0].plot(rSpan, one, 'k--', alpha = 0.5)
# ax[1,1].plot(rSpan, one, 'k--', alpha = 0.5)

# # ax[0,0].legend(bbox_to_anchor=(2.5, 1))

# ax[0,0].set_xscale('log')
# ax[0,0].set_yscale('log')

# ax[0,1].set_xscale('log')
# ax[0,1].set_yscale('log')

# ax[1,0].set_xscale('log')
# ax[1,0].set_yscale('log')

# ax[1,1].set_xscale('log')
# ax[1,1].set_yscale('log')

# # ax[0,1].set_yticks([])
# # ax[1,1].set_yticks([])

# # ax[0,0].set_ylim(0.001,10**6)
# # ax[0,1].set_ylim(0.001,10**6)

# ax[0,0].set_ylabel("Ratio of forces")
# ax[1,0].set_ylabel("Ratio of forces")

# ax[1,0].set_xlabel("Radius (pc)")
# ax[1,1].set_xlabel("Radius (pc)")

# # %%
# # Figure 2 - CR to RP force ratio

# modelOne = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True)
# modelTwo = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=0.3)

# regionOne = region("10^4 Mcl 10^5 Mshell", massNewStars=10**4, massShell=10**5)
# regionTwo = region("10^4 Mcl 10^5 Mshell", massNewStars=10**6, massShell=10**6)
  
# rSpan = np.logspace(1,3,50000) * u.pc

# clusterRatio = regionOne.massNewStars/(10**6 * u.Msun)

# eDotWind = modelOne.BPASSData.eDotWind[0] * clusterRatio
# eDotCR = modelOne.windToCREnergyFraction * eDotWind

# fDiff001 = (eDotCR * rSpan/(con.c * modelOne.meanFreePath)).cgs
# fDiff03 = (eDotCR * rSpan/(con.c * modelTwo.meanFreePath)).cgs

# fIon = (np.sqrt(12 * math.pi * modelOne.BPASSData.ionRate[0] * clusterRatio * phi * rSpan / alphaB) * con.k_B * T).cgs

# fig, (ax1) = plt.subplots(1, 1, dpi = 200, facecolor="white")

# ax1.plot(rSpan, fDiff001/fIon, 'k', label = r"$M_{\rm star} = 10^4, \lambda = 0.001$ pc")
# ax1.plot(rSpan, fDiff03/fIon, 'k--', label = r"$M_{\rm star} = 10^4, \lambda = 0.03$ pc")

# clusterRatio = regionTwo.massNewStars/(10**6 * u.Msun)

# eDotWind = modelOne.BPASSData.eDotWind[0] * clusterRatio
# eDotCR = modelOne.windToCREnergyFraction * eDotWind

# fDiff001 = (eDotCR * rSpan/(con.c * modelOne.meanFreePath)).cgs
# fDiff03 = (eDotCR * rSpan/(con.c * modelTwo.meanFreePath)).cgs

# fIon = (np.sqrt(12 * math.pi * modelOne.BPASSData.ionRate[0] * clusterRatio * phi * rSpan / alphaB) * con.k_B * T).cgs

# ax1.plot(rSpan, fDiff001/fIon, 'b', label = r"$M_{\rm star} = 10^6, \lambda = 0.001$ pc")
# ax1.plot(rSpan, fDiff03/fIon, 'b--', label = r"$M_{\rm star} = 10^6, \lambda = 0.03$ pc")

# ax1.legend()

# ax1.set_xscale('log')
# ax1.set_yscale('log')

# ax1.set_ylim(0.0001,10)

# ax1.set_xlabel("Radius (pc)")
# ax1.set_ylabel(r"Ratio of forces $F_{\rm CR}/F_{\rm ion}$")

# # %%
# modelOne = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True)
# modelTwo = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=0.3)

# regionOne = region("10^4 Mcl 10^5 Mshell", massNewStars=10**4, massShell=10**5)

  
# rSpan = np.logspace(1,3,50000) * u.pc

# clusterRatio = regionOne.massNewStars/(10**6 * u.Msun)

# eDotWind = modelOne.BPASSData.eDotWind[0] * clusterRatio
# eDotCR = modelOne.windToCREnergyFraction * eDotWind
# mDotWind = modelOne.BPASSData.mDotWind[0] * clusterRatio

# fDiff001 = (eDotCR * rSpan/(con.c * modelOne.meanFreePath)).cgs
# fDiff03 = (eDotCR * rSpan/(con.c * modelTwo.meanFreePath)).cgs

# fWind = np.sqrt(2 * eDotWind * mDotWind).cgs

# fig, (ax1) = plt.subplots(1, 1, dpi = 200, facecolor="white")

# ax1.plot(rSpan, fDiff001/fWind, 'k', label = r"$\lambda = 0.001$ pc")
# ax1.plot(rSpan, fDiff03/fWind, 'k--', label = r"$\lambda = 0.03$ pc")

# ax1.set_xscale('log')
# ax1.set_yscale('log')

# ax1.legend()

# ax1.set_ylim(0.001,20)

# ax1.set_xlabel("Radius (pc)")
# ax1.set_ylabel(r"Ratio of forces $F_{\rm CR}/F_{\rm wind}$")

# %%
