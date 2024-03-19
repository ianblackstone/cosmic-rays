import pandas as pd

lopezData = pd.read_csv("lopez14.csv")

modelOne = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True)

regionOne = region("10^4 Mcl 10^5 Mshell", massNewStars=10**6, massShell=10**5)

T = 10**4 * u.K
mStarNorm = 10**6 * u.Msun
phi = 0.73

cmap = plt.cm.copper

colorList = cmap(np.linspace(0,1,11))

radii = lopezData.radius.values * u.pc

pressure_direct = lopezData.pressure_direct.values * u.dyn * u.cm**-2
pressure_IR = lopezData.pressure_IR.values * u.dyn * u.cm**-2
pressure_Hii = lopezData.pressure_Hii.values * u.dyn * u.cm**-2
pressure_x = lopezData.pressure_x.values * u.dyn * u.cm**-2

rSpan = np.linspace(min(radii), max(radii))

clusterRatio = 0.1

sIonLopez = np.power(10,lopezData.logS.values) * u.s**-1

massRatioLopez1 = sIonLopez / modelOne.BPASSData.ionRate[0]
massRatioLopez3 = sIonLopez / modelOne.BPASSData.ionRate[5]
massRatioLopez5 = sIonLopez / modelOne.BPASSData.ionRate[7]
massRatioLopez10 = sIonLopez / modelOne.BPASSData.ionRate[10]

fIonOne = (np.sqrt(12 * math.pi * modelOne.BPASSData.ionRate[0] * clusterRatio * phi * rSpan / alphaB) * con.k_B * T).cgs

eDotWind1 = modelOne.BPASSData.eDotWind[0] * massRatioLopez1
eDotCR1 = modelOne.windToCREnergyFraction * eDotWind1

eDotWind3 = modelOne.BPASSData.eDotWind[5] * massRatioLopez3
eDotCR3 = modelOne.windToCREnergyFraction * eDotWind3

eDotWind5 = modelOne.BPASSData.eDotWind[7] * massRatioLopez5
eDotCR5 = modelOne.windToCREnergyFraction * eDotWind5

eDotWind10 = modelOne.BPASSData.eDotWind[10] * massRatioLopez10
eDotCR10 = modelOne.windToCREnergyFraction * eDotWind10

fDiff1 = (3 * eDotCR1 * radii/(con.c * modelOne.meanFreePath)).cgs
fDiff3 = (3 * eDotCR3 * radii/(con.c * modelOne.meanFreePath)).cgs
fDiff5 = (3 * eDotCR5 * radii/(con.c * modelOne.meanFreePath)).cgs
fDiff10 = (3 * eDotCR10 * radii/(con.c * modelOne.meanFreePath)).cgs

pOverK1 = (fDiff1 / (4 * math.pi * radii**2) / con.k_B).cgs
pOverK3 = (fDiff3 / (4 * math.pi * radii**2) / con.k_B).cgs
pOverK5 = (fDiff5 / (4 * math.pi * radii**2) / con.k_B).cgs
pOverK10 = (fDiff10 / (4 * math.pi * radii**2) / con.k_B).cgs

plt.figure(dpi = 200, facecolor = "white")

plt.scatter(radii, (pressure_direct / con.k_B).cgs,   s = 15, c = 'b',    marker = "o", label = r"$P_{\rm dir}$")
plt.scatter(radii, (pressure_IR / con.k_B).cgs,       s = 15, c = 'r',    marker = "d", label = r"$P_{\rm IR}$")
plt.scatter(radii, (pressure_Hii / con.k_B).cgs,      s = 15, c = 'y', marker = "*", label = r"$P_{\rm Hii}$")
plt.scatter(radii, (pressure_x / con.k_B).cgs,        s = 15, c = 'cyan',    marker = "s", label = r"$P_{\rm X}$")

for i, _ in enumerate(pOverK1):
	plt.plot([radii[i].value, radii[i].value], [pOverK1[i].value, pOverK5[i].value], color = "grey", alpha = 0.5)

plt.scatter(radii.value, pOverK3,                    s = 15, c = 'k',    marker = "x", label = r"$P_{\rm diff}$")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("R (pc)")
plt.ylabel(r"$P/k_B$ (K $cm^{-3}$)")

plt.legend()