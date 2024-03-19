dataTable = np.array([
    ["ONC",0.046,0.8,2.8,2.7,0.062,2.2,0.94,6.7],
    ["Arches",0.2,0.4,4.3,400,9.2,650,0.40,6.0],
    ["NGC 5253",5,1.0,4.5,4000,92,30000,0.55,6.0],
    ["M82 La",40,1.4,5.0,13000,290,17000,0.44,6.4],
    ["M82 F",5.5,1.5,4.0,1700,40,1100,0.82,6.4],
    ["M82 11",3.9,1.1,4.3,1200,28,1200,0.57,6.4],
    ["M82 9",23,2.5,4.0,7300,170,2700,1.2,6.4],
    ["M82 8",4.0,1.6,3.9,1300,29,680,0.91,6.4],
    ["M82 7",22,2.7,3.9,7000,160,2200,1.4,6.4],
    ["M82 6",2.7,1.4,3.9,850,20,520,0.83,6.4],
    ["M82 k",5.7,3.0,3.2,1800,41,290,2.3,6.4],
    ["M82 m",7.3,1.4,4.3,2300,53,2000,0.62,6.4],
    ["M82 q",2.8,1.9,3.4,890,20,280,1.4,6.4],
    ["M82 3",2.7,1.5,3.7,850,20,420,1.0,6.4],
    ["M82 1a",8.6,2.1,3.8,2700,62,1100,1.1,6.4],
    ["M82 1c",5.2,1.5,4.0,1600,38,1000,0.80,6.4],
    ["M82 r",3.0,1.7,3.6,950,22,370,1.2,6.4],
    ["M82 t",2.5,1.7,3.5,790,18,290,1.2,6.4],
    ["Ant B1c",42,29,1.0,5100,120,24,34,6.6],
    ["Ant B",50,49,0.47,5500,130,9.9,71,6.6],
    ["Ant D",19,40,0.30,1500,36,3.3,68,6.7],
    ["Ant C",41,21,1.5,1300,30,18,16,6.8],
    ["Ant D1",16,11,2.0,100,2.3,6.9,5.2,7.1],
    ["Ant D2",8.0,35,0.11,1000,23,1.8,79,6.6],
    ["Ant E1",2.6,13,0.91,170,3.8,1.9,21,6.7],
    ["Ant E2",4.1,25,0.25,170,3.9,0.7,48,6.8],
    ["Ant E3",0.7,11,0.60,170,3.9,1.2,31,6.5],
    ["Ant F",7.4,9.3,1.8,220,5.0,8.9,7.3,6.8],
    ["Ant E5",26,23,1.2,200,4.7,3.2,16,7.1],
    ["Ant F2",3.5,17,0.71,440,10,2.7,32,6.6],
    ["Ant F1",15,15,1.4,74,1.7,2.5,9.2,7.1],
    ["Ant E4",6.5,36,-0.01,1100,26,1.6,94,6.5],
    ["Ant A1",5.0,15,1.0,2100,48,12,30,6.3],
    ["Ant S",32,14,1.9,210,3.9,10,6.3,7.1]])

T = 10**4 * u.K

lowerAgeLimit = 6.3
upperAgeLimit = 6.7

ages = dataTable[:,-1].astype(float)

ageMask = ages >= lowerAgeLimit
ageMask *= ages <= upperAgeLimit

dataTableReduced = dataTable[ageMask]

modelThree = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True, meanFreePath=0.001)

names = dataTableReduced[:,0]
clusterMasses = dataTableReduced[:,1].astype(float) * 10**5 * u.Msun
radii = dataTableReduced[:,2].astype(float) * u.pc
ionRate = dataTableReduced[:,4].astype(float) * 10**49 * u.s**-1

rSpan = np.linspace(min(radii),max(radii), 5000)

clusterRatio = clusterMasses/(10**6 * u.Msun)





## Standard plot points and theory line

eDotWind = modelThree.BPASSData.eDotWind[0] * clusterRatio
eDotCR = modelThree.windToCREnergyFraction * eDotWind

eDotWindTheory = modelThree.BPASSData.eDotWind[0]
eDotCRTheory = modelThree.windToCREnergyFraction * eDotWindTheory

fDiff0001 = (eDotCR * radii/(con.c * modelThree.meanFreePath)).cgs
fDiff0001Theory = (eDotCRTheory * rSpan/(con.c * modelThree.meanFreePath)).cgs

fIon = (np.sqrt(12 * math.pi * ionRate * phi * radii / alphaB) * con.k_B * T).cgs
fIonTheory = (np.sqrt(12 * math.pi * modelThree.BPASSData.ionRate[0] * phi * rSpan / alphaB) * con.k_B * T).cgs

fig, ax = plt.subplots(dpi = 200, facecolor="white")

ax.scatter(radii, (fDiff0001/fIon), s = 10, c = 'k')
ax.plot(rSpan, (fDiff0001Theory/fIonTheory), c = 'b', linestyle = "dashed")




## Simple ratio

ionRateRatio = ionRate / modelThree.BPASSData.ionRate[0]

eDotWind = modelThree.BPASSData.eDotWind[0] * ionRateRatio
eDotCR = modelThree.windToCREnergyFraction * eDotWind

fDiff0001 = (eDotCR * radii/(con.c * modelThree.meanFreePath)).cgs

fIon = (np.sqrt(12 * math.pi * ionRate * phi * radii / alphaB) * con.k_B * T).cgs

ax.scatter(radii, (fDiff0001/fIon), s = 10, c = 'r')

## Add the annotations here since they will be lower than the cluster of datapoints
for i, name in enumerate(names):
    ax.annotate(name, (radii.value[i], (fDiff0001/fIon)[i]))




## Guess at cluster age

indices = []

for i, rate in enumerate(ionRate):
    indices.append((np.abs((modelThree.BPASSData.ionRate.value * clusterRatio[i]) - rate.value)).argmin())

eDotWind = modelThree.BPASSData.eDotWind[indices] * clusterRatio
eDotCR = modelThree.windToCREnergyFraction * eDotWind

fDiff0001 = (eDotCR * radii/(con.c * modelThree.meanFreePath)).cgs

fIon = (np.sqrt(12 * math.pi * ionRate * phi * radii / alphaB) * con.k_B * T).cgs

ax.scatter(radii, (fDiff0001/fIon), s = 10, c = 'b')



ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel("Radius (pc)")
ax.set_ylabel(r"Ratio of forces $F_{\rm CR}/F_{\rm ion}$")