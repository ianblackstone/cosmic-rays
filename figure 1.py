modelOne = model("All Forces", radiationPressure=True, windPressure=True, ionPressure=True)

masses = [10**4, 10**5, 10**6] * u.Msun

massRatio = 10**4 / 10**6
massRatio2 = 10**6 / 10**6

ages = modelOne.BPASSData.age[0:21]
ionRate = modelOne.BPASSData.ionRate[0:21] * massRatio
eDotWind = modelOne.BPASSData.eDotWind[0:21] * massRatio
mDotWind = modelOne.BPASSData.mDotWind[0:21] * massRatio
lBol = modelOne.BPASSData.luminosity[0:21] * massRatio

ionRate2 = modelOne.BPASSData.ionRate[0:21] * massRatio2
eDotWind2 = modelOne.BPASSData.eDotWind[0:21] * massRatio2

meanFreePath = modelOne.meanFreePath

R = 10 * u.pc

fCR = returnCRForce(R, eDotWind, meanFreePath)
fIon = returnIonForce(R, ionRate)
fWind = returnWindForce(eDotWind, mDotWind)
fRPThin, fRPThick = returnRPForce(R, lBol)

fCR2 = returnCRForce(R, eDotWind2, meanFreePath)
fIon2 = returnIonForce(R, ionRate2)

fig, ax = plt.subplots(1,2, dpi = 200, figsize = (11,4), facecolor = "white")

ax2 = ax[0].twinx()

lum = ax[0].plot(ages, lBol.to(u.Lsun), label = r"$L_{\rm bol}$")
wind = ax[0].plot(ages, eDotWind.to(u.Lsun), label = r"$\dot{E}_{\rm wind}$")
ion = ax2.plot(ages, ionRate / 10**49, "k", label = r"ionizing photon rate")

ax[1].plot(ages, fCR / fIon, label = r"CR/ion, $M_\star = 10^4\,M_\odot$")
ax[1].plot(ages, fCR2 / fIon2, label = r"CR/ion, $M_\star = 10^6\,M_\odot$")
ax[1].plot(ages, fCR / fWind, label = "CR/wind")
ax[1].plot(ages, fCR / fRPThin, label = r"CR/$RP_{\tau = 0.1}$")
ax[1].plot(ages, fCR / fRPThick, label = r"CR/$RP_{\tau = 10}$")

ax[0].set_xscale("log")
ax[0].set_yscale("log")

ax2.set_yscale("log")

ax[1].set_xscale("log")
ax[1].set_yscale("log")

ax[0].set_xlabel("Age (yr)")
ax[1].set_xlabel("Age (yr)")

ax[0].set_ylabel(r"Energy injection rate ($L_\odot$)")
ax2.set_ylabel(r"$S_{49}$")
ax[1].set_ylabel("Ratio")

lns = wind + lum + ion
labs = [l.get_label() for l in lns]
ax[0].legend(lns, labs)

ax[1].legend()

plt.tight_layout()