fig, ax1 = plt.subplots(dpi = 200, facecolor = "white")
ax2 = ax1.twinx()

ax1.plot(BPASSData.age, (BPASSData.luminosity).cgs / 10**35, label = "Luminosity")
ax1.plot(BPASSData.age, (BPASSData.eDotWind).cgs / 10**35, label = r"$\dot{E}_{\rm wind}$")
ax1.plot(BPASSData.age, (BPASSData.eDotSN).cgs / 10**35, label = r"$\dot{E}_{\rm SN}$")
ax2.plot(BPASSData.age, (BPASSData.ionRate).cgs / 10**35, 'k--', label = "Ionizing photon rate")

ax1.set_xlim(10**6, 10**9)
ax1.set_ylim(10**0, 10**8)
ax2.set_ylim(10**12, 10**18)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')

ax1.legend()
ax2.legend(loc = 5)

ax1.set_xlabel("Age (years)")
ax1.set_ylabel(r"$\times 10^{35}$ (erg/s)")
ax2.set_ylabel(r"$\times 10^{35}$ (number / s)")










plt.figure(dpi = 200, facecolor = "white")

plt.plot(BPASSData.age, (2 * BPASSData.eDotWind / BPASSData.vWind).cgs / 10**20, label = r"$\dot{P}_{\rm wind}$")
plt.plot(BPASSData.age, (BPASSData.luminosity / con.c).cgs / 10**20, label = "L/c")

plt.xlim(10**6, 10**9)
plt.ylim(10**8, 10**13)

plt.xscale('log')
plt.yscale('log')

plt.legend()

plt.xlabel("Age (years)")
plt.ylabel(r"$\times 10^{20}$ (dyn)")