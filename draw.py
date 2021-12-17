import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hist, mplhep
from hist import  Hist

cont = np.zeros((360, 180))

h  = Hist(
    hist.axis.Regular(90, 0, 180, name="theta"),
    hist.axis.Regular(180, 0, 360, name="phi"),
)

h1 = Hist(
    hist.axis.Regular(90, 0, 180, name="angle"),
)




i = 0
for theta in range(0, 180, 2):
    j = 0
    for phi in range(0, 360, 2):
        h[i, j] = (1-np.sin(theta/180*np.pi)**2*np.cos(phi/180.*np.pi)**2)
        j = j + 1
    i = i+1

j = 0
for i in range(0, 180, 2):
    h1[j] = np.cos(i/180*np.pi)**2
    j += 1

fig, ax = plt.subplots()
mplhep.histplot(h1, ax=ax, yerr=0)

#h.plot2d_full(
#    main_cmap="coolwarm",
#    top_ls="--",
#    top_color="orange",
#    top_lw=1,
#    side_ls=":",
#    side_lw=1,
#    side_color="steelblue",
#)

plt.savefig("genpdf1.pdf")
plt.show()



"""

ax.set_xlabel(r"$\theta$ [deg]", fontsize=15)
ax.set_ylabel(r"$\phi$ [deg]", fontsize=15)

plt.tight_layout()

plt.savefig("genpdf.pdf")

plt.show()


"""