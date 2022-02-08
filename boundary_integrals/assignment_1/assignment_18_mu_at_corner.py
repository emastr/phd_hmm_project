import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from figure_tools.figure_tools import *
from boundary_integrals.lambda_ops import *
from scipy.sparse.linalg import gmres
import matplotlib.cm as cm
from boundary_integrals.gaussleggrid import GaussLegGrid

L = 0.5
eps = 0.3

# Types: 1 - square, 2 - circle

boundary = 3
plot_grid = False

if boundary == 1:
    tau = lambda t: np.where(t <= 2, np.where(t <= 1, -np.ones_like(t)  + 1j * (1 - t * 2), ((t - 1) * 2 - 1) - 1j*np.ones_like(t)),\
                                      np.where(t <= 3, np.ones_like(t) + 1j * ((t - 2) * 2 - 1), ((4 - t)*2 - 1) + 1j*np.ones_like(t)))
    dtaudt = lambda t: np.where(t <= 2, np.where(t <= 1, -1j * 2 * np.ones_like(t), 2 * np.ones_like(t)),\
                                         np.where(t <= 3, 1j * 2 * np.ones_like(t), - 2 * np.ones_like(t)))
    ddtauddt = lambda t: np.zeros_like(t)
elif boundary == 2:
    tau = lambda t: np.cos(np.pi * t / 2) + 1j * np.sin(np.pi * t / 2)
    dtaudt = lambda t: np.pi / 2 * (-np.sin(np.pi * t / 2) + 1j * np.cos(np.pi * t / 2))
    ddtauddt = lambda t: (np.pi / 2)**2 * (-np.cos(np.pi * t / 2) - 1j * np.sin(np.pi * t / 2))
elif boundary == 3:
    tau = lambda t: np.sin(2*np.pi*t/4) + 2 * 1j * np.sin(np.pi * t / 4) - 1j
    dtaudt = lambda t: np.cos(2*np.pi*t/4)*2*np.pi/4 + 1j * 2* np.pi/4 * np.cos(np.pi * t / 4)
    ddtauddt = lambda t: -np.sin(2*np.pi*t/4)*(2*np.pi/4)**2 - 2*1j * (np.pi/4)**2*np.sin(np.pi*t/4)
else:
    p = boundary
    curve = lambda t, p: (1 - np.abs(t) ** p) ** (1/p)
    curvep = lambda t, p: -t * np.abs(t) ** (p-2) * (1 - np.abs(t) ** p) ** (1/p - 1)
    curvepp = lambda t, p: (1-p) * np.abs(t) ** (p-2) * (1 - np.abs(t) ** p) ** (1/p - 1)

    tau = lambda t: np.where(t <= 2, (1-t) + 1j * curve(1 - t, p), (t-3) - 1j * curve(t - 3, p))
    dtaudt = lambda t: np.where(t <=2, -1 - 1j * curvep(1 - t, p), 1 - 1j * curve(t-3, p))
    ddtauddt = lambda t: np.where(t <= 2, 1j * curvep(1 - t, p),  -1j * curve(t - 3, p))

def mask_fun(X, Y):
    if boundary == 1:
        mask = (X <= 1-eps) * (-1+eps <= X) * (Y <= 1-eps) * (-1+eps <= Y)
    elif boundary == 2:
        mask = (X ** 2 + Y ** 2) <= 1 -eps
    elif boundary == 3:
        #T = np.arctan2(Y.flatten(), X.flatten()).reshape((n_grid, n_grid))
        #mask = ((X ** 2 + Y ** 2) <= (np.sin(T) * (1 + np.cos(T)**2)**0.5 - eps)**2)
        mask = (X < (1-((Y+1-eps)/2/(1-eps))**2)**0.5 * (Y+1-eps)) * (X > -(1-((Y+1-eps)/2/(1-eps))**2)**0.5 * (Y+1-eps))
    else:
        p = boundary
        #curve = lambda t, p: (1 - np.abs(t) ** p) ** (1 / p)
        #curvep = lambda t, p: -t * np.abs(t) ** (p - 2) * (1 - np.abs(t) ** p) ** (1 / p - 1)
        #curvepp = lambda t, p: (1 - p) * np.abs(t) ** (p - 2) * (1 - np.abs(t) ** p) ** (1 / p - 1)
        mask = (np.abs(X) ** p + np.abs(Y) ** p) ** (1/p) <= 1 - eps
    return mask


zp = -1.3 + 1.3j
F = lambda z: np.imag(z ** 2 / (z - zp))
f = lambda t: F(tau(t))


def solve(t, weights, with_gmres=True, **kwargs):
    n_pts = len(t)
    tau_t = tau(t)
    dtaudt_t = dtaudt(t)
    ddtauddt_t = ddtauddt(t)
    f_t = f(t)

    K = np.zeros((n_pts, n_pts))

    for n in range(n_pts):
        for m in range(n_pts):
            if m != n:
                K[n, m] = np.imag(dtaudt_t[m] / (tau_t[m] - tau_t[n]) / np.pi * weights[m])
            else:
                K[n, m] = np.imag(ddtauddt_t[m] / 2 / dtaudt_t[m] / np.pi * weights[m])

    if with_gmres:
        mu, info = gmres(np.eye(n_pts) + K, f_t, **kwargs)
    else:
        mu = np.linalg.solve(np.eye(n_pts) + K, f_t)
        info = None
    #eigvals, eigvecs = np.linalg.eig(np.eye(n_pts) + K)
    #plt.scatter(np.real(eigvals), np.imag(eigvals))
    #mu = f_t
    # U function
    u = lambda z: np.dot(np.imag(dtaudt_t * weights/(tau_t - z) / np.pi), mu)
    return u, mu, np.eye(n_pts) + K, info

segments = np.linspace(0, 4, 17)
if boundary == 1:
    sharp_corners = np.isin(segments, np.array([0.,1.,2.,3.,4.])).astype(int)
if boundary == 2:
    #sharp_corners = np.isin(segments, np.array([0., 1., 2., 3., 4.])).astype(int)
    sharp_corners = np.isin(segments, np.array([])).astype(int)
if boundary == 3:
    sharp_corners = np.isin(segments, np.array([0., 4.])).astype(int)

mu0 = -0.2899882962966718

cmap=plt.get_cmap("prism")#, int(cols[-1]))
for i in range(1,20,5):
    #i=40
    gridObject = GaussLegGrid(segments, sharp_corners)
    gridObject.refine_corners_nply(i)

    gridpts, weights = gridObject.get_grid_and_weights()
    print(len(gridpts))

    u, mu, _, _ = solve(gridpts, weights, tol=1e-16)

    #istart = 150
    #iend = 200
    #plt.scatter(gridpts[istart:iend], mu[istart:iend] - mu[istart])
    #plt.yscale("log")
    #plt.xscale("log")

    #plt.plot(gridpts, mu - mu0)
    cols = np.array(list(range(len(gridpts))))
    cols = np.round(cols/16)
    #plt.plot(cols)
    #idx=200
    plt.plot(gridpts, mu)
    #plt.scatter(gridpts, np.abs(mu),c=cols/cols[-1], cmap=cmap)

idx0=10
idx = 20
#plt.scatter(gridpts[idx:], mu[idx:],c=cols[idx:]/cols[-1], cmap=cmap)
yvals = np.abs(mu-mu[idx0])
a = 0.4
p = 0.67
plt.plot(gridpts[idx:], a*gridpts[idx:]**p, color='red', linewidth=4, label="$y="+f"{a}" + "x^{" + f"{p}" + "}$")      # zp = -1.3 + 1.3i
#plt.plot(gridpts[idx:], 1.1*gridpts[idx:]**1.35, color='red', linewidth=4, label="$y=1.1x^{1.35}$")    # zp = 0 + 1.3i
plt.plot(gridpts[idx::6], yvals[idx::6], '.', color='black',label="data")#,c=cols[idx:]/cols[-1], cmap=cmap)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("$t$")
plt.ylabel("$\mu(t)-\mu(0)$")
ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.title("Density near corner")
plt.legend(edgecolor='none')
plt.xlim([gridpts[idx], gridpts[250]])
plt.ylim([yvals[idx], yvals[250]])
#plt.xlim([gridpts[90], gridpts[150]])
#plt.ylim([yvals[90], yvals[150]])
plt.figure(1).savefig("boundary_integrals/figures/corner_density2.pdf", bbox_inches="tight")


# Attempt at gradient descent
xdat = gridpts[200:350]
ydat = mu[200:350]-mu0

plt.plot(xdat,ydat)
# f(t) = t^a  f(t)/f(2t) = 1/2^a -> log2(f(2t)/f(t)) ~ a

fun = lambda t,a,b,c: a + b*(t**c)
gradfun = lambda t,a,b,c: np.array([1, t**c, np.log(t)*b*t**c])
gradloss = lambda t, a, b, c,y: gradfun(t,a,b,c)*(fun(t,a,b,c)-y)

# Initial guess
a = xdat[0]
c = 0.008
b = (ydat[0]-a)/xdat[0]**c
pars = np.array([a,b,c])
eta = 0.01 # Step sie
for i in range(10000):
    idx = np.random.randint(0, len(xdat))
    pars -= 1*gradloss(xdat[idx],pars[0],pars[1],pars[2],ydat[idx])


# f(t) = a + b x^c -> grad(f) = [1, x^c, c*b*x^{c-1}]
# grad((f(t) - y)^2) =
plt.plot(xdat,ydat)
plt.plot(xdat,fun(xdat, pars[0], pars[1], pars[2]))
plt.yscale("log")
plt.xscale("log")

i = 200
a_apx = []
t_apx = []
for j in range(10):
    y1 = mu[i]-mu[150]
    y2 = mu[i+16]-mu[150]
    print(gridpts[i+16]/gridpts[i])
    a_apx.append(np.log2(y2/y1))
    t_apx.append(gridpts[i])
    i += 16

plt.scatter(t_apx, a_apx, label="$\log_2(\mu(2t)/\mu(t))")


plt.xscale("log")
ax = plt.gca()
