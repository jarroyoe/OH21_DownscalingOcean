import sys
sys.path.append("/home/users/bdelorme/croco-v1.1/Python3_Modules")
from Modules import *
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 22
plt.rcParams['lines.linewidth'] = 2

def get_var(varname, simul, time):
    simul.update(time)
    return var(varname,simul).data[:,:,-1]

def get_d(simul,time):
    simul.update(time)
    return simul.date.split('-')[0]

def chunks(l,n):
    n = max(1,n)
    return (l[i:i+n] for i in range(0, len(l), n))

levsR = np.linspace(-1, 1, 81)
Rticks = [-1, 0, 1]

# Simu
sim = load('arcterx2')
times = np.arange(0, 360, 1)
lon = np.mean(sim.x, axis=1)
lat = np.mean(sim.y, axis=0)
lonm = (lon[1:] + lon[:-1]) / 2.
latm = (lat[1:] + lat[:-1]) / 2.
Nt = len(times)
nproc = 8
get_vort = fts.partial(get_var, 'vrt', sim)
get_date = fts.partial(get_d, sim)
vort = []
dates = []
for subtimes in chunks(times,nproc):
    print('Computing ', subtimes)
    pool = mp.Pool(nproc)
    results = pool.map(get_vort, subtimes)
    pool.close(); pool.join()
    vort = vort + results
    pool = mp.Pool(nproc)
    results = pool.map(get_date, subtimes)
    pool.close(); pool.join()
    dates = dates + results
vort = np.array(vort)
f = sim.f
f = (f[1:,:] + f[:-1,:]) / 2.
f = (f[:,1:] + f[:,:-1]) / 2.
Ro = vort / f
for it in range(Nt):
    print(it)
    plt.figure(figsize=(10,10))
    plt.contourf(lonm, latm, Ro[it,:,:].T, levsR, cmap='seismic', extend='both')
    cbar = plt.colorbar(ticks=Rticks)
    cbar.set_label(r'Ro [$\zeta/f$]')
    plt.xlabel(r'$x$ [$^{\circ}$E]')
    plt.ylabel(r'$y$ [$^{\circ}$N]')
    plt.title(dates[it])
    plt.tight_layout()
    plt.savefig('plots/arcvort_'+str(it)+'.png')
    plt.close()

# Simu
sim = load('archigh2')
times = np.arange(0, 720, 1)
lon = np.mean(sim.x, axis=1)
lat = np.mean(sim.y, axis=0)
lonm = (lon[1:] + lon[:-1]) / 2.
latm = (lat[1:] + lat[:-1]) / 2.
Nt = len(times)
nproc = 8
get_vort = fts.partial(get_var, 'vrt', sim)
get_date = fts.partial(get_d, sim)
vort = []
dates = []
for subtimes in chunks(times,nproc):
    print('Computing ', subtimes)
    pool = mp.Pool(nproc)
    results = pool.map(get_vort, subtimes)
    pool.close(); pool.join()
    vort = vort + results
    pool = mp.Pool(nproc)
    results = pool.map(get_date, subtimes)
    pool.close(); pool.join()
    dates = dates + results
vort = np.array(vort)
f = sim.f
f = (f[1:,:] + f[:-1,:]) / 2.
f = (f[:,1:] + f[:,:-1]) / 2.
Ro = vort / f
for it in range(Nt):
    print(it)
    plt.figure(figsize=(10,10))
    plt.contourf(lonm, latm, Ro[it,:,:].T, levsR, cmap='seismic', extend='both')
    cbar = plt.colorbar(ticks=Rticks)
    cbar.set_label(r'Ro [$\zeta/f$]')
    plt.xlabel(r'$x$ [$^{\circ}$E]')
    plt.ylabel(r'$y$ [$^{\circ}$N]')
    plt.title(dates[it])
    plt.tight_layout()
    plt.savefig('plots/highvort_'+str(it)+'.png')
    plt.close()

# Simu
sim = load('arczoom2')
times = np.arange(0, 720, 1)
lon = np.mean(sim.x, axis=1)
lat = np.mean(sim.y, axis=0)
lonm = (lon[1:] + lon[:-1]) / 2.
latm = (lat[1:] + lat[:-1]) / 2.
Nt = len(times)
nproc = 8
get_vort = fts.partial(get_var, 'vrt', sim)
get_date = fts.partial(get_d, sim)
vort = []
dates = []
for subtimes in chunks(times,nproc):
    print('Computing ', subtimes)
    pool = mp.Pool(nproc)
    results = pool.map(get_vort, subtimes)
    pool.close(); pool.join()
    vort = vort + results
    pool = mp.Pool(nproc)
    results = pool.map(get_date, subtimes)
    pool.close(); pool.join()
    dates = dates + results
vort = np.array(vort)
f = sim.f
f = (f[1:,:] + f[:-1,:]) / 2.
f = (f[:,1:] + f[:,:-1]) / 2.
Ro = vort / f
for it in range(Nt):
    print(it)
    plt.figure(figsize=(10,10))
    plt.contourf(lonm, latm, Ro[it,:,:].T, levsR, cmap='seismic', extend='both')
    cbar = plt.colorbar(ticks=Rticks)
    cbar.set_label(r'Ro [$\zeta/f$]')
    plt.xlabel(r'$x$ [$^{\circ}$E]')
    plt.ylabel(r'$y$ [$^{\circ}$N]')
    plt.title(dates[it])
    plt.tight_layout()
    plt.savefig('plots/zoomvort_'+str(it)+'.png')
    plt.close()
