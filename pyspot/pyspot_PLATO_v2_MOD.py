import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

from astropy.table import Table
t1 = Table.read('meunier_19a_t1.dat',format = 'ascii')

from numpy.random import default_rng
###LM next line commented in order to allow for global random seed
#rng = default_rng()

DAY2SEC = 86400
YEAR2DAY = 365
DEG2RAD = np.pi / 180.0
BV_SUN = 0.656
LRHK_SUN = -5.025 # from Lorenzo-Oliveira et al. (2018, A&A 619, A73)
PROT_SUN = 27.0
OMEGA_SUN = 2 * np.pi / (PROT_SUN * DAY2SEC)

####################################
# FROM TEFF TO ACTIVITY PARAMETERS #
####################################

# all relations used are based on Meunier et al. (2019, A&A, 627, A56) except where specified otherwise

def get_stpar_from_teff(teff):
    if teff < min(t1['Teff']):
        print('Warning: Teff is outside range of conversion table. Returning B-V for Teff={}'.format(min(t1['Teff'])))
        return max(t1['BV'])
    if teff > max(t1['Teff']):
        print('Warning: Teff is outside range of conversion table. Returning B-V for Teff={}'.format(max(t1['Teff'])))
        return min(t1['BV'])
    g = interp1d(t1['Teff'],t1['BV'])
    return g(teff)

def get_lrhk_from_S_and_bv(S, bv):
    # cf Noyes et al. (1984, ApJ 279 763, Appendix a)
    lCcf = 1.13 * bv**3 - 3.91 * bv**2 + 2.84 * bv - 0.47 
    if bv < 0.63:
        x = 0.63 - bv
        lCcf += 0.135 * x - 0.814 * x**2 + 6.03 * x**3
    return -4 + np.log10(1.34) + lCcf + np.log10(S)

def get_lrhk_from_bv(bv):
    if bv < 0.94:
        Smin = 0.144
    else:
        Smin = 0.0269231 * bv + 0.118892
    lrhkmin = get_lrhk_from_S_and_bv(Smin, bv)
    lrhkmax = -0.375 * bv - 4.4
    return rng.random() * (lrhkmax - lrhkmin) + lrhkmin

def get_ltauc_from_bv(bv):
    # cf Noyes et al. (1984, ApJ 279 763, Eqn 4)
    x = 1.0 - bv
    if x > 0:
        return 1.362 - 0.166 * x + 0.025 * x**2 - 5.323 * x**3
    else:
        return 1.362 - 0.14 * x
    
def get_prot_from_lrhk_and_bv(lrhk, bv):
    Ro = 0.808 - 2.966 * (lrhk + 4.52)
    delta = rng.random() * 0.4 - 0.2
    ltc = get_ltauc_from_bv(bv)
    return (Ro + delta) * 10**ltc 
    
def get_prange_from_teff_and_prot(teff, prot):
    p0 = -3.485 + 2.47810e-4 * teff
    p1 = 1.597 - 1.3510e-4 * teff
    alpha = 10**(p0 + p1 * np.log10(prot))
    pmax = 2 * prot / (2 - alpha)
    pmin = pmax * (1 - alpha)
    return pmin, pmax

def get_latrange(): 
    lat_min = 0.0
    lat_max = 32.0 + 20.0 * rng.random()
    return lat_min, lat_max # in degrees

def get_omega01_from_prange_and_latrange(pmin, pmax, lat_min, lat_max):
    omega_min = 2 * np.pi / pmax / DAY2SEC
    omega_max = 2 * np.pi / pmin / DAY2SEC
    s2min = np.sin(lat_min * DEG2RAD)**2
    s2max = np.sin(lat_max * DEG2RAD)**2
    omega_1 = (omega_min - omega_max) / (s2max - s2min)
    omega_0 = omega_max - omega_1 * s2min 
    return omega_0, omega_1 
    
def get_omega_from_lat_and_omega01(lat, omega_0, omega_1):
    return omega_0 + omega_1 * np.sin(lat * DEG2RAD)**2 # in radians per second
                       
def get_pcyc_from_prot(prot):
    delta = rng.random() * 0.6 - 0.3
    y = 0.84 * np.log10(1/prot) + 3.14 + delta
    return prot * 10**y

def get_acyc_from_bv_and_lrhk(bv, lrhk):
    if bv < 0.851:
        Acyc_max = 0.727 * bv - 0.292
    else:
        Acyc_max = 0.727 * 0.851 - 0.292
    Acyc_min = max([0.28 * bv - 0.196, 0.342 * lrhk + 1.703, 0.005])
    return rng.random() * (Acyc_max - Acyc_min) + Acyc_min

def get_arate_from_acyc(acyc):
    asun = get_acyc_from_bv_and_lrhk(BV_SUN, LRHK_SUN)
    return acyc/asun

###########################
# ACTIVE REGION EMERGENCE #
###########################

def regions(activityrate = 1, cycle_period = 10, cycle_overlap = 0, randspots = False, \
            maxlat = 70, minlat = 0, \
            tsim = 1000, tstart = 0, verbose  = True):

# ;  According to Schrijver and Harvey (1994), the number of active regions
# ;  emerging with areas in the range [A,A+dA] in a time dt is given by 
# ;
# ;    n(A,t) dA dt = a(t) A^(-2) dA dt ,
# ;
# ;  where A is the "initial" area of a bipole in square degrees, and t is
# ;  the time in days; a(t) varies from 1.23 at cycle minimum to 10 at cycle
# ;  maximum.
# ;
# ;  The bipole area is the area within the 25-Gauss contour in the
# ;  "initial" state, i.e. time of maximum development of the active region.
# ;  The assumed peak flux density in the initial sate is 1100 G, and
# ;  width = 0.2*bsiz (see disp_region). The parameters written onto the
# ;  file are corrected for further diffusion and correspond to the time
# ;  when width = 4 deg, the smallest width that can be resolved with lmax=63.
# ;
# ;  In our simulation we use a lower value of a(t) to account for "correlated"
# ;  regions.

    nbin=5                              # number of area bins
    delt=0.5                            # delta ln(A)
    amax=100.                           # orig. area of largest bipoles (deg^2)
    dcon = np.exp(0.5*delt)-np.exp(-0.5*delt)   # contant from integ. over bin

    if verbose:
        print('Creating regions with the following parameters:')
        print('Acivity rate: {} x Solar rate.'.format(activityrate))
        print('Activity cycle period: {} years.'.format(cycle_period))
        print('Maximum spot latitude: {} degrees.'.format(maxlat))
        print('Minimum spot latitude: {} degrees.'.format(minlat))
        print('Duration of simulation: {} days.'.format(tsim))
        print('Time at start of simulation: {} days.'.format(tstart))
        
    latrmsd = 6
    atm = 10 * activityrate    
    # a(t) at cycle maximum (deg^2/day)
    # cycle period (days)
    # cycle duration (days)
     
    ncycle = int(cycle_period * 365)         # cycle length in days   
    nclen = int((cycle_period + cycle_overlap) * 365)
    fact = np.exp(delt*np.arange(nbin))     # array of area reduction factors
    ftot = fact.sum()                         # sum of reduction factors
    bsiz = np.sqrt(amax/fact)               # array of bipole separations (deg)
    tau1 = 5                                  # first and last times (in days) for
    tau2 = 15                                 #   emergence of "correlated" regions
    prob = 0.001                              # total probability for "correlation"
    nlon = 36                                 # number of longitude bins
    nlat = 16                                 # number of latitude bins       
    nday1 = 0                                 # first day to be simulated
    ndays = int(tsim)                              # number of days to be simulated
    dt = 1

    # Initialize time since last emergence of a large region, as function
    # of longitude, latitude and hemisphere:
    tau = np.zeros((nlon,nlat,2),'int') + tau2
    dlon = 360. / nlon
    dlat = maxlat / nlat
    
    # Create arrays to store regions properties
    reg_tims = []
    reg_lats = []
    reg_lons = []
    reg_angs = []

    # Loop over time (in days):
    ncnt = 0
    ncur = 0
    start_day = 0
        
    for nd in range(ndays):
        nday = nd + nday1
            
        # Compute index of most recently started cycle:
        ncur_now = int(nday / ncycle)
        ncur_prev = int((nday-1) / ncycle)
        if ncur_now > ncur_prev:
            ncur = ncur + 1

        #  Initialize rate of emergence for largest regions, and add 1 day
        #  to time of last emergence:
        tau = tau + 1
        rc0 = np.zeros((nlon,nlat,2))
        l = (tau > tau1) & (tau <= tau2)
        if l.any():
            rc0[l] = prob / (tau2 - tau1)
 
        #  Loop over current and previous cycle:
        for icycle in [0,1]:
            nc = ncur-icycle # index of cycle
            if ncur == 0:
                start_day = nc * ncycle
            else:  
                if ncur == 1:
                    if icycle == 0:
                        start_day = ncycle * nc
                    elif icycle == 1:
                        start_day = 0
                else:
                    start_day = ncycle * nc
           
            nstart = start_day        # start date of cycle
            if (nday-nstart) < nclen:  
                ic = 1 - 2 * ((nc + 2) % 2) # +1 for even, -1 for odd cycle
                phase = float(nday-nstart) / nclen # phase within the cycle
                    
                # Emergence rate of largest "uncorrelated" regions (number per day,
                # both hemispheres), from Schrijver and Harvey (1994):
                ru0_tot = atm * np.sin(np.pi * phase)**2 * (1.0 * dcon) / amax
            
                # Emergence rate of largest "uncorrelated" regions per latitude/longitude
                # bin (number per day), as function of latitude:
                if randspots:
                    latavg = (maxlat - minlat) / 2. 
                    latrms = maxlat - minlat
                    nlat1 = np.floor(minlat / dlat).astype(int)
                    nlat2 = np.floor(maxlat / dlat).astype(int)
                    nlat2 = min([nlat2, nlat - 1])
                else:
                    latavg = maxlat + (minlat - maxlat)*phase #+ 5.*phase**2
                    latrms = (maxlat/5.) - latrmsd * phase # rms latitude (degrees)
                    nlat1 = np.floor(max([maxlat * 0.9 - 1.2 * maxlat * phase, 0.0]) / dlat).astype(int) # first and last index
                    nlat2 = np.floor(min([maxlat + 15. - maxlat * phase, maxlat]) / dlat).astype(int)
                    nlat2 = min([nlat2, nlat - 1])
                
                js = np.arange(nlat2 - nlat1).astype(int)

                p = np.zeros(nlat)
                for j in np.arange(nlat2-nlat1+1).astype(int) + nlat1:
                    p[j] = np.exp( - ((dlat * (0.5 + j) - latavg) / latrms)**2)
                ru0 = ru0_tot * p / (p.sum() * nlon * 2)
            
                # Loops over hemisphere and latitude:
                for k in [0,1]:
                    for j in np.arange(nlat2-nlat1+1).astype(int) + nlat1:
                        # Emergence rates of largest regions per longitude/latitude bin (number
                        # per day):
                        r0 = ru0[j] + rc0[:,j,k]
                        rtot = r0.sum()
                        ssum = rtot * ftot
                        x = rng.random()
                        if x <= ssum:
                            nb = 0
                            sumb = rtot * fact[0]
                            while x > sumb:
                                nb = nb + 1
                                sumb = sumb + rtot * fact[nb]
                            i = 0
                            sumb = sumb + (r0[0] - rtot) * fact[nb]
                            while x > sumb:
                                i = i + 1
                                sumb = sumb + r0[i] * fact[nb]
                            lon = dlon * (rng.random() + float(i))
                            lat = dlat * (rng.random() + float(j))
                            if (nday > tstart):
                                reg_tims.append(rng.random() + nday)
                                reg_lons.append(lon)
                                if k == 0:                       # Insert on N hemisphere
                                    reg_lats.append(lat)
                                else:
                                    reg_lats.append(-lat)
                                x = rng.normal()
                                while abs(x) > 1.6:
                                    x = rng.normal()
                                y = rng.normal()
                                while abs(y) >= 1.6:
                                    y = rng.normal()
                                z = rng.random()
                                if z > 0.14:
                                    ang = 0.5 * lat + 2.0 + 27. * x * y # tilt angle (degrees)
                                else:
                                    z = rng.normal()
                                    while z > 0.5:
                                        z = rng.normal()
                                    ang =  z * np.pi / 180 # yes I know this is weird.
                                reg_angs.append(ang)
                                if verbose:
                                    print(reg_tims[-1], reg_lats[-1], reg_lons[-1], reg_angs[-1])
                            ncnt = ncnt + 1
                            if nb < 1:
                                tau[i,j,k] = 0
              
    if verbose:
        print('Total number of regions:  ',ncnt)

    reg_arr = np.zeros((4, len(reg_tims)))
    reg_arr[0] = np.array(reg_tims)
    reg_arr[1] = np.array(reg_lats)
    reg_arr[2] = np.array(reg_lons)
    reg_arr[3] = np.array(reg_angs) * DEG2RAD
    return reg_arr

#####################
# FROM SPOTS TO LCS #
#####################

class spots():
    """Holds parameters for spots on a given star"""
    def __init__(self, reg_arr,
                 incl = None, omega_0 = OMEGA_SUN, omega_1 = 0.0, \
                 dur = None, threshold = 0.1):
        '''Generate initial parameter set for spots (emergence times
        and initial locations are p[)'''
        
        # set global stellar parameters which are the same for all spots
        # inclination (in degrees)
        if incl == None:
            self.incl = np.arcos(np.random.uniform()) / DEG2RAD
        else:
            self.incl = incl
        # rotation and differential rotation (in radians / sec)
        self.omega_0 = omega_0
        self.omega_1 = omega_1
        # regions parameters
        t0 = reg_arr[0,:]
        lat = reg_arr[1,:]
        lon = reg_arr[2,:]
        ang = reg_arr[3,:]
        # keep only spots emerging within specified time-span, with peak B-field > threshold
        if dur == None:
            self.dur = t0.max() 
        else:
            self.dur = dur
        l = (t0 < self.dur) * (ang > threshold)
        self.nspot = l.sum()
        self.t0 = t0[l]
        self.lat = lat[l]
        self.lon = lon[l]
        # The settings below are designed approximately match the distributions used in 
        # Borgniet et al. (2015) and Meunier et al. (2019)        
        # spot sizes
        self.amax = ang[l]**2 * 300 * 1e-6 
        # spot emergence and decay timescales
        mea = 15 * 1e-6
        med = 10 * 1e-6
        mu = np.log(med)
        sig = np.sqrt(2*np.log(mea/med))
        self.decay_rate = rng.lognormal(mean=mu, sigma=sig, size=self.nspot)

    def calci(self, time, i):
        '''Evolve one spot and calculate its impact on the stellar flux'''
        '''NB: Currently there is no spot drift or shear'''
        # Spot area (linear growth and decay)
        area = np.zeros(len(time)) 
        decay_time = self.amax[i] / self.decay_rate[i]
        emerge_time = decay_time / 10.0
        # exponential growth and decay
        l = time < self.t0[i]
        area[l] = self.amax[i] * np.exp(-(self.t0[i]-time[l]) / emerge_time)
        l = time >= self.t0[i]
        area[l] = self.amax[i] * np.exp(-(time[l]-self.t0[i]) / decay_time)
#         # linear growth and decay
#         l = (time >= (self.t0[i]-emerge_time)) * (time < self.t0[i])
#         area[l] = self.amax[i] * (self.t0[i]-time[l]) / emerge_time
#         l = (time >= self.t0[i]) * (time < (self.t0[i]+decay_time))
#         area[l] = self.amax[i] * (1-(time[l]-self.t0[i]) / decay_time)
        # Rotation rate
        ome = get_omega_from_lat_and_omega01(self.lat[i], self.omega_0, self.omega_1) # in radians per second
        # Fore-shortening 
        phase = ome * time * DAY2SEC + self.lon[i] * DEG2RAD # in radians
        beta = np.cos(self.incl * DEG2RAD) * np.sin(self.lat[i] * DEG2RAD) + \
            np.sin(self.incl * DEG2RAD) * np.cos(self.lat[i] * DEG2RAD) * np.cos(phase)
        # Differential effect on stellar flux
        dF = - area * beta
        dF[beta < 0] = 0
        return area, ome, beta, dF

    def calc(self, time):
        '''Calculations for all spots'''
        N = len(time)
        M = self.nspot
        area = np.zeros((M, N))
        ome = np.zeros(M)
        beta = np.zeros((M, N))
        dF = np.zeros((M, N))
        for i in np.arange(M):
            area_i, omega_i, beta_i, dF_i = self.calci(time, i)
            area[i,:] = area_i
            ome[i] = omega_i
            beta[i,:] = beta_i
            dF[i,:] = dF_i
        return area, ome, beta, dF

def simulate_lc(teff = 5777, dur = 700, cadence_hours = 6.0, \
                incl = None, isim = 0, odir = None, \
                verbose = True, doplot = True, random_seed=None):
    ###LM addition to the call procedure line -> , random_seed=None):
        
    ###LM The next five lines have been added to allow for the reproducibility of results
    global rng
    if random_seed is None:
        rng = default_rng(seed=8348435735)
    else:
        rng = default_rng(seed=random_seed) 
    
    if odir is None:
        odir = os.getcwd()
    # select parameters
    bv = get_stpar_from_teff(teff)
    lrhk = get_lrhk_from_bv(bv)
    prot = get_prot_from_lrhk_and_bv(lrhk, bv)
    pmin, pmax = get_prange_from_teff_and_prot(teff, prot)
    lmin, lmax = get_latrange()
    omega_0, omega_1 = get_omega01_from_prange_and_latrange(pmin, pmax, lmin, lmax)
    pcyc =  get_pcyc_from_prot(prot)
    clen = pcyc / YEAR2DAY
    coverlap = rng.random() * 0.1 * clen 
    acyc = get_acyc_from_bv_and_lrhk(bv, lrhk)
    arate = get_arate_from_acyc(acyc)
    if incl is None:
        incl = np.arccos(rng.random()) / DEG2RAD
    # save star's overall properties at the top of the regions file
    rfile = os.path.join(odir, 'regions_{:04d}.txt'.format(isim)) # save modified regions params
    flo = open(rfile, 'w')
    flo.write('# T_eff = {} K\n'.format(teff))
    flo.write('# B-V = {} mag\n'.format(bv))
    flo.write("# log R'_HK = {} \n".format(lrhk))
    flo.write('# P_rot = {} days\n'.format(prot))
    flo.write('# P_min = {} days\n'.format(pmin))
    flo.write('# P_max = {} days\n'.format(pmax))
    flo.write('# max. latitude = {} deg\n'.format(lmax))
    flo.write('# P_cycle = {} years\n'.format(clen))
    flo.write('# Cycle overlap = {} years\n'.format(coverlap))
    flo.write('# Activity rate = {} solar\n'.format(arate))
    flo.write('# sin(incl) = {}\n'.format(np.sin(incl * DEG2RAD)))
    flo.write('# \n')
    
    # print them to screen          
    if verbose:
        print('GLOBAL PROPERTIES')
        print('T_eff = {} K'.format(teff))
        print('B-V = {} mag'.format(bv))
        print("log R'_HK = {}".format(lrhk))
        print('P_rot = {} days'.format(prot))
        print('P_min = {} days'.format(pmin))
        print('P_max = {} days'.format(pmax))
        print('max. latitude = {} deg'.format(lmax))
        print('P_cycle = {} years'.format(clen))
        print('Cycle overlap = {} years'.format(coverlap))
        print('Activity rate = {} solar'.format(arate))
        print('sin(incl) = {}'.format(np.sin(incl * DEG2RAD)))
        print('')
        
    # simulate regions
    reg_arr = regions(activityrate = arate, cycle_period = clen, cycle_overlap = coverlap, \
                      maxlat = lmax, minlat = lmin, \
                      tsim = dur + pcyc, tstart = 0, verbose  = False)
    reg_arr[0] -= rng.random() * pcyc # make the simulation start at a random point in the cycle
    # simulate LC
    s = spots(reg_arr, incl = incl, omega_0 = omega_0, omega_1 = omega_1, \
              threshold = 0.1, dur = dur)
    time = np.r_[0:dur:cadence_hours/24.]
    area, ome, beta, dF = s.calc(time)

    # save individual spot properties
    header = '{:6s} {:6s} {:6s} {:6s} {:8s} {:6s} {:6s}'.format('LAT','LON', 'PROT', 'T_MAX', 'A_MAX', 'TAU', 'TAU_R')
    flo.write('# {}\n'.format(header))
#    if verbose:
#        print(header)
    header = '{:6s} {:6s} {:6s} {:6s} {:8s} {:6s} {:6s}'.format('deg','deg', 'days', 'days', 'muHem', 'days', 'periods')
    flo.write('# {}\n'.format(header))
#    if verbose:
#        print(header)
    for i in range(s.nspot):
        if area[i,:].max() == 0: 
            # spot came too early or too late or was too short lived given cadence
            continue
        prot = 2*np.pi / ome[i] / DAY2SEC
        lifetime = s.amax[i] / s.decay_rate[i]
        str_ = '{:6.1f} {:6.2f} {:6.2f} {:6.2f} {:8.2e} {:6.2f} {:6.2f}'.format(s.lat[i], s.lon[i], prot, s.t0[i], s.amax[i] * 1e6, lifetime, lifetime/prot)
        flo.write('{}\n'.format(str_))
#        if verbose:
#            print(str_)
                
    # save LC
    X = np.zeros((2,len(time)))
    X[0,:] = time
    X[1,:] = dF.sum(0)
    lfile = os.path.join(odir, 'lightcurve_{:04d}.txt'.format(isim)) # save LC
    np.savetxt(lfile,X.T)

    if doplot:
        fig, axes = plt.subplots(3,1, figsize=(6,4), sharex=True)
        ttl= '{:d} AR={:5.3f} CL={:6.3f} sin(i)={:6.2f} Pmin={:6.2f} Pmax={:6.2f}, Lmax={:5.2f}'.format(isim, arate, clen, \
                                                                                                   np.sin(incl * DEG2RAD), pmin, pmax, \
                                                                                                   lmax)
        axes[0].set_title(ttl)
        for j in range(s.nspot):
            if s.t0[j] < -10:
                continue
            if s.t0[j] > dur:
                continue
            axes[0].plot(s.t0[j], s.lat[j], 'ko', markersize = s.amax[j]*(1./3e-4)*5, alpha = 0.5)
        axes[0].set_ylim(-90,90)
        axes[0].set_ylabel('spot lat. (deg)')
        axes[1].plot(time, area.sum(0), 'k-')
        axes[1].set_ylabel('spot coverage')        
        axes[2].plot(time, dF.sum(0), 'k-')
        axes[2].set_ylabel('delta flux')
        axes[2].set_xlim(0, dur)
        axes[2].set_xlabel('time (days)')
        plt.savefig(os.path.join(odir, 'lightcurve_{:04d}.png'.format(isim)))
        if verbose:
            plt.show()
        else:
            plt.close('all')

    return

if __name__ == '__main__':
    simulate_lc(teff = 5777, dur = 700, cadence_hours = 6.0, \
                incl = 90.0, isim = 0, verbose = True, doplot = True)
    

    
