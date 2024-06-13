import os
from typing import Optional
from importlib_resources import files

import numpy as np
from numpy.typing import ArrayLike
from numpy.random import default_rng
from scipy.interpolate import interp1d

from astropy.table import Table

from . import spotmodels

import matplotlib.pyplot as plt

# Some global constants.
DAY2SEC = 86400
YEAR2DAY = 365
DEG2RAD = np.pi / 180.0
BV_SUN = 0.656
LRHK_SUN = -5.025  # from Lorenzo-Oliveira et al. (2018, A&A 619, A73)
PROT_SUN = 27.0
OMEGA_SUN = 2 * np.pi / (PROT_SUN * DAY2SEC)
ASUN = 0.12

# Table of stellar properties from Meunier et al. (2019, A&A 627, A56)
data_text = files('pyspot').joinpath('meunier_19a_t1.dat')
t1 = Table.read(data_text, format='ascii')

# Default random number generator.
RNG = default_rng(seed=8348435735)


def set_random_seed(random_seed):

    global RNG
    if random_seed is not None:
        RNG = default_rng(seed=random_seed)

    return

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
    g = interp1d(t1['Teff'], t1['BV'])
    return g(teff).item()


def get_lrhk_from_S_and_bv(S, bv):
    # conversion from S to R (Noyes et al. 1984a, ApJ 279 763, Appendix a)
    lCcf = 1.13 * bv ** 3 - 3.91 * bv ** 2 + 2.84 * bv - 0.47
    if bv < 0.63:
        x = 0.63 - bv
        lCcf += 0.135 * x - 0.814 * x ** 2 + 6.03 * x ** 3
    lrhk = -4 + np.log10(1.34) + lCcf + np.log10(S)
    # photospheric correction (Noyes et al. 1984a, ApJ 279 763, Appendix b)    
    lrphot = -4.898 + 1.918 * bv ** 2 - 2.893 * bv ** 3
    lrhk = np.log10(10 ** lrhk - 10 ** lrphot)
    return lrhk


def get_Smin_from_bv(bv):
    if bv < 0.94:
        Smin = 0.144
    elif bv < 1.07:
        x = (bv - 0.94) / (1.07 - 0.94)
        Smin = 0.144 + x * (0.19 - 0.144)
    else:
        x = (bv - 1.07) / (1.2 - 1.07)
        Smin = 0.19 + x * (0.48 - 0.19)
    return Smin


def get_lrhk_from_bv(bv):
    Smin = get_Smin_from_bv(bv)
    lrhkmin = get_lrhk_from_S_and_bv(Smin, bv)
    lrhkmax = -0.375 * bv - 4.4
    return RNG.random() * (lrhkmax - lrhkmin) + lrhkmin


def get_ltauc_from_bv(bv):
    # cf Noyes et al. (1984, ApJ 279 763, Eqn 4)
    x = 1.0 - bv
    if x > 0:
        return 1.362 - 0.166 * x + 0.025 * x**2 - 5.323 * x**3
    else:
        return 1.362 - 0.14 * x


def get_prot_from_lrhk_and_bv(lrhk, bv):
    Ro = 0.808 - 2.966 * (lrhk + 4.52)
    delta = RNG.random() * 0.4 - 0.2
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
    lat_max = 32.0 + 20.0 * RNG.random()
    return lat_min, lat_max  # in degrees


def get_omega01_from_prange_and_latrange(pmin, pmax, lat_min, lat_max):
    omega_min = 2 * np.pi / pmax / DAY2SEC
    omega_max = 2 * np.pi / pmin / DAY2SEC
    s2min = np.sin(lat_min * DEG2RAD)**2
    s2max = np.sin(lat_max * DEG2RAD)**2
    omega_1 = (omega_min - omega_max) / (s2max - s2min)
    omega_0 = omega_max - omega_1 * s2min 
    return omega_0, omega_1 


def get_omega_from_lat_and_omega01(lat, omega_0, omega_1):
    return omega_0 + omega_1 * np.sin(lat * DEG2RAD)**2  # in radians per second


def get_pcyc_from_prot(prot):
    delta = RNG.random() * 0.6 - 0.3
    y = 0.84 * np.log10(1/prot) + 3.14 + delta
    return prot * 10**y


def get_acyc_from_bv_and_lrhk(bv, lrhk, level='random'):
    if bv < 0.851:
        Acyc_max = 0.727 * bv - 0.292
    else:
        Acyc_max = 0.727 * 0.851 - 0.292
    Acyc_min = max([0.28 * bv - 0.196, 0.342 * lrhk + 1.703, 0.005])

    tmp = RNG.random()
    if level == 'random':
        pass
    elif level == 'high':
        tmp = 0.9 + 0.1*tmp
    elif level == 'low':
        tmp = 0.0 + 0.1*tmp
    else:
        raise ValueError(f"Invalid activity level '{level}'")

    return tmp * (Acyc_max - Acyc_min) + Acyc_min


def get_arate_from_acyc(acyc):
    # asun = get_acyc_from_bv_and_lrhk(BV_SUN, LRHK_SUN)
    return acyc/ASUN


def get_decay_rate(nspots):
    # The settings below are designed approximately match the distributions used in
    # Borgniet et al. (2015) and Meunier et al. (2019)

    mea = 15 * 1e-6
    med = 10 * 1e-6
    mu = np.log(med)
    sig = np.sqrt(2 * np.log(mea / med))
    decay_rate = RNG.lognormal(mean=mu, sigma=sig, size=nspots)

    return decay_rate

###########################
# ACTIVE REGION EMERGENCE #
###########################


def regions(activityrate: float = 1,
            cycle_period: float = 10,
            cycle_overlap: float = 0,
            randspots: bool = False,
            maxlat: float = 70.,
            minlat: float = 0.,
            tsim: float = 1000,
            tstart: float = 0,
            verbose: bool = True,
            random_seed: Optional[int] = None
            ) -> np.ndarray:
    """

    According to Schrijver and Harvey (1994), the number of active regions
    emerging with areas in the range [A,A+dA] in a time dt is given by

    n(A,t) dA dt = a(t) A^(-2) dA dt ,

    where A is the "initial" area of a bipole in square degrees, and t is
    the time in days; a(t) varies from 1.23 at cycle minimum to 10 at cycle
    maximum.

    The bipole area is the area within the 25-Gauss contour in the
    "initial" state, i.e. time of maximum development of the active region.
    The assumed peak flux density in the initial sate is 1100 G, and
    width = 0.2*bsiz (see disp_region). The parameters written onto the
    file are corrected for further diffusion and correspond to the time
    when width = 4 deg, the smallest width that can be resolved with lmax=63.

    In our simulation we use a lower value of a(t) to account for "correlated"
    regions.

    """

    set_random_seed(random_seed)

    nbin = 5  # number of area bins
    delt = 0.5  # delta ln(A)
    amax = 100.  # orig. area of largest bipoles (deg^2)
    dcon = np.exp(0.5*delt) - np.exp(-0.5*delt)  # contant from integ. over bin

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
    tau2 = 15                                 # emergence of "correlated" regions
    prob = 0.001                              # total probability for "correlation"
    nlon = 36                                 # number of longitude bins
    nlat = 16                                 # number of latitude bins       
    nday1 = 0                                 # first day to be simulated
    ndays = int(tsim)                              # number of days to be simulated
    dt = 1

    # Initialize time since last emergence of a large region, as function
    # of longitude, latitude and hemisphere:
    tau = np.zeros((nlon, nlat, 2), 'int') + tau2
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
        rc0 = np.zeros((nlon, nlat, 2))
        mask = (tau > tau1) & (tau <= tau2)
        if mask.any():
            rc0[mask] = prob / (tau2 - tau1)
 
        #  Loop over current and previous cycle:
        for icycle in [0, 1]:
            nc = ncur-icycle  # index of cycle
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
                ic = 1 - 2 * ((nc + 2) % 2)  # +1 for even, -1 for odd cycle
                phase = float(nday-nstart) / nclen  # phase within the cycle
                    
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
                    latavg = maxlat + (minlat - maxlat)*phase  # + 5.*phase**2
                    latrms = (maxlat/5.) - latrmsd * phase  # rms latitude (degrees)
                    nlat1 = np.floor(max([maxlat * 0.9 - 1.2 * maxlat * phase, 0.0]) / dlat).astype(int)  # first and last index
                    nlat2 = np.floor(min([maxlat + 15. - maxlat * phase, maxlat]) / dlat).astype(int)
                    nlat2 = min([nlat2, nlat - 1])
                
                js = np.arange(nlat2 - nlat1).astype(int)

                p = np.zeros(nlat)
                for j in np.arange(nlat2-nlat1+1).astype(int) + nlat1:
                    p[j] = np.exp(- ((dlat * (0.5 + j) - latavg) / latrms)**2)
                ru0 = ru0_tot * p / (p.sum() * nlon * 2)
            
                # Loops over hemisphere and latitude:
                for k in [0, 1]:
                    for j in np.arange(nlat2-nlat1+1).astype(int) + nlat1:
                        # Emergence rates of largest regions per longitude/latitude bin (number
                        # per day):
                        r0 = ru0[j] + rc0[:, j, k]
                        rtot = r0.sum()
                        ssum = rtot * ftot
                        x = RNG.random()
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
                            lon = dlon * (RNG.random() + float(i))
                            lat = dlat * (RNG.random() + float(j))
                            if nday > tstart:
                                reg_tims.append(RNG.random() + nday)
                                reg_lons.append(lon)
                                if k == 0:                       # Insert on N hemisphere
                                    reg_lats.append(lat)
                                else:
                                    reg_lats.append(-lat)
                                x = RNG.normal()
                                while abs(x) > 1.6:
                                    x = RNG.normal()
                                y = RNG.normal()
                                while abs(y) >= 1.6:
                                    y = RNG.normal()
                                z = RNG.random()
                                if z > 0.14:
                                    ang = 0.5 * lat + 2.0 + 27. * x * y  # tilt angle (degrees)
                                else:
                                    z = RNG.normal()
                                    while z > 0.5:
                                        z = RNG.normal()
                                    ang = z * np.pi / 180  # yes I know this is weird.
                                reg_angs.append(ang)
                                if verbose:
                                    print(reg_tims[-1], reg_lats[-1], reg_lons[-1], reg_angs[-1])
                            ncnt = ncnt + 1
                            if nb < 1:
                                tau[i, j, k] = 0
              
    if verbose:
        print('Total number of regions:  ', ncnt)

    reg_arr = np.zeros((4, len(reg_tims)))
    reg_arr[0] = np.array(reg_tims)
    reg_arr[1] = np.array(reg_lats)
    reg_arr[2] = np.array(reg_lons)
    reg_arr[3] = np.array(reg_angs) * DEG2RAD

    return reg_arr

#####################
# FROM SPOTS TO LCS #
#####################


# class Spots:
#     """ Holds parameters for spots on a given star.
#     """
#
#     def __init__(self,
#                  reg_arr: np.ndarray,
#                  incl: Optional[float] = None,
#                  omega_0: float = OMEGA_SUN,
#                  omega_1: float = 0.0,
#                  dur: Optional[float] = None,
#                  threshold: float = 0.1,
#                  random_seed: Optional[int] = None
#                  ) -> None:
#         """ Generate initial parameter set for spots (emergence times and
#             initial locations are p[)
#         """
#
#         set_random_seed(random_seed)
#
#         # set global stellar parameters which are the same for all spots
#         # inclination (in degrees)
#         if incl is None:
#             self.incl = np.arccos(np.random.uniform()) / DEG2RAD
#         else:
#             self.incl = incl
#
#         # rotation and differential rotation (in radians / sec)
#         self.omega_0 = omega_0
#         self.omega_1 = omega_1
#
#         # regions parameters
#         t0 = reg_arr[0, :]
#         lat = reg_arr[1, :]
#         lon = reg_arr[2, :]
#         ang = reg_arr[3, :]
#
#         # keep only spots emerging within specified time-span, with peak B-field > threshold
#         if dur is None:
#             self.dur = t0.max()
#         else:
#             self.dur = dur
#
#         # mask = (t0 < self.dur) * (ang > threshold)
#         mask = ang > threshold
#         self.nspot = mask.sum()
#         self.t0 = t0[mask]
#         self.lat = lat[mask]
#         self.lon = lon[mask]
#
#         # The settings below are designed approximately match the distributions used in
#         # Borgniet et al. (2015) and Meunier et al. (2019)
#         # spot sizes
#         self.amax = ang[mask]**2 * 300 * 1e-6
#
#         # spot emergence and decay timescales
#         mea = 15 * 1e-6
#         med = 10 * 1e-6
#         mu = np.log(med)
#         sig = np.sqrt(2*np.log(mea/med))
#         self.decay_rate = RNG.lognormal(mean=mu, sigma=sig, size=self.nspot)
#
#     def _calci(self,
#                time: np.ndarray,
#                i: int
#                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         """ Evolve one spot and calculate its impact on the stellar flux.
#
#         NB: Currently there is no spot drift or shear
#
#         """
#
#         # Spot area (linear growth and decay)
#         area = np.zeros(len(time))
#         decay_time = self.amax[i] / self.decay_rate[i]
#         emerge_time = decay_time / 10.0
#
#         # exponential growth and decay
#         mask = time < self.t0[i]
#         area[mask] = self.amax[i] * np.exp(-(self.t0[i]-time[mask]) / emerge_time)
#         mask = time >= self.t0[i]
#         area[mask] = self.amax[i] * np.exp(-(time[mask]-self.t0[i]) / decay_time)
#
# #         # linear growth and decay
# #         l = (time >= (self.t0[i]-emerge_time)) * (time < self.t0[i])
# #         area[l] = self.amax[i] * (self.t0[i]-time[l]) / emerge_time
# #         l = (time >= self.t0[i]) * (time < (self.t0[i]+decay_time))
# #         area[l] = self.amax[i] * (1-(time[l]-self.t0[i]) / decay_time)
#
#         # Rotation rate
#         ome = get_omega_from_lat_and_omega01(self.lat[i], self.omega_0, self.omega_1)  # in radians per second
#
#         # Fore-shortening
#         phase = ome * time * DAY2SEC + self.lon[i] * DEG2RAD  # in radians
#         beta = (np.cos(self.incl * DEG2RAD) * np.sin(self.lat[i] * DEG2RAD) +
#                 np.sin(self.incl * DEG2RAD) * np.cos(self.lat[i] * DEG2RAD) * np.cos(phase))
#
#         # Differential effect on stellar flux
#         delta_flux = - 2 * area * beta
#         delta_flux[beta < 0] = 0
#
#         return area, ome, beta, delta_flux
#
#     def calc(self, time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         """ Calculations for all spots.
#         """
#
#         N = len(time)
#         M = self.nspot
#         area = np.zeros((M, N))
#         ome = np.zeros(M)
#         beta = np.zeros((M, N))
#         delta_flux = np.zeros((M, N))
#         for i in np.arange(M):
#             area_i, omega_i, beta_i, dflux_i = self._calci(time, i)
#             area[i, :] = area_i
#             ome[i] = omega_i
#             beta[i, :] = beta_i
#             delta_flux[i, :] = dflux_i
#
#         return area, ome, beta, delta_flux


def simulate_lc(effective_temperature: float = 5777.,
                duration_days: float = 700.,
                cadence_hours: float = 6.,
                stellar_inclination: Optional[float] = None,
                activity_level: str = 'random',
                activity_phase: tuple[float, float] = (0., 1.),
                min_area: float = 1e-8,
                evolution: str = 'squared-exponential',
                ld_type: str = 'linear',
                ld_pars: ArrayLike = (0.6,),
                simulation_label: Optional[str] = None,
                output_dir: Optional[str] = None,
                verbose: bool = True,
                diagnostic_plots: bool = True,
                random_seed: Optional[int] = None
                ) -> None:
    """ Simulate a lightcurve.

    Parameters
    ----------
    effective_temperature: float
        Effective temperature of the star to be simulated in kelvin
        (default: 5777 K).
    duration_days: float
        The duration of the simulated lightcurve in days (default 700 days).
    cadence_hours: float
        The cadence at which to simulate the lightcurve in hours
        (default 6 hours).
    stellar_inclination: float or None
        The stellar inclination in degrees if not given a random value is chosen (
        default: None).
    activity_level: str
        The activity level at which to simulate the star. Takes values:
        'random', 'low' lower 10th percentile, 'high' upper 10th percentile
        (default: 'random').
    activity_phase: tuple
        Set the phase range in the activity cycle that corresponds to the middle
        of the duration (default: [0, 1]).
    min_area: float
        Do not consider spots when their area is smaller than this value in
        hemispheres (default: 1e-8 hemispheres).
    evolution: str
        The way the spot area evolves with time.
    ld_type : str
        The limd-darkening law used (default: 'linear').
    ld_pars : tuple
        The limb-darkening parameters to use (default: (0.6,)).
    simulation_label: str
        String used in the output file names (default: 'test').
    output_dir: str
        The name of the output directory. If it does not exist it will be
        created (default: current directory).
    verbose: bool
        Print diagnostic messages to terminal (default: True).
    diagnostic_plots: bool
        If True create diagnostic plots (default: True).
    random_seed: int
        The random seed to use with the random number generator, if None a
        default random seed is used (default: None).

    """
    
    set_random_seed(random_seed)

    if simulation_label is None:
        simulation_label = 'test'

    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.makedirs(output_dir, exist_ok=True)

    # select parameters
    bv = get_stpar_from_teff(effective_temperature)
    lrhk = get_lrhk_from_bv(bv)
    prot = get_prot_from_lrhk_and_bv(lrhk, bv)
    pmin, pmax = get_prange_from_teff_and_prot(effective_temperature, prot)
    lmin, lmax = get_latrange()
    omega_0, omega_1 = get_omega01_from_prange_and_latrange(pmin, pmax, lmin, lmax)
    pcyc = get_pcyc_from_prot(prot)
    clen = pcyc / YEAR2DAY
    coverlap = RNG.random() * 0.1 * clen
    acyc = get_acyc_from_bv_and_lrhk(bv, lrhk, level=activity_level)
    arate = get_arate_from_acyc(acyc)
    if stellar_inclination is None:
        stellar_inclination = np.arccos(RNG.random()) / DEG2RAD
    
    # save star's overall properties at the top of the regions file
    meta_data = dict()
    meta_data["T_eff"] = effective_temperature
    meta_data["B-V"] = bv
    meta_data["log R'_HK"] = lrhk
    meta_data["P_rot"] = prot
    meta_data["P_min"] = pmin
    meta_data["P_max"] = pmax
    meta_data["max. latitude"] = lmax
    meta_data["P_cycle"] = clen
    meta_data["Cycle overlap"] = coverlap
    meta_data["Activity rate"] = arate

    # print them to screen          
    if verbose:
        print('GLOBAL PROPERTIES')
        print('T_eff = {} K'.format(effective_temperature))
        print('B-V = {} mag'.format(bv))
        print("log R'_HK = {}".format(lrhk))
        print('P_rot = {} days'.format(prot))
        print('P_min = {} days'.format(pmin))
        print('P_max = {} days'.format(pmax))
        print('max. latitude = {} deg'.format(lmax))
        print('P_cycle = {} years'.format(clen))
        print('Cycle overlap = {} years'.format(coverlap))
        print('Activity rate = {} solar'.format(arate))
        print('sin(incl) = {}'.format(np.sin(stellar_inclination * DEG2RAD)))
        print('')

    # Simulate a generous time-span to ensure we have all the spots we need.
    n = np.ceil(duration_days/pcyc/2) + 1
    span = (2*n + 1)*pcyc
        
    # simulate regions
    reg_arr = regions(activityrate=arate, cycle_period=clen, cycle_overlap=coverlap,
                      maxlat=lmax, minlat=lmin,
                      tsim=span, tstart=0, verbose=verbose)

    # Pick a time t0 such that the middle of the duration falls within a certain phase range of the activity cycle.
    phase0 = activity_phase[0] + RNG.random() * (activity_phase[1] - activity_phase[0])
    t0 = (n + phase0)*pcyc - duration_days/2
    reg_arr[0] -= t0

    # simulate LC
    # s = Spots(reg_arr, incl=stellar_inclination, omega_0=omega_0, omega_1=omega_1,
    #           threshold=0.1, dur=duration_days)
    # time = np.r_[0:duration_days:cadence_hours/24.]
    # area, ome, beta, delta_flux = s.calc(time)

    # Unpack the regions and compute omega and decay_rate for each spot.
    tmax = reg_arr[0, :]
    lat = reg_arr[1, :]
    lon = reg_arr[2, :]
    amax = reg_arr[3, :]**2 * 300 * 1e-6  # Area in hemispheres.
    omega = get_omega_from_lat_and_omega01(lat, omega_0, omega_1)
    decay_rate = get_decay_rate(len(tmax))

    prot = 2 * np.pi / omega / DAY2SEC
    lifetime = amax / decay_rate

    # Create the spots table.
    spots_table = Table([lat, lon, prot, tmax, amax*1e6, lifetime, lifetime/prot],
                        names=('LAT', 'LON', 'PROT', 'T_MAX', 'A_MAX', 'TAU', 'TAU_R'),
                        meta=meta_data)

    # Remove spots that do not contribute to the simulated lightcurve.
    time = np.r_[0:duration_days:cadence_hours / 24.]
    spots_table = spotmodels.filter_spots_table(time,
                                                spots_table,
                                                min_area=min_area,
                                                evolution=evolution)

    # Write the spot properties to file.
    filename = os.path.join(output_dir, 'regions_{}.ecsv'.format(simulation_label))
    spots_table.write(filename, overwrite=True)

    # Simulate the lightcurve.
    flux = spotmodels.kipping_spot_model(time,
                                         spots_table,
                                         inc_star=stellar_inclination,
                                         ld_type=ld_type,
                                         ld_pars=ld_pars,
                                         min_area=min_area,
                                         evolution=evolution)

    # Write the lightcurve to file.
    tab = Table([time, flux],
                names=('TIME', 'FLUX'))
    filename = os.path.join(output_dir, 'lightcurve_{}.ecsv'.format(simulation_label))
    tab.write(filename, overwrite=True)

    if diagnostic_plots:
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        ttl = '{} AR={:.3f} CL={:.3f} sin(i)={:.2f} Pmin={:.2f} Pmax={:.2f}, Lmax={:.2f}'
        ttl = ttl.format(simulation_label, arate, clen, np.sin(stellar_inclination * DEG2RAD), pmin, pmax, lmax)
        axes[0].set_title(ttl)

        area = np.zeros_like(time)
        for row in spots_table:
            area += spotmodels.compute_spot_area(time, row, min_area=min_area, evolution=evolution)
            axes[0].plot(row['T_MAX'], row['LAT'], 'ko', markersize=row['A_MAX']*1e-6*(1./3e-4)*5, alpha=0.5)

        axes[0].set_ylim(-90, 90)
        axes[0].set_ylabel('spot lat. (deg)')
        axes[1].plot(time, area, 'k-')
        axes[1].set_ylabel('spot coverage')        
        axes[2].plot(time, flux - 1, 'k-')
        axes[2].set_ylabel('delta flux')
        axes[2].set_xlim(0, duration_days)
        axes[2].set_xlabel('time (days)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lightcurve_{}.png'.format(simulation_label)), dpi=180)
        if verbose:
            plt.show()
        else:
            plt.close('all')

    return


def main():

    # variations = [('F5', 6452., 90.),
    #               ('G5', 5612., 90.),
    #               ('K4', 4607., 90.)]

    variations = [('F5', 6550., 90., 'high'),
                  ('G5', 5660., 90., 'high'),
                  ('K5', 4440., 90., 'high'),
                  ('F5', 6550., 730., 'high'),
                  ('G5', 5660., 730., 'high'),
                  ('K5', 4440., 730., 'high')]

    for star_type, teff, dur, level in variations:

        for i in range(100):
            simulate_lc(effective_temperature=teff,
                        duration_days=dur,
                        cadence_hours=6.0,
                        stellar_inclination=90.0,
                        activity_level=level,
                        simulation_label=f'{star_type}_d{dur:.1f}_teff{teff:.1f}_{level}_{i}',
                        output_dir='spots_tables_20240416',
                        verbose=False,
                        diagnostic_plots=True)

    return


if __name__ == '__main__':
    main()
