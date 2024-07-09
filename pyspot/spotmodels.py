import warnings

import numpy as np
from numpy.typing import ArrayLike
from astropy.table import Table

DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi


################################
# Functions for limb-darkening #
################################

def parse_limb_darkening(ld_type: str,
                         ld_pars: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
    """ Parse various limb-darkening laws into the non-linear form.
    """

    if ld_type not in ['uniform', 'linear', 'quadratic', 'nonlinear']:
        raise ValueError(f"Unknown limb-darkening law: {ld_type}")

    ld_idx = np.arange(5)
    ld_pars_ = np.zeros(5)

    if ld_type == 'uniform':
        pass
    if ld_type == 'linear':
        ld_pars_[2] = ld_pars[0]
    if ld_type == 'quadratic':
        ld_pars_[2] = ld_pars[0] + 2*ld_pars[1]
        ld_pars_[4] = -ld_pars[1]
    if ld_type == 'nonlinear':
        ld_pars_[1] = ld_pars[0]
        ld_pars_[2] = ld_pars[1]
        ld_pars_[3] = ld_pars[2]
        ld_pars_[4] = ld_pars[3]

    ld_pars_[0] = 1 - np.sum(ld_pars_[1:])

    return ld_idx, ld_pars_


def limb_darkening(mu: np.ndarray,
                   ld_type: str,
                   ld_pars: ArrayLike
                   ) -> np.ndarray:
    """ Compute limb-darkening values for various limb-darkening laws.
    """

    if ld_type not in ['uniform', 'linear', 'quadratic', 'nonlinear']:
        raise ValueError(f"Unknown limb-darkening law: {ld_type}")

    ld_idx, ld_pars = parse_limb_darkening(ld_type, ld_pars)

    ld_val = 1
    for i in range(1, 5):
        ld_val = ld_val - ld_pars[i]*(1 - mu**(i/2))

    return ld_val


def quadratic_limb_darkening(mu: np.ndarray,
                             ld_pars: ArrayLike
                             ) -> tuple[np.ndarray, float]:
    """ Compute limb-darkening values for the quadratic law, including a
        normalization factor. Used only by the pyspot model.
    """

    ld_val = 1 - ld_pars[0]*(1 - mu) - ld_pars[1]*(1 - mu)**2
    ld_norm = 0.5*(1 - ld_pars[0]/3 - ld_pars[1]/6)

    return ld_val, ld_norm


###########################################
# The pyspot and kipping starspot models. #
###########################################

def compute_spot_area(time: np.ndarray,
                      spot_params: Table,
                      min_area: float,
                      evolution: str = 'exponential'
                      ) -> np.ndarray:
    """ Compute the area of a single spot as a function of time.

    Parameters
    ----------
    time: np.ndarray
        Array of times at which to compute spot parameters.
    spot_params: Table
        A row from an astropy Table containg the parameters of the spot.
    min_area: float
        The smallest spot area to be considered in units of hemispheres.
    evolution: str
        The temporal evolution of the spot area (default: 'exponential').

    Returns
    -------
    area: np.ndarray
        The size of the spot in units of hemispheres.

    """

    # Extract spot parameters.
    amax = spot_params['A_MAX']*1e-6
    tmax = spot_params['T_MAX']
    decay_time = spot_params['TAU']
    emerge_time = decay_time/10.0

    # Compute the spot area.
    tmp1 = (time - tmax)/emerge_time
    tmp2 = (time - tmax)/decay_time

    if evolution == 'exponential':
        area = np.where(time < tmax, np.exp(-np.abs(tmp1)), np.exp(-np.abs(tmp2)))
    elif evolution == 'squared-exponential':
        area = np.where(time < tmax, np.exp(-0.5 * tmp1 ** 2), np.exp(-0.5 * tmp2 ** 2))
    else:
        raise ValueError(f"Unknown value for spot evolution profile: {evolution}")

    area = amax*area
    area = np.where(area < min_area, 0, area)

    return area


def compute_spot_location(time: np.ndarray,
                          spot_params: Table
                          ) -> tuple[np.ndarray, np.ndarray]:
    """ Compute the location of a single spot as a function of time.

    Parameters
    ----------
    time: np.ndarray
        Array of times at which to compute spot parameters.
    spot_params: Table
        A row from an astropy Table containg the parameters of the spot.

    Returns
    -------
    lat: np.ndarray
        The spot latitude.
    lon: np.ndarray
        The spot longitude.

    """

    # Extract spot parameters.
    lat = spot_params['LAT']
    lon = spot_params['LON']
    prot = spot_params['PROT']

    # Compute projected spot position as a function of time.
    lat = lat * np.ones_like(time)
    lon = lon + 360 * time / prot
    lon = np.mod(lon - 180, 360) - 180

    return lat, lon


def compute_spot_parameters(time: np.ndarray,
                            spot_params: Table,
                            inc_star: float = 90.,
                            min_area: float = 0.,
                            evolution: str = 'exponential'
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Compute the latitude, longitude, angular distance from the center of the
        stellar disk, and radius in angular units for a given starspot.

    Parameters
    ----------
    time: np.ndarray
        Array of times at which to compute spot parameters.
    spot_params: Table
        A row from an astropy Table containg the parameters of the spot.
    inc_star: float
        The stellar inclination in degrees (default: 90 degrees).
    min_area: float
        The smallest spot area to be considered in units of hemispheres
        (default: 0).
    evolution: str
        The temporal evolution of the spot area.

    Returns
    -------
    alpha: np.ndarray
        The spot radius in degrees.
    beta: np.ndarray
        The spot location in degrees from the center of the stellar disk.
    lat: np.ndarray
        The spot latitude.
    lon: np.ndarray
        The spot longitude.

    """

    # Compute the spot position and area.
    lat, lon = compute_spot_location(time, spot_params)
    area = compute_spot_area(time, spot_params, min_area, evolution=evolution)

    # Convert degrees to radians.
    inc_star = inc_star * DEG2RAD
    lat, lon = lat * DEG2RAD, lon * DEG2RAD

    # Compute the spot angle beta.
    cos_beta = np.cos(inc_star) * np.sin(lat) + np.sin(inc_star) * np.cos(lat) * np.cos(lon)
    beta = np.arccos(cos_beta)

    # Convert spot areas to spot radii.
    sa = 2 * np.pi * area  # Spot area in steradians.
    alpha = np.arccos(1 - sa / (2 * np.pi))  # Spot radius in radians.

    return alpha, beta, lat, lon


def filter_spots_table(time: np.ndarray,
                       spots_table: Table,
                       min_area: float = 1e-8,
                       evolution: str = 'exponential'
                       ) -> Table:
    """ Given an array of times compute which spots can contribute to the
        lightcurve.

    Parameters
    ----------
    time: np.ndarray
        Array of times at which to compute the lightcurve.
    spots_table: Table
        An astropy Table containg the parameters of the spots to be used.
    min_area: float
        The smallest spots area to be considered in units of hemispheres
        (default: 1e-8).
    evolution: str
        The temporal evolution of the spot area.

    Returns
    -------
    spots_table: Table
        A version of the input spots_table containing only the relevant spots.

    """

    tmin = np.amin(time)
    tmax = np.amax(time)

    area = compute_spot_area(tmin, spots_table, min_area=min_area, evolution=evolution)
    mask1 = (spots_table['T_MAX'] < tmin) & (area < min_area)

    area = compute_spot_area(tmax, spots_table, min_area=min_area, evolution=evolution)
    mask2 = (spots_table['T_MAX'] > tmax) & (area < min_area)

    mask = mask1 | mask2

    return spots_table[~mask]


def pyspot_spot_model(time: np.ndarray,
                      spots_table: Table,
                      inc_star: float = 90.,
                      ld_pars: ArrayLike = (0.6, 0.0),
                      min_area: float = 0.,
                      evolution: str = 'exponential'
                      ) -> np.ndarray:
    """ Computes the flux coming from a star covered in evolving starspots using
        the pyspot implementation with quadratic limb-darkening. Does not take
        overlapping spots into account.

    Parameters
    ----------
    time : np.ndarray
        The times for which to compute the flux.
    spots_table : astropy.table.Table
        The parameters of the star spots.
    inc_star : float
        The inclination of the star in degrees (default: 90.).
    ld_pars : tuple
        The limb-darkening parameters to use (default: (0.6, 0.0)).
    min_area : float
        The smallest spot areas to consider in hemispheres,
        when the spot area is below this threshold it will be set to zero (default: 0.).
    evolution : str
        Time evolution of the spot-area, either an 'exponential' or
        'squared-exponential' profile may be used.

    Returns
    -------
    flux : np.ndarray
        The stellar flux values."""

    spots_table = filter_spots_table(time,
                                     spots_table,
                                     min_area=min_area,
                                     evolution=evolution)

    flux = np.ones_like(time)
    for i in range(len(spots_table)):

        alpha, beta, lat, lon = compute_spot_parameters(time,
                                                        spots_table[i],
                                                        inc_star=inc_star,
                                                        min_area=min_area,
                                                        evolution=evolution)

        cos_beta = np.cos(beta)
        cos_beta = np.where(cos_beta < 0, 0, cos_beta)

        # Compute the flux.
        ld_val, ld_norm = quadratic_limb_darkening(cos_beta, ld_pars)
        flux = flux - (1 - np.cos(alpha)) * cos_beta * ld_val / ld_norm

    return flux


def zeta_func(x: np.ndarray) -> np.ndarray:
    """ The zeta function defined in Kipping 2012, equation 17.
    """

    val = np.cos(x)*np.heaviside(x, 0.5)*np.heaviside(np.pi/2 - x, 0.5) + np.heaviside(-x, 0.5)

    return val


def kipping_spot_model(time: np.ndarray,
                       spots_table: Table,
                       inc_star: float = 90.,
                       ld_type: str = 'linear',
                       ld_pars: ArrayLike = (0.6,),
                       min_area: float = 0.,
                       evolution: str = 'exponential'
                       ) -> np.ndarray:
    """ Computes the flux coming from a star covered in evolving starspots
        following Kipping 2012.

    Parameters
    ----------
    time : np.ndarray
        The times for which to compute the flux.
    spots_table : astropy.table.Table
        The parameters of the star spots.
    inc_star : float
        The inclination of the star in degrees (default: 90.).
    ld_type : str
        The limd-darkening law used (default: 'linear').
    ld_pars : tuple
        The limb-darkening parameters to use (default: (0.6,)).
    min_area : float
        The smallest spot areas to consider in hemispheres,
        when the spot area is below this threshold it will be set to zero (default: 0.).
    evolution : str
        Time evolution of the spot-area, either an 'exponential' or
        'squared-exponential' profile may be used.

    Returns
    -------
    flux : np.ndarray
        The stellar flux values.

    """

    spots_table = filter_spots_table(time,
                                     spots_table,
                                     min_area=min_area,
                                     evolution=evolution)

    ld_idx, ld_pars = parse_limb_darkening(ld_type, ld_pars)
    const1 = np.sum(ld_idx * ld_pars / (ld_idx + 4))

    flux = np.ones_like(time) - const1
    for i in range(len(spots_table)):

        alpha, beta, _, _ = compute_spot_parameters(time,
                                                    spots_table[i],
                                                    inc_star=inc_star,
                                                    min_area=min_area,
                                                    evolution=evolution)

        mask = alpha > 0
        if not np.any(mask):
            continue

        args, = np.where(mask)
        imin = np.amin(args)
        imax = np.amax(args) + 1  # Add 1 because slices are exclusive.

        alpha = alpha[imin:imax]
        beta = beta[imin:imax]

        # Convert to complex for use with Kipping 2012, equation 14.
        alpha = alpha.astype('complex256')
        beta = beta.astype('complex256')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            cot_alpha = 1/np.tan(alpha)
            cot_beta = 1/np.tan(beta)
            xi = np.sin(alpha)*np.arccos(-cot_alpha*cot_beta)
            psi = np.sqrt(1 - np.cos(alpha)**2/np.sin(beta)**2)
            a = np.arccos(np.cos(alpha)/np.sin(beta))
            b = np.cos(beta)*np.sin(alpha)*xi
            c = np.cos(alpha)*np.sin(beta)*psi

        alpha = alpha.real
        beta = beta.real
        sky_area = (a + b - c).real

        # The above equations seems to contain some singularities at beta=0,pi, this fixes them.
        sky_area = np.where(beta > np.pi/2 - alpha, sky_area, np.pi * np.sin(alpha)**2 * np.cos(beta))
        sky_area = np.where(beta < np.pi/2 + alpha, sky_area, 0)

        # Equations C23.
        zeta_neg = zeta_func((beta - alpha).real)
        zeta_pos = zeta_func((beta + alpha).real)

        denom = zeta_neg**2 - zeta_pos**2
        denom = np.where(denom < 1e-6, 1, denom)

        const2 = 0
        for j in ld_idx:
            exp = (j + 4)/2
            num = zeta_neg**exp - zeta_pos**exp
            const2 += (4*ld_pars[j])/(j + 4)*num/denom

        flux[imin:imax] = flux[imin:imax] - sky_area/np.pi*const2

    flux = flux/(1 - const1)

    return flux.astype('float64')


def main():
    return


if __name__ == '__main__':
    main()
