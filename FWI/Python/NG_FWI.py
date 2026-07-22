# Computes hourly FWI indices for an input hourly weather stream

### Import packages ###
import datetime
import logging
import argparse
from math import exp, log, pow, sqrt
from typing import Optional, Union
import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange

# Import from other CFFDRS code files
import util

logger = logging.getLogger(__name__)

### Variable Definitions ###
# Only change these if you know what you are doing, or
# reach out to the CFS Fire Danger Group for more info

# Startup moisture code values
FFMC_DEFAULT = 85.0
DMC_DEFAULT = 6.0
DC_DEFAULT = 15.0

# Precipitation intercept
FFMC_INTERCEPT = 0.5
DMC_INTERCEPT = 1.5
DC_INTERCEPT = 2.8

# Drying variables
DMC_REGRESSION = 2.22e-4
DC_REGRESSION = 1.5e-2
DMC_OFFSET_TEMP = 0.0
DC_OFFSET_TEMP = 0.0

# Grassland fuel load (grass_fuel_load, kg/m^2)
DEFAULT_GRASS_FUEL_LOAD = 0.35

# Transition from matted to standing grass in a calendar year (default July 1st)
GRASS_TRANSITION = True  # default True, False for GFMC to always be standing
MON_STANDING = 7
DAY_STANDING = 1

# For input data that can't be split by year (i.e. data runs between Dec 31 - Jan 1)
# If True, every station's data needs to be sequential (one continuous run)
CONTINUOUS_MULTIYEAR = False  # default False, True to not split by year

### Functions ###

##
# Convert to fine fuel moisture content (%)
# @param ffmc       Fine Fuel Moisture Code (FFMC)
# @return           fine fuel moisture content (%)
@njit(cache = True)
def ffmc_to_mcffmc(ffmc: float) -> float:
    """Convert Fine Fuel Moisture Code to fine fuel moisture content (%)."""
    C_FFMC = 14875 / 101
    return C_FFMC * (101 - ffmc) / (59.5 + ffmc)

##
# Convert to FFMC
# @param mcffmc     fine fuel moisture content (%)
# @return           FFMC
@njit(cache = True)
def mcffmc_to_ffmc(mcffmc: float) -> float:
    """Convert fine fuel moisture content (%) to Fine Fuel Moisture Code."""
    C_FFMC = 14875 / 101
    return 59.5 * (250 - mcffmc) / (C_FFMC + mcffmc)

##
# Convert to duff moisture content (%)
# @param dmc        Duff Moisture Code (DMC)
# @return           duff moisture content (%)
@njit(cache = True)
def dmc_to_mcdmc(dmc: float) -> float:
   """Convert Duff Moisture Code to duff moisture content (%)."""
   return (280 / exp(dmc / 43.43)) + 20

##
# Convert to DMC
# @param mcdmc      duff moisture content (%)
# @return           DMC
@njit(cache = True)
def mcdmc_to_dmc(mcdmc: float) -> float:
   """Convert duff moisture content (%) to Duff Moisture Code."""
   return 43.43 * log(280 / (mcdmc - 20))

##
# Convert to DC moisture content (%)
# @param dc         Drought Code (DC)
# @return           DC moisture content (%)
@njit(cache = True)
def dc_to_mcdc(dc: float) -> float:
   """Convert Drought Code to drought moisture content (%)."""
   return 400 * exp(-dc / 400)

##
# Convert to DC
# @param mcdc       DC moisture content (%)
# @return           DC
@njit(cache = True)
def mcdc_to_dc(mcdc: float) -> float:
   """Convert drought moisture content (%) to Drought Code."""
   return 400 * log(400 / mcdc)

##
# Calculate hourly fine fuel moisture content. Needs to be converted to get FFMC
#
# @param lastmc          Previous fine fuel moisture content (%)
# @param temp            Temperature (Celcius)
# @param rh              Relative Humidity (percent, 0-100)
# @param ws              Wind Speed (km/h)
# @param rain            Rainfall AFTER intercept (mm)
# @param time_increment  Duration of timestep (hr, default 1.0)
# @return                Hourly fine fuel moisture content (%)
@njit(cache = True)
def hourly_fine_fuel_moisture(
    lastmc: float,
    temp: float,
    rh: float,
    ws: float,
    rain: float,
    time_increment: float = 1.0
) -> float:
    """Calculate hourly fine fuel moisture content for one timestep."""
    rf = 42.5
    drf = 0.0579
    # use moisture directly instead of converting to/from ffmc
    # expects any rain intercept to already be applied
    mo = lastmc
    if rain != 0.0:
        # duplicated in both formulas, so calculate once
        # lastmc == mo, but use lastmc since mo changes after first equation
        mo += rf * rain * exp(-100.0 / (251 - lastmc)) * (1.0 - exp(-6.93 / rain))
        if lastmc > 150:
            mo += 0.0015 * pow(lastmc - 150, 2) * sqrt(rain)
        if mo > 250.0:
            mo = 250.0
    # duplicated in both formulas, so calculate once
    e1 = 0.18 * (21.1 - temp) * (1.0 - (exp(-0.115 * rh)))
    ed = 0.942 * pow(rh, 0.679) + (11.0 * exp((rh - 100) / 10.0)) + e1
    ew = 0.618 * pow(rh, 0.753) + (10.0 * exp((rh - 100) / 10.0)) + e1
    m = ew if (mo < ed) else ed
    if mo != ed:
        # these are the same formulas with a different value for a1
        a1 = (rh / 100.0) if (mo > ed) else ((100.0 - rh) / 100.0)
        k0_or_k1 = 0.424 * (1 - pow(a1, 1.7)) + (0.0694 * sqrt(ws) * (1 - pow(a1, 8)))
        kd_or_kw = 2.0 * drf * k0_or_k1 * exp(0.0365 * temp)
        m += (mo - m) * pow(10, (-kd_or_kw * time_increment))
    return m

##
# Calculate duff moisture content
#
# @param last_mcdmc             Previous duff moisture content (%)
# @param hr                     Time of day (hr)
# @param temp                   Temperature (Celcius)
# @param rh                     Relative Humidity (%)
# @param prec                   Hourly precipitation (mm)
# @param sunrise                Sunrise (hr)
# @param sunset                 Sunset (hr)
# @param prec_cumulative_prev   Cumulative precipitation since start of rain (mm)
# @param time_increment         Duration of timestep (hr, default 1.0)
# @return                       Hourly duff moisture content (%)
@njit(cache = True)
def duff_moisture_code(
    last_mcdmc: float,
    hr: float,
    temp: float,
    rh: float,
    prec: float,
    sunrise: float,
    sunset: float,
    prec_cumulative_prev: float,
    time_increment: float = 1.0  # duration of timestep, in hours
) -> float:
    """Calculate hourly duff moisture content for one timestep."""
    # wetting
    if prec_cumulative_prev + prec > DMC_INTERCEPT:  # prec_cumulative above threshold
        if prec_cumulative_prev <= DMC_INTERCEPT:  # just passed threshold
            rw = (prec_cumulative_prev + prec) * 0.92 - 1.27
        else:  # previously passed threshold
            rw = prec * 0.92

        last_dmc = mcdmc_to_dmc(last_mcdmc)
        if last_dmc <= 33:
            b = 100.0 / (0.3 * last_dmc + 0.5)
        elif last_dmc <= 65:
            b = -1.3 * log(last_dmc) + 14.0
        else:
            b = 6.2 * log(last_dmc) - 17.2

        mr = last_mcdmc + (1e3 * rw) / (b * rw + 48.77)
    else:  # prec_cumulative below threshold
        mr = last_mcdmc

    if mr > 300.0:
        mr = 300.0

    # drying
    # since sunset can be > 24, check hr + 24 (ignoring change between days)
    if (sunrise <= hr <= sunset or
        (hr < 6 and sunrise <= hr + 24 <= sunset)):  # daytime
        if temp < 0:
            temp = 0.0
        rk = DMC_REGRESSION * (temp + DMC_OFFSET_TEMP) * (100.0 - rh)
        invtau = rk / 43.43
        mcdmc = (mr - 20.0) * exp(-time_increment * invtau) + 20.0
    else:  # nighttime
        mcdmc = mr

    if mcdmc > 300.0:
        mcdmc = 300.0

    return(mcdmc)

##
# Calculate drought code moisture content
#
# @param last_mcdc              Previous drought code moisture content (%)
# @param hr                     Time of day (hr)
# @param temp                   Temperature (Celcius)
# @param prec                   Hourly precipitation (mm)
# @param sunrise                Sunrise (hr)
# @param sunset                 Sunset (hr)
# @param prec_cumulative_prev   Cumulative precipitation since start of rain (mm)
# @param time_increment         Duration of timestep (hr, default 1.0)
# @return                       Hourly drought code moisture content (%)
@njit(cache = True)
def drought_code(
    last_mcdc: float,
    hr: float,
    temp: float,
    prec: float,
    sunrise: float,
    sunset: float,
    prec_cumulative_prev: float,
    time_increment: float = 1.0
) -> float:
    """Calculate hourly drought moisture content for one timestep."""
    # wetting
    if prec_cumulative_prev + prec > DC_INTERCEPT:  # prec_cumulative above threshold
        if prec_cumulative_prev <= DC_INTERCEPT:  # just passed threshold
            rw = (prec_cumulative_prev + prec) * 0.83 - 1.27
        else:  # previously passed threshold
            rw = prec * 0.83
        mr = last_mcdc + 3.937 * rw / 2.0
    else:
        mr = last_mcdc

    if mr > 400.0:
        mr = 400.0

    # drying
    # since sunset can be > 24, check hr + 24 (ignoring change between days)
    if (sunrise <= hr <= sunset or
        (hr < 6 and sunrise <= hr + 24 <= sunset)):  # daytime
        if temp > 0:
            pe = DC_REGRESSION * (temp + DC_OFFSET_TEMP) + 3.0 / 16.0
        else:
            pe = 0
        invtau = pe / 400.0
        mcdc = mr * exp(-time_increment * invtau)
    else:  # nighttime
        mcdc = mr

    if mcdc > 400.0:
        mcdc = 400.0

    return(mcdc)

##
# Calculate Initial Spread Index (ISI)
#
# @param wind            Wind Speed (km/h)
# @param ffmc            Fine Fuel Moisure Code
# @return                Initial Spread Index
@njit(cache = True)
def initial_spread_index(ws: float, ffmc: float) -> float:
    """Calculate Initial Spread Index from wind speed and FFMC."""
    fm = ffmc_to_mcffmc(ffmc)
    fw = (12 * (1 - exp(-0.0818 * (ws - 28)))) if (40 <= ws) else exp(0.05039 * ws)
    ff = 91.9 * exp(-0.1386 * fm) * (1.0 + fm**5.31 / 4.93e07)
    isi = 0.208 * fw * ff
    return isi

##
# Calculate Build-up Index (BUI)
#
# @param dmc             Duff Moisture Code
# @param dc              Drought Code
# @return                Build-up Index
@njit(cache = True)
def buildup_index(dmc: float, dc: float) -> float:
    """Calculate Build-up Index from DMC and DC."""
    bui = 0.0 if (0 == dmc and 0 == dc) else (0.8 * dc * dmc / (dmc + 0.4 * dc))
    if bui < dmc:
        p = (dmc - bui) / dmc
        cc = 0.92 + pow(0.0114 * dmc, 1.7)
        bui = dmc - cc * p
        if bui <= 0:
            bui = 0.0
    return bui

##
# Calculate Fire Weather Index (FWI)
#
# @param isi             Initial Spread Index
# @param bui             Build-up Index
# @return                Fire Weather Index
@njit(cache = True)
def fire_weather_index(isi: float, bui: float) -> float:
    """Calculate Fire Weather Index from ISI and BUI."""
    if bui > 80:
        bb = 0.1 * isi * 1000 / (25 + 108.64 / exp(0.023 * bui))
    else:
        bb = 0.1 * isi * (0.626 * pow(bui, 0.809) + 2)
    fwi = bb if bb <= 1 else exp(2.72 * pow(0.434 * log(bb), 0.647))
    return fwi


@njit(cache = True)
def daily_severity_rating(fwi: float) -> float:
    """Calculate Daily Severity Rating from FWI."""
    return 0.0272 * pow(fwi, 1.77)

##
# Calculate hourly grassland fuel moisture content. Needs to be converted to get GFMC.
#
# @param lastmc          Previous grassland fuel moisture content (percent)
# @param temp            Temperature (Celcius)
# @param rh              Relative Humidity (percent, 0-100)
# @param ws              Wind Speed (km/h)
# @param rain            Rainfall (mm)
# @param solrad          Solar radiation (kW/m^2)
# @param load            Grassland Fuel Load (kg/m^2)
# @param time_increment  Duration of timestep (hr, default 1.0)
# @return                Grassland fuel moisture content (percent)
@njit(cache = True)
def hourly_grass_fuel_moisture(
    lastmc: float,
    temp: float,
    rh: float,
    ws: float,
    rain: float,
    solrad: float,
    load: float,
    time_increment: float = 1.0
) -> float:
    """Calculate hourly grassland fuel moisture content for one timestep."""

    rf = 0.27
    drf = 0.389633
    # use moisture directly instead of converting to/from ffmc
    # expects any rain intercept to already be applied
    mo = lastmc
    if rain != 0.0:
        mo += rain / load * 100.0
        if mo > 250:
            mo = 250.0
    # fuel temp from CEVW
    tf = temp + 17.9 * solrad * exp(-0.034 * ws)
    # fuel humidity
    if tf > temp:
        rhf = (rh * 6.107 * pow(10.0, 7.5 * temp / (temp + 237.0)) /
            (6.107 * pow(10.0, 7.5 * tf / (tf + 237.0))))
    else:
        rhf = rh
    # 18.85749879,18.85749879,7.77659602,21.24361786,19.22479551,19.22479551
    # duplicated in both formulas, so calculate once
    e1 = rf * (26.7 - tf) * (1.0 - (1.0 / exp(0.115 * rhf)))
    # GRASS EMC
    ed = 1.62 * pow(rhf, 0.532) + (13.7 * exp((rhf - 100) / 13.0)) + e1
    ew = 1.42 * pow(rhf, 0.512) + (12.0 * exp((rhf - 100) / 18.0)) + e1

    moed = mo - ed
    moew = mo - ew

    e = None
    a1 = None
    m = None
    moe = None

    if (moed == 0) or (moew >= 0 and moed < 0):
        m = mo
        if (moed == 0):
            e = ed
        if moew >= 0:
            e = ew
    else:
        if moed > 0:
            a1 = rhf / 100.0
            e = ed
            moe = moed
        else:
            a1 = (100.0 - rhf) / 100.0
            e = ew
            moe = moew
        if (a1 < 0):
            # avoids complex number in a1^1.7 xkd calculation
            a1 = 0
        xkd = (0.424 * (1 - a1 ** 1.7) + (0.0694 * sqrt(ws) * (1 - a1 ** 8)))
        xkd = xkd * drf * exp(0.0365 * tf)
        m = e + moe * exp(-1.0 * log(10.0) * xkd * time_increment)
    return m

@njit(cache = True)
def Pign(mc: float, wind2m: float, Cint: float, Cmc: float, Cws: float) -> float:
    """Calculate probability of sustained flaming from a small flaming brand."""
    #  Thisd is the general standard form for the probability of sustained flaming models for each FF cover type
    #     here :
    #       mc is cured moisture (%) in the litter fuels being ignited
    #       wind2m (km/h)  is the estimated 2 metre standard height for wind at hte site of the fire ignition
    #       Cint, Cmc and Cws   are coefficients for the standard Pign model form for a given FF cover type

    #       return >> is the Probability of Sustained flaming from a single small flaming ember/brand
    Prob = 1.0 / (1.0 + exp(-1.0 * (Cint + Cmc * mc + Cws * wind2m)))
    return Prob

@njit(cache = True)
def curing_factor(cur: float) -> float:
    """Calculate the grass spread adjustment for percent curing."""
    # cur is the percentage cure of the grass fuel complex.  100= fully cured
    #   ....The OPPOSITE (100-x) of greenness...

    #    This is the Cruz et al (2015) model with the original precision of the coefficent estimates
    #    and as in CSIRO code:https://research.csiro.au/spark/resources/model-library/csiro-grassland-models/
    cf = (1.036 / (1 + 103.989 * exp(-0.0996 * (cur - 20)))) if (cur >= 20.0) else 0.0
    return cf

@njit(cache = True)
def mcgfmc_to_gfmc(mc: float, cur: float, wind: float) -> float:
    """Convert cured grass moisture content to Grass Fuel Moisture Code."""
    #   THIS is the way to get the CODE value from cured grassland moisture
    #     IT takes fully cured grass moisture  (from the grass moisture model (100% cured)  (from the FMS...updated version of Wotton 2009)
    #        and a estimate of the fuel complex curing (as percent cured)
    #        and estimated wind speed (necessary for a calc
    #     and calculated the probability of sustainable flaming ignition  (a funciton of MC  and wind)
    #     THEN it accounts for curing effect on probability of fire spread sustainability, using the curing factor from Cruz et al (2015) for grass
    #     THEN from this calcuates an 'effective moisture content' which is the moisture content that would be required to achieve
    #        the curing adjusted probabiltiy of sustained flaming if one were calcuating it directly through the standard Pign equation.
    #     and THEN converts this effective moisture content to a CODE value via the FF-scale the FFMC uses for consistency

    #     relies on models of:
    #        Prob of sustained flaming for grass model (PsusF(grass)
    #        and  the curing_factor  function
    #        AND and estiamte of open 10 m to 2 m wind reduction (0.75)...hardcoded in here now.....

    # MC is moisture content (%)
    # cur=percent curing of the grassland  (%)
    # wind=  10 m open wind (km/h)

    #     currently (NOv 2023) the coefficients for the PsusF(grass) models are hardcoded into the GFMC function

    wind2m_open_factor = 0.75

    Intercept = 1.49
    Cmoisture = -0.11
    Cwind = 0.075
    # GRASS: these coefficients (above) could change down the road .....explicitly coded in above*/
    # /* convert from 10 m wind in open to 2 m wind in open COULD be updated */
    wind2m = wind2m_open_factor * wind

    probign = Pign(mc, wind2m, Intercept, Cmoisture, Cwind)

    # /* adjust ignition diretctly with the curing function on ROS */
    newPign = curing_factor(cur) * probign

    # /* now to back calc effective moisture - algebraically reverse the Pign equation*/
    # /* 250 is a saturation value just a check*/
    egmc = (
        ((log(newPign / (1.0 - newPign)) - Intercept - Cwind * wind2m) / Cmoisture)
        if (newPign > 0.0)
        else 250
    )

    if egmc > 250.0:
        egmc = 250.0
    return mcffmc_to_ffmc(egmc)

@njit(cache = True)
def matted_grass_spread_ROS(ws: float, mc: float, cur: float) -> float:
    """Calculate matted grass rate of spread (m/min)."""
    #  /*  CUT grass  Rate  of spread from cheney 1998  (and new CSIRO grassland code
    #   We use this for MATTED grass in our post-winter context
    #   --ws=10 m open wind km/h
    #   --mc = moisture content in  cured grass  (%)
    #   --cur = percentage of grassland cured  (%)
    #   output should be ROS in m/min   */
    fw = 16.67 * (
        (0.054 + 0.209 * ws) if (ws < 5) else (1.1 + 0.715 * (ws - 5.0) ** 0.844)
    )
    fm = (
        exp(-0.108 * mc)
        if mc < 12
        else (
            0.6838 - 0.0342 * mc
            if (mc < 20.0 and ws < 10.0)
            else 0.547 - 0.0228 * mc
            if (mc < 23.9 and ws >= 10.0)
            else 0.0
        )
    )
    if (fm < 0):
      fm = 0.0
    cf = curing_factor(cur)
    return fw * fm * cf

@njit(cache = True)
def standing_grass_spread_ROS(ws: float, mc: float, cur: float) -> float:
    """Calculate standing grass rate of spread (m/min)."""
    #  /*  standing grass  Rate  of spread from cheney 1998  (and new CSIRO grassland code)
    #   We use this for standing grass in our post-winter context
    #   ITS only the WIND function that chnges here between cut and standing
    #   --ws=10 m open wind km/h
    #   --mc = moisture content in grass  (%)
    #   --cur = percentage of grassland cured  (%)
    #   output should be ROS in m/min   */
    fw = 16.67 * (
        (0.054 + 0.269 * ws) if (ws < 5) else (1.4 + 0.838 * (ws - 5.0) ** 0.844)
    )
    fm = (
        exp(-0.108 * mc)
        if mc < 12
        else (
            0.6838 - 0.0342 * mc
            if (mc < 20.0 and ws < 10.0)
            else 0.547 - 0.0228 * mc
            if (mc < 23.9 and ws >= 10.0)
            else 0.0
        )
    )
    if (fm < 0):
      fm = 0.0
    cf = curing_factor(cur)
    return fw * fm * cf

##
# Calculate Grassland Spread Index (GSI)
#
# @param ws              Wind Speed (km/h)
# @param mc              Grass moisture content (percent)
# @param cur             Degree of curing (percent, 0-100)
# @param standing        Grass standing (True/False)
# @return                Grassland Spread Index
@njit(cache = True)
def grass_spread_index(ws: float, mc: float, cur: float, standing: bool) -> float:
    """Calculate Grassland Spread Index."""
    #  So we don't have to transition midseason between standing and matted grass spread rate models
    #  We will simply scale   GSI   by the average of the   matted and standing spread rates

    #now allowing switch between standing and matted grass
    ros = None
    if (standing):
      #standing
      ros = standing_grass_spread_ROS(ws, mc, cur)

    else:
      #matted
      ros = matted_grass_spread_ROS(ws, mc, cur)


    return 1.11 * ros

##
# Calculate Grassland Fire Weather Index
#
# @param gsi               Grassland Spread Index
# @param load              Grassland Fuel Load (kg/m^2)
# @return                  Grassland Fire Weather Index
@njit(cache = True)
def grass_fire_weather_index(gsi: float, load: float) -> float:
    """Calculate Grassland Fire Weather Index."""
    # this just converts back to ROS in m/min
    ros = gsi / 1.11
    Fint = 300.0 * load * ros
    if Fint > 100:
        return(log(Fint / 60.0) / 0.14)
    else:
        return(Fint / 25.0)

# Calculate number of drying "units" this hour contributes
@njit(cache = True)
def drying_units() -> float:  # temp, rh, ws, rain, solrad
    """Return the canopy drying units accumulated in one hour."""
    # for now, just add 1 drying "unit" per hour
    return 1.0

@njit(cache = True)
def _rain_since_intercept_reset(
    rain: float,
    rain_total_prev: float,
    drying_since_intercept: float
) -> tuple[float, float]:
    """Update the numerical canopy state without using a Python dictionary."""
    # for now, want 5 "units" of drying (which is 1 per hour to start)
    TARGET_DRYING_SINCE_INTERCEPT = 5.0
    if rain > 0 or rain_total_prev == 0:  # if raining, reset drying
        drying_since_intercept = 0.0
    else:
        drying_since_intercept += drying_units()
        if drying_since_intercept >= TARGET_DRYING_SINCE_INTERCEPT:
            # reset rain if intercept reset criteria met
            rain_total_prev = 0.0
            drying_since_intercept = 0.0
    return rain_total_prev, drying_since_intercept

def rain_since_intercept_reset(rain: float, canopy: dict[str, float]) -> dict[str, float]:
    """Update canopy rain interception state in place."""
    rain_total_prev, drying_since_intercept = _rain_since_intercept_reset(
        rain,
        canopy["rain_total_prev"],
        canopy["drying_since_intercept"]
    )
    canopy["rain_total_prev"] = rain_total_prev
    canopy["drying_since_intercept"] = drying_since_intercept
    return canopy


_GRID_OUTPUT_VARIABLES = (
    "mcffmc",
    "ffmc",
    "dmc",
    "dc",
    "isi",
    "bui",
    "fwi",
    "dsr",
    "mcgfmc_matted",
    "mcgfmc_standing",
    "gfmc",
    "gsi",
    "gfwi",
    "prec_cumulative",
    "canopy_drying",
)


@njit(cache = True, parallel = True)
def _hFWI_kernel(
    temp: np.ndarray,
    rh: np.ndarray,
    ws: np.ndarray,
    prec: np.ndarray,
    solrad: np.ndarray,
    sunrise: np.ndarray,
    sunset: np.ndarray,
    percent_cured: np.ndarray,
    grass_fuel_load: np.ndarray,
    hr: np.ndarray,
    standing: np.ndarray,
    mcffmc_old: np.ndarray,
    dmc_old: np.ndarray,
    dc_old: np.ndarray,
    mcgfmc_matted_old: np.ndarray,
    mcgfmc_standing_old: np.ndarray,
    prec_cumulative_old: np.ndarray,
    canopy_drying_old: np.ndarray,
) -> np.ndarray:
    """Calculate one or more grid cells in parallel, advancing along time."""
    n_time, n_cell = temp.shape
    results = np.empty((len(_GRID_OUTPUT_VARIABLES), n_time, n_cell))

    for cell in prange(n_cell):
        mcffmc = mcffmc_old[cell]
        mcdmc = dmc_to_mcdmc(dmc_old[cell])
        mcdc = dc_to_mcdc(dc_old[cell])
        mcgfmc_matted = mcgfmc_matted_old[cell]
        mcgfmc_standing = mcgfmc_standing_old[cell]
        rain_total_prev = prec_cumulative_old[cell]
        drying_since_intercept = canopy_drying_old[cell]

        for i in range(n_time):
            rain_total_prev, drying_since_intercept = _rain_since_intercept_reset(
                prec[i, cell], rain_total_prev, drying_since_intercept
            )

            # determine rain for ffmc and whether or not intercept should happen now
            if rain_total_prev + prec[i, cell] <= FFMC_INTERCEPT:
                rain_ffmc = 0.0
            elif rain_total_prev > FFMC_INTERCEPT:
                rain_ffmc = prec[i, cell]
            else:
                rain_ffmc = rain_total_prev + prec[i, cell] - FFMC_INTERCEPT

            mcffmc = hourly_fine_fuel_moisture(
                mcffmc,
                temp[i, cell],
                rh[i, cell],
                ws[i, cell],
                rain_ffmc
            )
            ffmc = mcffmc_to_ffmc(mcffmc)
            mcdmc = duff_moisture_code(
                mcdmc,
                hr[i],
                temp[i, cell],
                rh[i, cell],
                prec[i, cell],
                sunrise[i, cell],
                sunset[i, cell],
                rain_total_prev
            )
            dmc = mcdmc_to_dmc(mcdmc)
            mcdc = drought_code(
                mcdc,
                hr[i],
                temp[i, cell],
                prec[i, cell],
                sunrise[i, cell],
                sunset[i, cell],
                rain_total_prev
            )
            dc = mcdc_to_dc(mcdc)
            isi = initial_spread_index(ws[i, cell], ffmc)
            bui = buildup_index(dmc, dc)
            fwi = fire_weather_index(isi, bui)
            dsr = daily_severity_rating(fwi)

            rain_total_prev += prec[i, cell]
            mcgfmc_matted = hourly_grass_fuel_moisture(
                mcgfmc_matted,
                temp[i, cell],
                rh[i, cell],
                ws[i, cell],
                prec[i, cell],
                solrad[i, cell],
                grass_fuel_load[i, cell]
            )
            mcgfmc_standing = hourly_grass_fuel_moisture(
                mcgfmc_standing,
                temp[i, cell],
                rh[i, cell],
                ws[i, cell],
                prec[i, cell] * 0.06,
                0.0,
                grass_fuel_load[i, cell]
            )

            mcgfmc = mcgfmc_standing if standing[i] else mcgfmc_matted
            gfmc = mcgfmc_to_gfmc(
                mcgfmc, percent_cured[i, cell], ws[i, cell]
            )
            gsi = grass_spread_index(
                ws[i, cell], mcgfmc, percent_cured[i, cell], standing[i]
            )
            gfwi = grass_fire_weather_index(gsi, grass_fuel_load[i, cell])

            results[0, i, cell] = mcffmc
            results[1, i, cell] = ffmc
            results[2, i, cell] = dmc
            results[3, i, cell] = dc
            results[4, i, cell] = isi
            results[5, i, cell] = bui
            results[6, i, cell] = fwi
            results[7, i, cell] = dsr
            results[8, i, cell] = mcgfmc_matted
            results[9, i, cell] = mcgfmc_standing
            results[10, i, cell] = gfmc
            results[11, i, cell] = gsi
            results[12, i, cell] = gfwi
            results[13, i, cell] = rain_total_prev
            results[14, i, cell] = drying_since_intercept

    return results


def _hFWI(
    ds_wx: xr.Dataset,
    ffmc_old: Union[float, xr.DataArray] = FFMC_DEFAULT,
    mcffmc_old: Optional[Union[float, xr.DataArray]] = None,
    dmc_old: Union[float, xr.DataArray] = DMC_DEFAULT,
    dc_old: Union[float, xr.DataArray] = DC_DEFAULT,
    mcgfmc_matted_old: Union[float, xr.DataArray] = ffmc_to_mcffmc(FFMC_DEFAULT),
    mcgfmc_standing_old: Union[float, xr.DataArray] = ffmc_to_mcffmc(FFMC_DEFAULT),
    prec_cumulative: Union[float, xr.DataArray] = 0.0,
    canopy_drying: Union[float, xr.DataArray] = 0.0,
) -> xr.Dataset:
    """Calculate hourly FWI for an xarray grid.

    The calculation is sequential over ``time`` and parallel over all ``lat``
    and ``lon`` cells. Weather variables may already have all three dimensions
    or be broadcastable to them. Startup state may be scalar or a spatial
    DataArray.

    Args:
        ds_wx: Hourly weather with ``time``, ``lat``, and ``lon`` dimensions.
            Required variables are ``temp``, ``rh``, ``ws``, ``prec``,
            ``solrad``, ``sunrise``, ``sunset``, ``percent_cured``, and
            ``grass_fuel_load``.
        ffmc_old: Initial FFMC, as a scalar or spatial field.
        mcffmc_old: Initial fine-fuel moisture content. When provided, this is
            used instead of ``ffmc_old``.
        dmc_old: Initial DMC, as a scalar or spatial field.
        dc_old: Initial DC, as a scalar or spatial field.
        mcgfmc_matted_old: Initial matted-grass moisture content.
        mcgfmc_standing_old: Initial standing-grass moisture content.
        prec_cumulative: Initial cumulative precipitation for the active event.
        canopy_drying: Initial consecutive canopy-drying hours.

    Returns:
        A copy of ``ds_wx`` with hourly FWI state and index variables added.

    Raises:
        ValueError: If dimensions or variables are missing, time is not
            sequentially hourly, or the input spans multiple calendar years.
    """
    required_dims = {"time", "lat", "lon"}
    required_variables = {
        "temp", "rh", "ws", "prec", "solrad", "sunrise", "sunset",
        "percent_cured", "grass_fuel_load"
    }
    missing_dims = required_dims.difference(ds_wx.dims)
    missing_variables = required_variables.difference(ds_wx.data_vars)
    if missing_dims:
        raise ValueError("Missing required dimensions: " + ", ".join(sorted(missing_dims)))
    if missing_variables:
        raise ValueError(
            "Missing required variables: " + ", ".join(sorted(missing_variables))
        )

    time = pd.DatetimeIndex(ds_wx["time"].values)
    if len(time) == 0:
        raise ValueError("Expected at least one timestep")
    if len(time) > 1 and not np.all(np.diff(time.values) == np.timedelta64(1, "h")):
        raise ValueError("Expected hourly weather input to be sequential")
    if len(np.unique(time.year)) != 1:
        raise ValueError("_hFWI currently accepts one calendar year at a time")

    dims = ("time", "lat", "lon")
    shape = (ds_wx.sizes["time"], ds_wx.sizes["lat"], ds_wx.sizes["lon"])
    template = xr.DataArray(
        np.empty(shape),
        coords={dim: ds_wx.coords[dim] for dim in dims},
        dims=dims
    )

    def weather_values(name: str) -> np.ndarray:
        values, _ = xr.broadcast(ds_wx[name], template)
        return np.ascontiguousarray(values.transpose(*dims).values.reshape(shape[0], -1))

    spatial_template = template.isel(time = 0, drop = True)

    def state_values(value: Union[float, xr.DataArray]) -> np.ndarray:
        if isinstance(value, xr.DataArray):
            values, _ = xr.broadcast(value, spatial_template)
            return np.ascontiguousarray(values.transpose("lat", "lon").values.ravel())
        return np.full(shape[1] * shape[2], float(value))

    if mcffmc_old is None:
        ffmc_start = state_values(ffmc_old)
        C_FFMC = 14875 / 101
        mcffmc_start = C_FFMC * (101 - ffmc_start) / (59.5 + ffmc_start)
    else:
        mcffmc_start = state_values(mcffmc_old)

    DATE_GRASS_STANDING = datetime.date(time[0].year, MON_STANDING, DAY_STANDING)
    if DATE_GRASS_STANDING < time[0].date():
        DATE_GRASS_STANDING = datetime.date(
            time[0].year + 1, MON_STANDING, DAY_STANDING
        )
    standing = np.asarray([
        not GRASS_TRANSITION or timestamp.date() >= DATE_GRASS_STANDING
        for timestamp in time
    ])

    logger.debug(
        "Calculating hourly FWI for %s timesteps and %s grid cells",
        shape[0], shape[1] * shape[2]
    )
    values = _hFWI_kernel(
        weather_values("temp"),
        weather_values("rh"),
        weather_values("ws"),
        weather_values("prec"),
        weather_values("solrad"),
        weather_values("sunrise"),
        weather_values("sunset"),
        weather_values("percent_cured"),
        weather_values("grass_fuel_load"),
        time.hour.to_numpy(dtype = float),
        standing,
        np.ascontiguousarray(mcffmc_start),
        state_values(dmc_old),
        state_values(dc_old),
        state_values(mcgfmc_matted_old),
        state_values(mcgfmc_standing_old),
        state_values(prec_cumulative),
        state_values(canopy_drying),
    )

    result = ds_wx.copy()
    for i, name in enumerate(_GRID_OUTPUT_VARIABLES):
        result[name] = xr.DataArray(
            values[i].reshape(shape),
            coords=template.coords,
            dims=dims
        )
    return result

##
# Calculate hourly FWI indices from hourly weather stream for a single station
#
# @param    w                   hourly values weather stream
# @param    ffmc_old            previous value FFMC (this or mcffmc_old should be None)
# @param    mcffmc_old          previous value mcffmc (this or ffmc_old should be None)
# @param    dmc_old             previous value for DMC
# @param    dc_old              previous value for DC
# @param    mcgfmc_matted_old   previous value for matted mcgfmc
# @param    mcgfmc_standing_old previous value for standing mcgfmc
# @param    prec_cumulative     cumulative precipitation this rainfall
# @param    canopy_drying       consecutive hours of no rain
# @return                       hourly values FWI and weather stream
def _stnHFWI(
    w: pd.DataFrame,
    ffmc_old: Optional[Union[float, str]],
    mcffmc_old: Optional[Union[float, str]],
    dmc_old: float,
    dc_old: float,
    mcgfmc_matted_old: float,
    mcgfmc_standing_old: float,
    prec_cumulative: float,
    canopy_drying: float
) -> pd.DataFrame:
    """Calculate hourly FWI indices for one sequential station-year.

    Args:
        w: Sequential hourly weather for one station and one calendar year.
        ffmc_old: Initial FFMC, or ``None`` when ``mcffmc_old`` is supplied.
        mcffmc_old: Initial fine-fuel moisture content, or ``None`` when
            ``ffmc_old`` is supplied.
        dmc_old: Initial DMC.
        dc_old: Initial DC.
        mcgfmc_matted_old: Initial matted-grass moisture content.
        mcgfmc_standing_old: Initial standing-grass moisture content.
        prec_cumulative: Initial cumulative precipitation for the active event.
        canopy_drying: Initial consecutive canopy-drying hours.

    Returns:
        Hourly weather with the calculated FWI state and index columns.
    """
    if not CONTINUOUS_MULTIYEAR and len(w["yr"].unique()) != 1:
        logger.warning("_stnHFWI() function received more than one year")
    if not util.is_sequential_hours(w):
        raise RuntimeError("Expected hourly weather input to be sequential")
    if len(w["id"].unique()) != 1:
        raise RuntimeError("_stnHFWI() function only accepts a single station ID")
    if len(w["lat"].unique()) != 1:
        raise RuntimeError("Expected a single latitude (lat) each station year")
    if len(w["long"].unique()) != 1:
        raise RuntimeError("Expected a single longitude (long) each station year")
    if len(w["timezone"].unique()) != 1:
        raise RuntimeError("Expected a single UTC offset (timezone) each station year")
    if len(w["grass_fuel_load"].unique()) != 1:
        raise RuntimeError("Expected a single grass_fuel_load value each station year")
    r = w.copy()
    if mcffmc_old == None or mcffmc_old == "None":
        if ffmc_old == None or ffmc_old == "None":
            raise ValueError("Either ffmc_old OR mcffmc_old should be NA, not both")
        else:
            mcffmc = ffmc_to_mcffmc(ffmc_old)
    else:
        if ffmc_old == None or ffmc_old == "None":
            mcffmc = mcffmc_old
        else:
            raise ValueError("One of ffmc_old OR mcffmc_old should be NA, not neither")
    mcgfmc_matted = mcgfmc_matted_old
    mcgfmc_standing = mcgfmc_standing_old
    mcdmc = dmc_to_mcdmc(dmc_old)
    mcdc = dc_to_mcdc(dc_old)
    # FIX: just use loop for now so it matches C code
    canopy = {"rain_total_prev": prec_cumulative,
        "drying_since_intercept": canopy_drying}
    # transition btwn matted and standing grassland fuel
    # does not account for fire seasons continuous across multiple years
    DATE_GRASS_STANDING = datetime.date(r.at[0, "yr"], MON_STANDING, DAY_STANDING)
    if DATE_GRASS_STANDING < r.at[0, "date"]:  # use next year if date already passed
        DATE_GRASS_STANDING = datetime.date(r.at[0, "yr"] + 1,
            MON_STANDING, DAY_STANDING)
    results = []
    for i in range(len(r)):
        cur = r.iloc[i].to_dict()
        canopy = rain_since_intercept_reset(cur["prec"], canopy)
        # determine rain for ffmc and whether or not intercept should happen now
        if canopy["rain_total_prev"] + cur["prec"] <= FFMC_INTERCEPT:  # not enough rain
            rain_ffmc = 0.0
        elif canopy["rain_total_prev"] > FFMC_INTERCEPT:  # already saturated canopy
            rain_ffmc = cur["prec"]
        else:
            rain_ffmc = canopy["rain_total_prev"] + cur["prec"] - FFMC_INTERCEPT
        mcffmc = hourly_fine_fuel_moisture(
            mcffmc,
            cur["temp"],
            cur["rh"],
            cur["ws"],
            rain_ffmc
        )
        cur["mcffmc"] = mcffmc
        # convert to code for output, but keep using moisture % for precision
        cur["ffmc"] = mcffmc_to_ffmc(mcffmc)
        # not ideal, but at least encapsulates the code for each index
        mcdmc = duff_moisture_code(
            mcdmc,
            cur["hr"],
            cur["temp"],
            cur["rh"],
            cur["prec"],
            cur["sunrise"],
            cur["sunset"],
            canopy["rain_total_prev"]
        )
        cur["dmc"] = mcdmc_to_dmc(mcdmc)
        mcdc = drought_code(
            mcdc,
            cur["hr"],
            cur["temp"],
            cur["prec"],
            cur["sunrise"],
            cur["sunset"],
            canopy["rain_total_prev"]
        )
        cur["dc"] = mcdc_to_dc(mcdc)
        cur["isi"] = initial_spread_index(cur["ws"], cur["ffmc"])
        cur["bui"] = buildup_index(cur["dmc"], cur["dc"])
        cur["fwi"] = fire_weather_index(cur["isi"], cur["bui"])
        cur["dsr"] = daily_severity_rating(cur["fwi"])
        # done using canopy, can update for next step
        canopy["rain_total_prev"] += cur["prec"]
        # grass updates
        mcgfmc_matted = hourly_grass_fuel_moisture(
            mcgfmc_matted,
            cur["temp"],
            cur["rh"],
            cur["ws"],
            cur["prec"],
            cur["solrad"],
            cur["grass_fuel_load"]
        )
        #for standing grass we make a come very simplifying assumptions based on obs from the field (echo bay study):
        #standing not really affected by rain -- to introduce some effect we introduce just a simplification of the FFMC Rain absorption function
        #which averages 6% or so for rains  (<5mm...between 7% and 5%,    lower for larger rains)(NO intercept)
        #AND the solar radiation exposure is less, and the cooling from the wind is stronger.  SO we assume there is effectively no extra
        #heating of the grass from solar
        #working at the margin like this should make a nice bracket for moisture between the matted and standing that users can use
        #...reality will be in between the matt and stand
        mcgfmc_standing = hourly_grass_fuel_moisture(
            mcgfmc_standing,
            cur["temp"],
            cur["rh"],
            cur["ws"],
            cur["prec"] * 0.06,
            0.0,
            cur["grass_fuel_load"]
        )

        # check if matted to standing transition happened already
        if GRASS_TRANSITION and cur["date"] < DATE_GRASS_STANDING:
            standing = False
            mcgfmc = mcgfmc_matted
        else:
            standing = True
            mcgfmc = mcgfmc_standing

        cur["mcgfmc_matted"] = mcgfmc_matted
        cur["mcgfmc_standing"] = mcgfmc_standing
        cur["gfmc"] = mcgfmc_to_gfmc(mcgfmc, cur["percent_cured"], cur["ws"])
        cur["gsi"] = grass_spread_index(cur["ws"], mcgfmc, cur["percent_cured"], standing)
        cur["gfwi"] = grass_fire_weather_index(cur["gsi"], cur["grass_fuel_load"])
        # save wetting variables for timestep-by-timestep runs
        cur["prec_cumulative"] = canopy["rain_total_prev"]
        cur["canopy_drying"] = canopy["drying_since_intercept"]
        # append results for this row
        results.append(cur)

    r = pd.DataFrame(results)
    return r

##
# Calculate hourly FWI indices from hourly weather stream.
#
# @param    df_wx               hourly values weather stream
# @param    timezone            UTC offset (default None for column provided in df_wx)
# @param    ffmc_old            previous value for FFMC (startup 85, None for mcffmc_old)
# @param    mcffmc_old          previous value mcffmc (default None for ffmc_old input)
# @param    dmc_old             previous value for DMC (startup 6)
# @param    dc_old              previous value for DC (startup 15)
# @param    mcgfmc_matted_old   previous value for matted mcgfmc (startup FFMC = 85)
# @param    mcgfmc_standing_old previous value for standing mcgfmc (startup FFMC = 85)
# @param    prec_cumulative     cumulative precipitation this rainfall (default 0)
# @param    canopy_drying       consecutive hours of no rain (default 0)
# @param    silent              suppresses informative log messages (default False)
# @param    round_out           decimals to truncate output to, None for none (default 4)
# @return                       hourly values FWI and weather stream
def hFWI(
    df_wx: pd.DataFrame,
    timezone: Optional[float] = None,
    ffmc_old: Optional[Union[float, str]] = FFMC_DEFAULT,
    mcffmc_old: Optional[Union[float, str]] = None,
    dmc_old: float = DMC_DEFAULT,
    dc_old: float = DC_DEFAULT,
    mcgfmc_matted_old: float = ffmc_to_mcffmc(FFMC_DEFAULT),
    mcgfmc_standing_old: float = ffmc_to_mcffmc(FFMC_DEFAULT),
    prec_cumulative: float = 0.0,
    canopy_drying: float = 0,
    silent: bool = False,
    round_out: Optional[Union[int, str]] = 4
) -> pd.DataFrame:
    """Calculate hourly FWI indices from one or more station weather streams.

    Args:
        df_wx: Hourly station weather. Column names are handled case
            insensitively.
        timezone: UTC offset, or ``None`` to use the ``timezone`` column.
        ffmc_old: Initial FFMC, or ``None`` when ``mcffmc_old`` is supplied.
        mcffmc_old: Initial fine-fuel moisture content, or ``None`` when
            ``ffmc_old`` is supplied.
        dmc_old: Initial DMC.
        dc_old: Initial DC.
        mcgfmc_matted_old: Initial matted-grass moisture content.
        mcgfmc_standing_old: Initial standing-grass moisture content.
        prec_cumulative: Initial cumulative precipitation for the active event.
        canopy_drying: Initial consecutive canopy-drying hours.
        silent: Suppress informative progress messages when true.
        round_out: Decimal places in output, or ``None`` for full precision.

    Returns:
        Hourly weather with the calculated FWI state and index columns.
    """
    if not silent:
        logger.info("FWI2025 (%s)", util.version())

    wx = df_wx.copy()
    # make all column names lower case
    wx.columns = map(str.lower, wx.columns)
    og_names = wx.columns
    # check for required columns
    req_cols = ["lat", "long", "yr", "mon", "day", "hr", "temp", "rh", "ws", "prec"]
    for col in req_cols:
        if not col in wx.columns:
            raise RuntimeError("Missing required input column: " + col)
    # check timezone
    if timezone == None:
        if not "timezone" in wx.columns:
            raise RuntimeError("Either provide a timezone column or " +
                "specify argument in hFWI()")
    else:
        wx["timezone"] = float(timezone)
    # check for optional columns that have a default
    had_stn = "id" in og_names
    had_minute = "minute" in og_names
    if not had_stn:
        wx["id"] = "STN"
    if not had_minute:
        wx["minute"] = 0
    # check for optional columns that can be calculated
    had_timestamp = "timestamp" in og_names
    had_date = "date" in og_names
    if not had_timestamp:
        wx["timestamp"] = wx.apply(
            lambda row: datetime.datetime(
                row["yr"], row["mon"], row["day"], row["hr"], row["minute"]
                ), axis=1
            )
    if not had_date:
        wx["date"] = wx["timestamp"].apply(lambda ts: ts.date())
    if not "grass_fuel_load" in og_names:
        wx["grass_fuel_load"] = DEFAULT_GRASS_FUEL_LOAD
    if not "percent_cured" in og_names:
        wx["percent_cured"] = wx.apply(lambda row:
            util.seasonal_curing(row["yr"], row["mon"], row["day"]), axis = 1)
    if not "solrad" in wx.columns:
        needs_solrad = True
    else:
        needs_solrad = False
    # check for values outside valid ranges
    if any(isinstance(tz, str) for tz in wx["timezone"]):
        raise ValueError("UTC offset (timezone) should be a number, not a string")
    if not (all(wx["rh"] >= 0) and all(wx["rh"] <= 100)):
        raise ValueError("All relative humidity (rh) must be between 0-100%")
    if not all(wx["ws"] >= 0):
        raise ValueError("All wind speed (ws) must be >= 0")
    if not all(wx["prec"] >= 0):
        raise ValueError("All precipitation (prec) must be >= 0")
    if not (all(wx["mon"] >= 1) and all(wx["mon"] <= 12)):
        raise ValueError("All months (mon) must be between 1-12")
    if (not needs_solrad) and (not all(wx["solrad"] >= 0)):
        raise ValueError("All solar radiation (solrad) must be >= 0")
    if ("percent_cured" in og_names) and (not (
        all(wx["percent_cured"] >= 0) and all(wx["percent_cured"] <= 100))):
        raise ValueError("All percent_cured must be between 0-100%")
    if ("grass_fuel_load" in og_names) and (not (all(wx["grass_fuel_load"] > 0))):
        raise ValueError("All grass_fuel_load must be > 0")
    if not (all(wx["day"] >= 1) and all(wx["day"] <= 31)):
        raise ValueError("All day must be 1-31")
    if mcffmc_old == None or mcffmc_old == "None":
        if ffmc_old == None or ffmc_old == "None":
            raise ValueError("Either ffmc_old OR mcffmc_old should be None, not both")
        elif not (0 <= ffmc_old <= 101):
            raise ValueError("ffmc_old must be between 0-101")
    else:
        if ffmc_old == None or ffmc_old == "None":
            if not (0 <= mcffmc_old <= 250):
                raise ValueError("mcffmc_old must be between 0-250%")
        else:
            raise ValueError("One of ffmc_old OR mcffmc_old should be None, not neither")
    if not (dmc_old >= 0):
        raise ValueError("dmc_old must be >= 0")
    if not (dc_old >= 0):
        raise ValueError("dc_old must be >= 0")

    # log message with startup values used
    if not silent:
        logger.info(
            "Startup values: ffmc=%s, mcffmc=%s, dmc=%s, dc=%s, " +
            "mcgfmc_matted=%.4f, mcgfmc_standing=%.4f, " +
            "prec_cumulative=%s, canopy_drying=%s",
            ffmc_old, mcffmc_old, dmc_old, dc_old,
            mcgfmc_matted_old, mcgfmc_standing_old,
            prec_cumulative, canopy_drying
        )

    # loop over every station year if not continuous multiyear data
    results = None
    split = ["id", "yr"]
    if CONTINUOUS_MULTIYEAR:
        split = ["id"]  # if continuous multiyear data, only split by ID

    for idx, by_year in wx.groupby(split, sort = False):
        if not silent and not CONTINUOUS_MULTIYEAR:
            logger.info("Running %s for %s", idx[0], idx[1])
        elif not silent and CONTINUOUS_MULTIYEAR:
            logger.info("Running station %s", idx[0])
        logger.debug("Running for %s", idx)
        w = by_year.reset_index(drop = True)
        w = util.get_sunlight(w, get_solrad = needs_solrad)
        r = _stnHFWI(w, ffmc_old, mcffmc_old, dmc_old, dc_old,
            mcgfmc_matted_old, mcgfmc_standing_old,
            prec_cumulative, canopy_drying)
        results = pd.concat([results, r])
    results.reset_index(drop = True, inplace = True)

    # remove optional variables that we added
    if not had_stn:
        results = results.drop(columns = "id")
    if not had_minute:
        results = results.drop(columns = "minute")
    if not had_timestamp:
        results = results.drop(columns = "timestamp")
    if not had_date:
        results = results.drop(columns = "date")

    # round decimal places of output columns
    if not (round_out == None or round_out == "None"):
        outcols = ["sunrise", "sunset", "sunlight_hours",
            "mcffmc", "ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr",
            "mcgfmc_matted", "mcgfmc_standing", "gfmc", "gsi", "gfwi",
            "prec_cumulative", "canopy_drying"]
        if "solrad" not in og_names:
            outcols.insert(0, "solrad")
        if "percent_cured" not in og_names:
            outcols.insert(0, "percent_cured")
        if "grass_fuel_load" not in og_names:
            outcols.insert(0, "grass_fuel_load")
        results[outcols] = results[outcols].map(round, ndigits = int(round_out))

    return results

if __name__ == "__main__":
    # run hFWI by command line. run with option -h or --help to see usage
    parser = argparse.ArgumentParser(prog = "NG_FWI")
    # add all inputs to hFWI
    parser.add_argument("input", help = "Input csv data file")
    parser.add_argument("output", help = "Output csv file name and location")
    parser.add_argument("timezone", nargs = "?", default = None,
        help = "UTC offset (default None for column provided in input)")
    parser.add_argument("ffmc_old", nargs = "?", default = FFMC_DEFAULT,
        help = "Starting value for FFMC (startup 85, None for mcffmc_old)")
    parser.add_argument("mcffmc_old", nargs = "?", default = None,
        help = "Starting value for mcffmc (default None for ffmc_old input)")
    parser.add_argument("dmc_old", nargs = "?", default = DMC_DEFAULT, type = float,
        help = "Starting DMC (default 6)")
    parser.add_argument("dc_old", nargs = "?", default = DC_DEFAULT, type = float,
        help = "Starting DC (default 15)")
    parser.add_argument("mcgfmc_matted_old", nargs = "?",
        default = ffmc_to_mcffmc(FFMC_DEFAULT), type = float,
        help = "Starting mcgfmc for matted fuels (default mcffmc when FFMC = 85)")
    parser.add_argument("mcgfmc_standing_old", nargs = "?",
        default = ffmc_to_mcffmc(FFMC_DEFAULT), type = float,
        help = "Starting mcgfmc for standing fuels (default mcffmc when FFMC = 85)")
    parser.add_argument("prec_cumulative", nargs = "?", default = 0.0, type = float,
        help = "Cumulative precipitation of rain event (default 0)")
    parser.add_argument("canopy_drying", nargs = "?", default = 0.0, type = float,
        help = "Canopy drying, or consecutive hours of no prec (default 0)")
    parser.add_argument("-s", "--silent", action = "store_true")
    parser.add_argument("-r", "--round_out", default = 4, nargs = "?",
        help = "Decimal places to truncate outputs to, None for no rounding (default 4)")

    args = parser.parse_args()
    logging.basicConfig(
        level = logging.WARNING if args.silent else logging.INFO,
        format = "%(levelname)s %(name)s: %(message)s"
    )
    df_in = pd.read_csv(args.input)
    df_out = hFWI(df_in, args.timezone, args.ffmc_old, args.mcffmc_old,
        args.dmc_old, args.dc_old, args.mcgfmc_matted_old, args.mcgfmc_standing_old,
        args.prec_cumulative, args.canopy_drying, args.silent, args.round_out)
    df_out.to_csv(args.output, index = False)
