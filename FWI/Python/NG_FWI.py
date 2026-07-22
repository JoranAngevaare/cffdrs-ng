"""Calculate hourly components of the Next Generation Fire Weather Index.

The public numerical functions in this module are deliberately thin wrappers.
Their private counterparts contain scalar, side-effect-free numerical kernels
that can be compiled by Numba and reused by future gridded orchestration code.
The pandas-based :func:`hFWI` interface remains available for station data.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from collections.abc import Callable, MutableMapping
from typing import Any, Optional, TypeVar, Union, cast

import numpy as np
import pandas as pd

import util

logger = logging.getLogger(__name__)

_Function = TypeVar("_Function", bound=Callable[..., Any])
_OptionalFloat = Optional[Union[float, str]]

try:
    from numba import njit as _numba_njit
except ImportError as exc:
    _numba_njit = None
    _NUMBA_IMPORT_ERROR: Optional[str] = str(exc)
else:
    _NUMBA_IMPORT_ERROR = None


def _njit(function: _Function) -> _Function:
    """Compile a numerical kernel when a compatible Numba is available."""

    if _numba_njit is None:
        return function
    return cast(_Function, _numba_njit(cache=True)(function))


NUMBA_AVAILABLE = _numba_njit is not None


# Startup moisture code values
FFMC_DEFAULT = 85.0
DMC_DEFAULT = 6.0
DC_DEFAULT = 15.0

# Precipitation intercepts
FFMC_INTERCEPT = 0.5
DMC_INTERCEPT = 1.5
DC_INTERCEPT = 2.8

# Drying variables
DMC_REGRESSION = 2.22e-4
DC_REGRESSION = 1.5e-2
DMC_OFFSET_TEMP = 0.0
DC_OFFSET_TEMP = 0.0

# Grassland fuel load (kg/m^2)
DEFAULT_GRASS_FUEL_LOAD = 0.35

# Transition from matted to standing grass (default July 1)
GRASS_TRANSITION = True
MON_STANDING = 7
DAY_STANDING = 1

# Treat data spanning New Year as one continuous station stream when true.
CONTINUOUS_MULTIYEAR = False


@_njit
def _ffmc_to_mcffmc(ffmc: float) -> float:
    """Numba-compatible FFMC-to-moisture conversion kernel."""

    coefficient = 14875.0 / 101.0
    return coefficient * (101.0 - ffmc) / (59.5 + ffmc)


def ffmc_to_mcffmc(ffmc: float) -> float:
    """Convert Fine Fuel Moisture Code (FFMC) to moisture content (%)."""

    return _ffmc_to_mcffmc(ffmc)


@_njit
def _mcffmc_to_ffmc(mcffmc: float) -> float:
    """Numba-compatible moisture-to-FFMC conversion kernel."""

    coefficient = 14875.0 / 101.0
    return 59.5 * (250.0 - mcffmc) / (coefficient + mcffmc)


def mcffmc_to_ffmc(mcffmc: float) -> float:
    """Convert fine-fuel moisture content (%) to FFMC."""

    return _mcffmc_to_ffmc(mcffmc)


@_njit
def _dmc_to_mcdmc(dmc: float) -> float:
    """Numba-compatible DMC-to-moisture conversion kernel."""

    return 280.0 / np.exp(dmc / 43.43) + 20.0


def dmc_to_mcdmc(dmc: float) -> float:
    """Convert Duff Moisture Code (DMC) to moisture content (%)."""

    return _dmc_to_mcdmc(dmc)


@_njit
def _mcdmc_to_dmc(mcdmc: float) -> float:
    """Numba-compatible moisture-to-DMC conversion kernel."""

    return 43.43 * np.log(280.0 / (mcdmc - 20.0))


def mcdmc_to_dmc(mcdmc: float) -> float:
    """Convert duff moisture content (%) to DMC."""

    return _mcdmc_to_dmc(mcdmc)


@_njit
def _dc_to_mcdc(dc: float) -> float:
    """Numba-compatible DC-to-moisture conversion kernel."""

    return 400.0 * np.exp(-dc / 400.0)


def dc_to_mcdc(dc: float) -> float:
    """Convert Drought Code (DC) to moisture content (%)."""

    return _dc_to_mcdc(dc)


@_njit
def _mcdc_to_dc(mcdc: float) -> float:
    """Numba-compatible moisture-to-DC conversion kernel."""

    return 400.0 * np.log(400.0 / mcdc)


def mcdc_to_dc(mcdc: float) -> float:
    """Convert drought-code moisture content (%) to DC."""

    return _mcdc_to_dc(mcdc)


@_njit
def _hourly_fine_fuel_moisture(
    lastmc: float,
    temp: float,
    rh: float,
    ws: float,
    rain: float,
    time_increment: float = 1.0,
) -> float:
    """Numba-compatible hourly fine-fuel moisture kernel."""

    rainfall_factor = 42.5
    drying_factor = 0.0579
    moisture = lastmc

    if rain != 0.0:
        moisture += (
            rainfall_factor
            * rain
            * np.exp(-100.0 / (251.0 - lastmc))
            * (1.0 - np.exp(-6.93 / rain))
        )
        if lastmc > 150.0:
            moisture += 0.0015 * (lastmc - 150.0) ** 2 * np.sqrt(rain)
        if moisture > 250.0:
            moisture = 250.0

    humidity_term = 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    drying_equilibrium = (
        0.942 * rh**0.679 + 11.0 * np.exp((rh - 100.0) / 10.0) + humidity_term
    )
    wetting_equilibrium = (
        0.618 * rh**0.753 + 10.0 * np.exp((rh - 100.0) / 10.0) + humidity_term
    )
    equilibrium = (
        wetting_equilibrium if moisture < drying_equilibrium else drying_equilibrium
    )

    if moisture != drying_equilibrium:
        humidity_ratio = (
            rh / 100.0 if moisture > drying_equilibrium else (100.0 - rh) / 100.0
        )
        base_rate = 0.424 * (1.0 - humidity_ratio**1.7) + (
            0.0694 * np.sqrt(ws) * (1.0 - humidity_ratio**8)
        )
        rate = 2.0 * drying_factor * base_rate * np.exp(0.0365 * temp)
        equilibrium += (moisture - equilibrium) * 10.0 ** (-rate * time_increment)

    return equilibrium


def hourly_fine_fuel_moisture(
    lastmc: float,
    temp: float,
    rh: float,
    ws: float,
    rain: float,
    time_increment: float = 1.0,
) -> float:
    """Calculate fine-fuel moisture content for one timestep.

    Args:
        lastmc: Moisture content from the previous timestep, in percent.
        temp: Air temperature in degrees Celsius.
        rh: Relative humidity in percent.
        ws: Wind speed in km/h.
        rain: Rain after canopy interception, in mm.
        time_increment: Timestep duration in hours.

    Returns:
        Fine-fuel moisture content in percent.
    """

    return _hourly_fine_fuel_moisture(lastmc, temp, rh, ws, rain, time_increment)


@_njit
def _duff_moisture_code(
    last_mcdmc: float,
    hr: float,
    temp: float,
    rh: float,
    prec: float,
    sunrise: float,
    sunset: float,
    prec_cumulative_prev: float,
    time_increment: float = 1.0,
) -> float:
    """Numba-compatible hourly duff-moisture kernel."""

    if prec_cumulative_prev + prec > DMC_INTERCEPT:
        if prec_cumulative_prev <= DMC_INTERCEPT:
            effective_rain = (prec_cumulative_prev + prec) * 0.92 - 1.27
        else:
            effective_rain = prec * 0.92

        last_dmc = _mcdmc_to_dmc(last_mcdmc)
        if last_dmc <= 33.0:
            coefficient = 100.0 / (0.3 * last_dmc + 0.5)
        elif last_dmc <= 65.0:
            coefficient = -1.3 * np.log(last_dmc) + 14.0
        else:
            coefficient = 6.2 * np.log(last_dmc) - 17.2

        moisture_after_rain = last_mcdmc + (1000.0 * effective_rain) / (
            coefficient * effective_rain + 48.77
        )
    else:
        moisture_after_rain = last_mcdmc

    if moisture_after_rain > 300.0:
        moisture_after_rain = 300.0

    is_daytime = sunrise <= hr <= sunset or (
        hr < 6.0 and sunrise <= hr + 24.0 <= sunset
    )
    if is_daytime:
        drying_temp = max(temp, 0.0)
        drying_rate = DMC_REGRESSION * (drying_temp + DMC_OFFSET_TEMP) * (100.0 - rh)
        inverse_time_constant = drying_rate / 43.43
        moisture = (moisture_after_rain - 20.0) * np.exp(
            -time_increment * inverse_time_constant
        ) + 20.0
    else:
        moisture = moisture_after_rain

    return min(moisture, 300.0)


def duff_moisture_code(
    last_mcdmc: float,
    hr: float,
    temp: float,
    rh: float,
    prec: float,
    sunrise: float,
    sunset: float,
    prec_cumulative_prev: float,
    time_increment: float = 1.0,
) -> float:
    """Calculate duff moisture content for one timestep.

    Precipitation is accumulated across a rain event before the DMC intercept is
    applied. Drying occurs only between sunrise and sunset.

    Args:
        last_mcdmc: Previous duff moisture content in percent.
        hr: Local hour of day.
        temp: Air temperature in degrees Celsius.
        rh: Relative humidity in percent.
        prec: Hourly precipitation in mm.
        sunrise: Local sunrise hour.
        sunset: Local sunset hour; values may exceed 24.
        prec_cumulative_prev: Rain accumulated before this timestep in mm.
        time_increment: Timestep duration in hours.

    Returns:
        Duff moisture content in percent.
    """

    return _duff_moisture_code(
        last_mcdmc,
        hr,
        temp,
        rh,
        prec,
        sunrise,
        sunset,
        prec_cumulative_prev,
        time_increment,
    )


@_njit
def _drought_code(
    last_mcdc: float,
    hr: float,
    temp: float,
    prec: float,
    sunrise: float,
    sunset: float,
    prec_cumulative_prev: float,
    time_increment: float = 1.0,
) -> float:
    """Numba-compatible hourly drought-moisture kernel."""

    if prec_cumulative_prev + prec > DC_INTERCEPT:
        if prec_cumulative_prev <= DC_INTERCEPT:
            effective_rain = (prec_cumulative_prev + prec) * 0.83 - 1.27
        else:
            effective_rain = prec * 0.83
        moisture_after_rain = last_mcdc + 3.937 * effective_rain / 2.0
    else:
        moisture_after_rain = last_mcdc

    if moisture_after_rain > 400.0:
        moisture_after_rain = 400.0

    is_daytime = sunrise <= hr <= sunset or (
        hr < 6.0 and sunrise <= hr + 24.0 <= sunset
    )
    if is_daytime:
        potential_evaporation = (
            DC_REGRESSION * (temp + DC_OFFSET_TEMP) + 3.0 / 16.0 if temp > 0.0 else 0.0
        )
        inverse_time_constant = potential_evaporation / 400.0
        moisture = moisture_after_rain * np.exp(-time_increment * inverse_time_constant)
    else:
        moisture = moisture_after_rain

    return min(moisture, 400.0)


def drought_code(
    last_mcdc: float,
    hr: float,
    temp: float,
    prec: float,
    sunrise: float,
    sunset: float,
    prec_cumulative_prev: float,
    time_increment: float = 1.0,
) -> float:
    """Calculate drought-code moisture content for one timestep.

    Args:
        last_mcdc: Previous drought-code moisture content in percent.
        hr: Local hour of day.
        temp: Air temperature in degrees Celsius.
        prec: Hourly precipitation in mm.
        sunrise: Local sunrise hour.
        sunset: Local sunset hour; values may exceed 24.
        prec_cumulative_prev: Rain accumulated before this timestep in mm.
        time_increment: Timestep duration in hours.

    Returns:
        Drought-code moisture content in percent.
    """

    return _drought_code(
        last_mcdc,
        hr,
        temp,
        prec,
        sunrise,
        sunset,
        prec_cumulative_prev,
        time_increment,
    )


@_njit
def _initial_spread_index(ws: float, ffmc: float) -> float:
    """Numba-compatible Initial Spread Index kernel."""

    moisture = _ffmc_to_mcffmc(ffmc)
    wind_factor = (
        12.0 * (1.0 - np.exp(-0.0818 * (ws - 28.0)))
        if ws >= 40.0
        else np.exp(0.05039 * ws)
    )
    fuel_factor = 91.9 * np.exp(-0.1386 * moisture) * (1.0 + moisture**5.31 / 4.93e07)
    return 0.208 * wind_factor * fuel_factor


def initial_spread_index(ws: float, ffmc: float) -> float:
    """Calculate Initial Spread Index from wind speed and FFMC."""

    return _initial_spread_index(ws, ffmc)


@_njit
def _buildup_index(dmc: float, dc: float) -> float:
    """Numba-compatible Build-up Index kernel."""

    buildup = 0.0 if dmc == 0.0 and dc == 0.0 else 0.8 * dc * dmc / (dmc + 0.4 * dc)
    if buildup < dmc:
        proportion = (dmc - buildup) / dmc
        coefficient = 0.92 + (0.0114 * dmc) ** 1.7
        buildup = dmc - coefficient * proportion
        if buildup <= 0.0:
            buildup = 0.0
    return buildup


def buildup_index(dmc: float, dc: float) -> float:
    """Calculate Build-up Index from DMC and DC."""

    return _buildup_index(dmc, dc)


@_njit
def _fire_weather_index(isi: float, bui: float) -> float:
    """Numba-compatible Fire Weather Index kernel."""

    if bui > 80.0:
        intermediate = 0.1 * isi * 1000.0 / (25.0 + 108.64 / np.exp(0.023 * bui))
    else:
        intermediate = 0.1 * isi * (0.626 * bui**0.809 + 2.0)
    if intermediate <= 1.0:
        return intermediate
    return np.exp(2.72 * (0.434 * np.log(intermediate)) ** 0.647)


def fire_weather_index(isi: float, bui: float) -> float:
    """Calculate Fire Weather Index from ISI and BUI."""

    return _fire_weather_index(isi, bui)


@_njit
def _daily_severity_rating(fwi: float) -> float:
    """Numba-compatible Daily Severity Rating kernel."""

    return 0.0272 * fwi**1.77


def daily_severity_rating(fwi: float) -> float:
    """Calculate Daily Severity Rating from FWI."""

    return _daily_severity_rating(fwi)


@_njit
def _hourly_grass_fuel_moisture(
    lastmc: float,
    temp: float,
    rh: float,
    ws: float,
    rain: float,
    solrad: float,
    load: float,
    time_increment: float = 1.0,
) -> float:
    """Numba-compatible hourly grass-fuel moisture kernel."""

    rainfall_factor = 0.27
    drying_factor = 0.389633
    moisture = lastmc
    if rain != 0.0:
        moisture += rain / load * 100.0
        if moisture > 250.0:
            moisture = 250.0

    fuel_temp = temp + 17.9 * solrad * np.exp(-0.034 * ws)
    if fuel_temp > temp:
        fuel_rh = (
            rh
            * 6.107
            * 10.0 ** (7.5 * temp / (temp + 237.0))
            / (6.107 * 10.0 ** (7.5 * fuel_temp / (fuel_temp + 237.0)))
        )
    else:
        fuel_rh = rh

    humidity_term = (
        rainfall_factor * (26.7 - fuel_temp) * (1.0 - 1.0 / np.exp(0.115 * fuel_rh))
    )
    drying_equilibrium = (
        1.62 * fuel_rh**0.532 + 13.7 * np.exp((fuel_rh - 100.0) / 13.0) + humidity_term
    )
    wetting_equilibrium = (
        1.42 * fuel_rh**0.512 + 12.0 * np.exp((fuel_rh - 100.0) / 18.0) + humidity_term
    )

    difference_dry = moisture - drying_equilibrium
    difference_wet = moisture - wetting_equilibrium
    if difference_dry == 0.0 or (difference_wet >= 0.0 and difference_dry < 0.0):
        return moisture

    if difference_dry > 0.0:
        humidity_ratio = fuel_rh / 100.0
        equilibrium = drying_equilibrium
        difference = difference_dry
    else:
        humidity_ratio = (100.0 - fuel_rh) / 100.0
        equilibrium = wetting_equilibrium
        difference = difference_wet

    humidity_ratio = max(humidity_ratio, 0.0)
    drying_rate = 0.424 * (1.0 - humidity_ratio**1.7) + 0.0694 * np.sqrt(ws) * (
        1.0 - humidity_ratio**8
    )
    drying_rate *= drying_factor * np.exp(0.0365 * fuel_temp)
    return equilibrium + difference * np.exp(
        -np.log(10.0) * drying_rate * time_increment
    )


def hourly_grass_fuel_moisture(
    lastmc: float,
    temp: float,
    rh: float,
    ws: float,
    rain: float,
    solrad: float,
    load: float,
    time_increment: float = 1.0,
) -> float:
    """Calculate grass-fuel moisture content for one timestep.

    Args:
        lastmc: Moisture content from the previous timestep, in percent.
        temp: Air temperature in degrees Celsius.
        rh: Relative humidity in percent.
        ws: Wind speed in km/h.
        rain: Rainfall in mm.
        solrad: Solar radiation in kW/m^2.
        load: Grass fuel load in kg/m^2.
        time_increment: Timestep duration in hours.

    Returns:
        Grass-fuel moisture content in percent.
    """

    return _hourly_grass_fuel_moisture(
        lastmc, temp, rh, ws, rain, solrad, load, time_increment
    )


@_njit
def _pign(mc: float, wind2m: float, cint: float, cmc: float, cws: float) -> float:
    """Numba-compatible sustained-flaming probability kernel."""

    return 1.0 / (1.0 + np.exp(-(cint + cmc * mc + cws * wind2m)))


def Pign(mc: float, wind2m: float, Cint: float, Cmc: float, Cws: float) -> float:
    """Calculate the probability of sustained flaming.

    Parameter names retain their historical capitalization for API compatibility.
    """

    return _pign(mc, wind2m, Cint, Cmc, Cws)


@_njit
def _curing_factor(cur: float) -> float:
    """Numba-compatible grass curing-factor kernel."""

    if cur < 20.0:
        return 0.0
    return 1.036 / (1.0 + 103.989 * np.exp(-0.0996 * (cur - 20.0)))


def curing_factor(cur: float) -> float:
    """Calculate the grass spread adjustment for percent curing."""

    return _curing_factor(cur)


@_njit
def _mcgfmc_to_gfmc(mc: float, cur: float, wind: float) -> float:
    """Numba-compatible grass-moisture-to-GFMC conversion kernel."""

    wind2m_open_factor = 0.75
    intercept = 1.49
    moisture_coefficient = -0.11
    wind_coefficient = 0.075
    wind2m = wind2m_open_factor * wind

    ignition_probability = _pign(
        mc, wind2m, intercept, moisture_coefficient, wind_coefficient
    )
    adjusted_probability = _curing_factor(cur) * ignition_probability
    if adjusted_probability > 0.0:
        effective_moisture = (
            np.log(adjusted_probability / (1.0 - adjusted_probability))
            - intercept
            - wind_coefficient * wind2m
        ) / moisture_coefficient
    else:
        effective_moisture = 250.0

    effective_moisture = min(effective_moisture, 250.0)
    return _mcffmc_to_ffmc(effective_moisture)


def mcgfmc_to_gfmc(mc: float, cur: float, wind: float) -> float:
    """Convert cured-grass moisture content to Grass Fuel Moisture Code."""

    return _mcgfmc_to_gfmc(mc, cur, wind)


@_njit
def _matted_grass_spread_ros(ws: float, mc: float, cur: float) -> float:
    """Numba-compatible matted-grass rate-of-spread kernel."""

    wind_factor = 16.67 * (
        0.054 + 0.209 * ws if ws < 5.0 else 1.1 + 0.715 * (ws - 5.0) ** 0.844
    )
    if mc < 12.0:
        moisture_factor = np.exp(-0.108 * mc)
    elif mc < 20.0 and ws < 10.0:
        moisture_factor = 0.6838 - 0.0342 * mc
    elif mc < 23.9 and ws >= 10.0:
        moisture_factor = 0.547 - 0.0228 * mc
    else:
        moisture_factor = 0.0
    moisture_factor = max(moisture_factor, 0.0)
    return wind_factor * moisture_factor * _curing_factor(cur)


def matted_grass_spread_ROS(ws: float, mc: float, cur: float) -> float:
    """Calculate matted-grass rate of spread in m/min."""

    return _matted_grass_spread_ros(ws, mc, cur)


@_njit
def _standing_grass_spread_ros(ws: float, mc: float, cur: float) -> float:
    """Numba-compatible standing-grass rate-of-spread kernel."""

    wind_factor = 16.67 * (
        0.054 + 0.269 * ws if ws < 5.0 else 1.4 + 0.838 * (ws - 5.0) ** 0.844
    )
    if mc < 12.0:
        moisture_factor = np.exp(-0.108 * mc)
    elif mc < 20.0 and ws < 10.0:
        moisture_factor = 0.6838 - 0.0342 * mc
    elif mc < 23.9 and ws >= 10.0:
        moisture_factor = 0.547 - 0.0228 * mc
    else:
        moisture_factor = 0.0
    moisture_factor = max(moisture_factor, 0.0)
    return wind_factor * moisture_factor * _curing_factor(cur)


def standing_grass_spread_ROS(ws: float, mc: float, cur: float) -> float:
    """Calculate standing-grass rate of spread in m/min."""

    return _standing_grass_spread_ros(ws, mc, cur)


@_njit
def _grass_spread_index(ws: float, mc: float, cur: float, standing: bool) -> float:
    """Numba-compatible Grassland Spread Index kernel."""

    rate_of_spread = (
        _standing_grass_spread_ros(ws, mc, cur)
        if standing
        else _matted_grass_spread_ros(ws, mc, cur)
    )
    return 1.11 * rate_of_spread


def grass_spread_index(ws: float, mc: float, cur: float, standing: bool) -> float:
    """Calculate Grassland Spread Index for matted or standing grass."""

    return _grass_spread_index(ws, mc, cur, standing)


@_njit
def _grass_fire_weather_index(gsi: float, load: float) -> float:
    """Numba-compatible Grassland Fire Weather Index kernel."""

    rate_of_spread = gsi / 1.11
    fire_intensity = 300.0 * load * rate_of_spread
    if fire_intensity > 100.0:
        return np.log(fire_intensity / 60.0) / 0.14
    return fire_intensity / 25.0


def grass_fire_weather_index(gsi: float, load: float) -> float:
    """Calculate Grassland Fire Weather Index from GSI and fuel load."""

    return _grass_fire_weather_index(gsi, load)


@_njit
def _drying_units() -> float:
    """Return the drying units accumulated in one hour."""

    return 1.0


def drying_units() -> float:
    """Return the number of canopy-drying units accumulated this hour."""

    return _drying_units()


@_njit
def _rain_since_intercept_reset(
    rain: float,
    rain_total_prev: float,
    drying_since_intercept: float,
) -> tuple[float, float]:
    """Numba-compatible canopy interception state transition."""

    target_drying_since_intercept = 5.0
    if rain > 0.0 or rain_total_prev == 0.0:
        drying_since_intercept = 0.0
    else:
        drying_since_intercept += _drying_units()
        if drying_since_intercept >= target_drying_since_intercept:
            rain_total_prev = 0.0
            drying_since_intercept = 0.0
    return rain_total_prev, drying_since_intercept


def rain_since_intercept_reset(
    rain: float,
    canopy: MutableMapping[str, float],
) -> MutableMapping[str, float]:
    """Update canopy rain and drying state for one timestep.

    The supplied mapping is mutated and returned for compatibility with the
    original station implementation.

    Args:
        rain: Rainfall during the current timestep in mm.
        canopy: Mapping containing ``rain_total_prev`` and
            ``drying_since_intercept``.

    Returns:
        The updated input mapping.
    """

    rain_total, drying = _rain_since_intercept_reset(
        rain,
        canopy["rain_total_prev"],
        canopy["drying_since_intercept"],
    )
    canopy["rain_total_prev"] = rain_total
    canopy["drying_since_intercept"] = drying
    return canopy


def _stnHFWI(
    w: pd.DataFrame,
    ffmc_old: _OptionalFloat,
    mcffmc_old: _OptionalFloat,
    dmc_old: float,
    dc_old: float,
    mcgfmc_matted_old: float,
    mcgfmc_standing_old: float,
    prec_cumulative: float,
    canopy_drying: float,
) -> pd.DataFrame:
    """Calculate an hourly FWI stream for one station and one year."""

    if not CONTINUOUS_MULTIYEAR and len(w["yr"].unique()) != 1:
        logger.warning("_stnHFWI received more than one year")
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

    result_frame = w.copy()
    if mcffmc_old is None or mcffmc_old == "None":
        if ffmc_old is None or ffmc_old == "None":
            raise ValueError("Either ffmc_old OR mcffmc_old should be NA, not both")
        mcffmc = ffmc_to_mcffmc(float(ffmc_old))
    else:
        if ffmc_old is not None and ffmc_old != "None":
            raise ValueError("One of ffmc_old OR mcffmc_old should be NA, not neither")
        mcffmc = float(mcffmc_old)

    mcgfmc_matted = mcgfmc_matted_old
    mcgfmc_standing = mcgfmc_standing_old
    mcdmc = dmc_to_mcdmc(dmc_old)
    mcdc = dc_to_mcdc(dc_old)
    canopy: MutableMapping[str, float] = {
        "rain_total_prev": prec_cumulative,
        "drying_since_intercept": canopy_drying,
    }

    date_grass_standing = datetime.date(
        int(result_frame.at[0, "yr"]), MON_STANDING, DAY_STANDING
    )
    if date_grass_standing < result_frame.at[0, "date"]:
        date_grass_standing = datetime.date(
            int(result_frame.at[0, "yr"]) + 1, MON_STANDING, DAY_STANDING
        )

    results: list[dict[str, Any]] = []
    for i in range(len(result_frame)):
        current = result_frame.iloc[i].to_dict()
        canopy = rain_since_intercept_reset(current["prec"], canopy)

        rain_total = canopy["rain_total_prev"] + current["prec"]
        if rain_total <= FFMC_INTERCEPT:
            rain_ffmc = 0.0
        elif canopy["rain_total_prev"] > FFMC_INTERCEPT:
            rain_ffmc = current["prec"]
        else:
            rain_ffmc = rain_total - FFMC_INTERCEPT

        mcffmc = hourly_fine_fuel_moisture(
            mcffmc,
            current["temp"],
            current["rh"],
            current["ws"],
            rain_ffmc,
        )
        current["mcffmc"] = mcffmc
        current["ffmc"] = mcffmc_to_ffmc(mcffmc)

        mcdmc = duff_moisture_code(
            mcdmc,
            current["hr"],
            current["temp"],
            current["rh"],
            current["prec"],
            current["sunrise"],
            current["sunset"],
            canopy["rain_total_prev"],
        )
        current["dmc"] = mcdmc_to_dmc(mcdmc)

        mcdc = drought_code(
            mcdc,
            current["hr"],
            current["temp"],
            current["prec"],
            current["sunrise"],
            current["sunset"],
            canopy["rain_total_prev"],
        )
        current["dc"] = mcdc_to_dc(mcdc)
        current["isi"] = initial_spread_index(current["ws"], current["ffmc"])
        current["bui"] = buildup_index(current["dmc"], current["dc"])
        current["fwi"] = fire_weather_index(current["isi"], current["bui"])
        current["dsr"] = daily_severity_rating(current["fwi"])

        canopy["rain_total_prev"] += current["prec"]
        mcgfmc_matted = hourly_grass_fuel_moisture(
            mcgfmc_matted,
            current["temp"],
            current["rh"],
            current["ws"],
            current["prec"],
            current["solrad"],
            current["grass_fuel_load"],
        )
        mcgfmc_standing = hourly_grass_fuel_moisture(
            mcgfmc_standing,
            current["temp"],
            current["rh"],
            current["ws"],
            current["prec"] * 0.06,
            0.0,
            current["grass_fuel_load"],
        )

        if GRASS_TRANSITION and current["date"] < date_grass_standing:
            standing = False
            mcgfmc = mcgfmc_matted
        else:
            standing = True
            mcgfmc = mcgfmc_standing

        current["mcgfmc_matted"] = mcgfmc_matted
        current["mcgfmc_standing"] = mcgfmc_standing
        current["gfmc"] = mcgfmc_to_gfmc(
            mcgfmc, current["percent_cured"], current["ws"]
        )
        current["gsi"] = grass_spread_index(
            current["ws"], mcgfmc, current["percent_cured"], standing
        )
        current["gfwi"] = grass_fire_weather_index(
            current["gsi"], current["grass_fuel_load"]
        )
        current["prec_cumulative"] = canopy["rain_total_prev"]
        current["canopy_drying"] = canopy["drying_since_intercept"]
        results.append(current)

    return pd.DataFrame(results)


def _hFWI(
    df_wx: pd.DataFrame,
    timezone: Optional[float] = None,
    ffmc_old: _OptionalFloat = FFMC_DEFAULT,
    mcffmc_old: _OptionalFloat = None,
    dmc_old: float = DMC_DEFAULT,
    dc_old: float = DC_DEFAULT,
    mcgfmc_matted_old: float = ffmc_to_mcffmc(FFMC_DEFAULT),
    mcgfmc_standing_old: float = ffmc_to_mcffmc(FFMC_DEFAULT),
    prec_cumulative: float = 0.0,
    canopy_drying: float = 0.0,
    silent: bool = False,
    round_out: Optional[Union[int, str]] = 4,
) -> pd.DataFrame:
    """Implement the pandas station-data orchestration for :func:`hFWI`."""

    if not silent:
        logger.info("FWI2025 (%s)", util.version())
        if _NUMBA_IMPORT_ERROR is not None:
            logger.info("Running without Numba acceleration: %s", _NUMBA_IMPORT_ERROR)

    weather = df_wx.copy()
    weather.columns = weather.columns.map(str.lower)
    original_names = weather.columns.copy()

    required_columns = [
        "lat",
        "long",
        "yr",
        "mon",
        "day",
        "hr",
        "temp",
        "rh",
        "ws",
        "prec",
    ]
    for column in required_columns:
        if column not in weather.columns:
            raise RuntimeError(f"Missing required input column: {column}")

    if timezone is None:
        if "timezone" not in weather.columns:
            raise RuntimeError(
                "Either provide a timezone column or specify argument in hFWI()"
            )
    else:
        weather["timezone"] = float(timezone)

    had_station = "id" in original_names
    had_minute = "minute" in original_names
    if not had_station:
        weather["id"] = "STN"
    if not had_minute:
        weather["minute"] = 0

    had_timestamp = "timestamp" in original_names
    had_date = "date" in original_names
    if not had_timestamp:
        weather["timestamp"] = weather.apply(
            lambda row: datetime.datetime(
                int(row["yr"]),
                int(row["mon"]),
                int(row["day"]),
                int(row["hr"]),
                int(row["minute"]),
            ),
            axis=1,
        )
    if not had_date:
        weather["date"] = weather["timestamp"].apply(lambda ts: ts.date())
    if "grass_fuel_load" not in original_names:
        weather["grass_fuel_load"] = DEFAULT_GRASS_FUEL_LOAD
    if "percent_cured" not in original_names:
        weather["percent_cured"] = weather.apply(
            lambda row: util.seasonal_curing(
                int(row["yr"]), int(row["mon"]), int(row["day"])
            ),
            axis=1,
        )
    needs_solrad = "solrad" not in weather.columns

    if any(isinstance(tz, str) for tz in weather["timezone"]):
        raise ValueError("UTC offset (timezone) should be a number, not a string")
    if not (weather["rh"].between(0.0, 100.0).all()):
        raise ValueError("All relative humidity (rh) must be between 0-100%")
    if not (weather["ws"] >= 0.0).all():
        raise ValueError("All wind speed (ws) must be >= 0")
    if not (weather["prec"] >= 0.0).all():
        raise ValueError("All precipitation (prec) must be >= 0")
    if not weather["mon"].between(1, 12).all():
        raise ValueError("All months (mon) must be between 1-12")
    if not needs_solrad and not (weather["solrad"] >= 0.0).all():
        raise ValueError("All solar radiation (solrad) must be >= 0")
    if (
        "percent_cured" in original_names
        and not weather["percent_cured"].between(0.0, 100.0).all()
    ):
        raise ValueError("All percent_cured must be between 0-100%")
    if (
        "grass_fuel_load" in original_names
        and not (weather["grass_fuel_load"] > 0.0).all()
    ):
        raise ValueError("All grass_fuel_load must be > 0")
    if not weather["day"].between(1, 31).all():
        raise ValueError("All day must be 1-31")

    if mcffmc_old is None or mcffmc_old == "None":
        if ffmc_old is None or ffmc_old == "None":
            raise ValueError("Either ffmc_old OR mcffmc_old should be None, not both")
        if not 0.0 <= float(ffmc_old) <= 101.0:
            raise ValueError("ffmc_old must be between 0-101")
    else:
        if ffmc_old is None or ffmc_old == "None":
            if not 0.0 <= float(mcffmc_old) <= 250.0:
                raise ValueError("mcffmc_old must be between 0-250%")
        else:
            raise ValueError(
                "One of ffmc_old OR mcffmc_old should be None, not neither"
            )
    if dmc_old < 0.0:
        raise ValueError("dmc_old must be >= 0")
    if dc_old < 0.0:
        raise ValueError("dc_old must be >= 0")

    if not silent:
        logger.info(
            "Startup state: ffmc=%s, mcffmc=%s, dmc=%s, dc=%s, "
            "mcgfmc_matted=%.4f, mcgfmc_standing=%.4f, "
            "prec_cumulative=%s, canopy_drying=%s",
            ffmc_old,
            mcffmc_old,
            dmc_old,
            dc_old,
            mcgfmc_matted_old,
            mcgfmc_standing_old,
            prec_cumulative,
            canopy_drying,
        )

    split = ["id"] if CONTINUOUS_MULTIYEAR else ["id", "yr"]
    station_results: list[pd.DataFrame] = []
    for index, station_weather in weather.groupby(split, sort=False):
        if not silent:
            logger.info("Running station group %s", index)
        logger.debug("Calculating FWI for station group %s", index)
        station_weather = station_weather.reset_index(drop=True)
        station_weather = util.get_sunlight(station_weather, get_solrad=needs_solrad)
        station_results.append(
            _stnHFWI(
                station_weather,
                ffmc_old,
                mcffmc_old,
                dmc_old,
                dc_old,
                mcgfmc_matted_old,
                mcgfmc_standing_old,
                prec_cumulative,
                canopy_drying,
            )
        )

    results = pd.concat(station_results, ignore_index=True)
    if not had_station:
        results = results.drop(columns="id")
    if not had_minute:
        results = results.drop(columns="minute")
    if not had_timestamp:
        results = results.drop(columns="timestamp")
    if not had_date:
        results = results.drop(columns="date")

    if round_out is not None and round_out != "None":
        output_columns = [
            "sunrise",
            "sunset",
            "sunlight_hours",
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
        ]
        if "solrad" not in original_names:
            output_columns.insert(0, "solrad")
        if "percent_cured" not in original_names:
            output_columns.insert(0, "percent_cured")
        if "grass_fuel_load" not in original_names:
            output_columns.insert(0, "grass_fuel_load")
        results[output_columns] = results[output_columns].map(
            round, ndigits=int(round_out)
        )

    return results


def hFWI(
    df_wx: pd.DataFrame,
    timezone: Optional[float] = None,
    ffmc_old: _OptionalFloat = FFMC_DEFAULT,
    mcffmc_old: _OptionalFloat = None,
    dmc_old: float = DMC_DEFAULT,
    dc_old: float = DC_DEFAULT,
    mcgfmc_matted_old: float = ffmc_to_mcffmc(FFMC_DEFAULT),
    mcgfmc_standing_old: float = ffmc_to_mcffmc(FFMC_DEFAULT),
    prec_cumulative: float = 0.0,
    canopy_drying: float = 0.0,
    silent: bool = False,
    round_out: Optional[Union[int, str]] = 4,
) -> pd.DataFrame:
    """Calculate hourly FWI components for one or more station-year streams.

    Args:
        df_wx: Hourly weather observations. Required columns are ``lat``,
            ``long``, ``yr``, ``mon``, ``day``, ``hr``, ``temp``, ``rh``,
            ``ws``, and ``prec``.
        timezone: UTC offset to apply to every row. When omitted, a ``timezone``
            column must be present.
        ffmc_old: Starting FFMC. Set to ``None`` when using ``mcffmc_old``.
        mcffmc_old: Starting fine-fuel moisture content in percent. Set to
            ``None`` when using ``ffmc_old``.
        dmc_old: Starting Duff Moisture Code.
        dc_old: Starting Drought Code.
        mcgfmc_matted_old: Starting matted-grass moisture content in percent.
        mcgfmc_standing_old: Starting standing-grass moisture content in percent.
        prec_cumulative: Rain accumulated in the active rain event, in mm.
        canopy_drying: Consecutive canopy-drying units.
        silent: Suppress informational log records when true.
        round_out: Decimal places for calculated output, or ``None`` for full
            precision.

    Returns:
        A copy of the hourly weather data with calculated FWI fields appended.

    Raises:
        RuntimeError: If required data or sequential station hours are missing.
        ValueError: If weather or startup values are outside accepted ranges.

    Note:
        Startup state is currently scalar and therefore shared by all station
        groups. Location-specific state is not yet supported by this wrapper.
    """

    return _hFWI(
        df_wx,
        timezone,
        ffmc_old,
        mcffmc_old,
        dmc_old,
        dc_old,
        mcgfmc_matted_old,
        mcgfmc_standing_old,
        prec_cumulative,
        canopy_drying,
        silent,
        round_out,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for hourly FWI calculation."""

    parser = argparse.ArgumentParser(prog="NG_FWI")
    parser.add_argument("input", help="Input CSV data file")
    parser.add_argument("output", help="Output CSV file name and location")
    parser.add_argument(
        "timezone",
        nargs="?",
        default=None,
        help="UTC offset (default: use the input timezone column)",
    )
    parser.add_argument(
        "ffmc_old",
        nargs="?",
        default=FFMC_DEFAULT,
        help="Starting FFMC (default: 85; use None with mcffmc_old)",
    )
    parser.add_argument(
        "mcffmc_old",
        nargs="?",
        default=None,
        help="Starting fine-fuel moisture (default: None)",
    )
    parser.add_argument("dmc_old", nargs="?", default=DMC_DEFAULT, type=float)
    parser.add_argument("dc_old", nargs="?", default=DC_DEFAULT, type=float)
    parser.add_argument(
        "mcgfmc_matted_old",
        nargs="?",
        default=ffmc_to_mcffmc(FFMC_DEFAULT),
        type=float,
    )
    parser.add_argument(
        "mcgfmc_standing_old",
        nargs="?",
        default=ffmc_to_mcffmc(FFMC_DEFAULT),
        type=float,
    )
    parser.add_argument("prec_cumulative", nargs="?", default=0.0, type=float)
    parser.add_argument("canopy_drying", nargs="?", default=0.0, type=float)
    parser.add_argument("-s", "--silent", action="store_true")
    parser.add_argument(
        "-r",
        "--round_out",
        default=4,
        nargs="?",
        help="Output decimal places, or None for no rounding (default: 4)",
    )
    return parser


def _main() -> None:
    """Run the command-line interface."""

    args = _build_argument_parser().parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.silent else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    input_frame = pd.read_csv(args.input)
    output_frame = hFWI(
        input_frame,
        args.timezone,
        args.ffmc_old,
        args.mcffmc_old,
        args.dmc_old,
        args.dc_old,
        args.mcgfmc_matted_old,
        args.mcgfmc_standing_old,
        args.prec_cumulative,
        args.canopy_drying,
        args.silent,
        args.round_out,
    )
    output_frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    _main()
