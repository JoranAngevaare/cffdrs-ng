import pandas as pd
import pytest

import NG_FWI


@pytest.fixture
def one_hour_weather():
    return pd.DataFrame(
        [
            {
                "lat": 52.0,
                "long": 5.0,
                "yr": 2026,
                "mon": 7,
                "day": 1,
                "hr": 12,
                "temp": 24.0,
                "rh": 35.0,
                "ws": 18.0,
                "prec": 0.0,
                "timezone": 2.0,
                "solrad": 0.65,
                "percent_cured": 80.0,
                "grass_fuel_load": 0.35,
            }
        ]
    )


def test_hfwi_calculates_characterized_values_for_one_location(one_hour_weather):
    result = NG_FWI.hFWI(one_hour_weather, silent=True, round_out=None)

    expected = {
        "sunrise": 5.390614255769746,
        "sunset": 22.05812999375557,
        "sunlight_hours": 16.667515737985823,
        "mcffmc": 14.179505946909696,
        "ffmc": 86.9045166296068,
        "dmc": 6.346319999999991,
        "dc": 15.547499999999923,
        "isi": 6.809600070553351,
        "bui": 6.336881347789444,
        "fwi": 5.847041015012055,
        "dsr": 0.6195065069519732,
        "mcgfmc_matted": 9.266095156395558,
        "mcgfmc_standing": 12.7277711227051,
        "gfmc": 83.105431633907,
        "gsi": 33.889390059401954,
        "gfwi": 28.416841145793654,
        "prec_cumulative": 0.0,
        "canopy_drying": 0.0,
    }

    assert len(result) == 1
    assert result.loc[0, list(expected)].to_dict() == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="hFWI reuses one scalar startup state for every station",
)
def test_hfwi_joined_locations_match_independent_location_runs(one_hour_weather):
    station_1 = one_hour_weather.assign(id="station-1")
    station_2 = one_hour_weather.assign(
        id="station-2",
        lat=52.5,
        long=5.5,
        temp=25.0,
        rh=40.0,
        ws=15.0,
    )
    joined_weather = pd.concat([station_1, station_2], ignore_index=True)

    station_1_startup = {"ffmc_old": 85.0, "dmc_old": 6.0, "dc_old": 15.0}
    station_2_startup = {"ffmc_old": 72.0, "dmc_old": 28.0, "dc_old": 110.0}

    independent_results = {
        "station-1": NG_FWI.hFWI(
            station_1,
            silent=True,
            round_out=None,
            **station_1_startup,
        ),
        "station-2": NG_FWI.hFWI(
            station_2,
            silent=True,
            round_out=None,
            **station_2_startup,
        ),
    }
    expected = pd.concat(independent_results.values(), ignore_index=True)

    # hFWI only permits one startup state for this joined call.
    joined_result = NG_FWI.hFWI(
        joined_weather,
        silent=True,
        round_out=None,
        **station_1_startup,
    )

    output_columns = [
        "id",
        "mcffmc",
        "ffmc",
        "dmc",
        "dc",
        "isi",
        "bui",
        "fwi",
        "dsr",
    ]

    pd.testing.assert_frame_equal(
        joined_result.loc[:, output_columns],
        expected.loc[:, output_columns],
    )
