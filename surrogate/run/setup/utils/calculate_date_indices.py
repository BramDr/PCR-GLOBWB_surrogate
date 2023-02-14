import numpy as np
import pandas as pd
import numba as nb


@nb.njit
def _calculate_indices_multiyear_dmy(to_years: np.ndarray,
                                     to_months: np.ndarray,
                                     to_days: np.ndarray,
                                     from_years: np.ndarray,
                                     from_months: np.ndarray,
                                     from_days: np.ndarray):

    indices_len = to_years.size
    indices = np.full(indices_len, fill_value=-1, dtype=np.int64)

    from_years_max = from_years.max()
    from_years_min = from_years.min()
    for to_index, (to_year, to_month, to_day) in enumerate(zip(to_years, to_months, to_days)):

        if to_year > from_years_max:
            to_year = from_years_max
        elif to_year < from_years_min:
            to_year = from_years_min

        year_sel = from_years == to_year
        month_sel = from_months == to_month
        day_sel = from_days == to_day
        from_sel = np.logical_and(year_sel, np.logical_and(month_sel, day_sel))
        from_index = np.where(from_sel)[0][0]
        indices[to_index] = from_index

    return indices


@nb.njit
def _calculate_indices_multiyear(to_years: np.ndarray,
                                 to_factors: np.ndarray,
                                 from_years: np.ndarray,
                                 from_factors: np.ndarray):

    indices_len = to_years.size
    indices = np.full(indices_len, fill_value=-1, dtype=np.int64)

    from_years_max = from_years.max()
    from_years_min = from_years.min()
    for to_index, (to_year, to_factor) in enumerate(zip(to_years, to_factors)):

        if to_year > from_years_max:
            to_year = from_years_max
        elif to_year < from_years_min:
            to_year = from_years_min

        year_sel = from_years == to_year
        factor_sel = from_factors == to_factor
        from_sel = np.logical_and(year_sel, factor_sel)
        from_index = np.where(from_sel)[0][0]
        indices[to_index] = from_index

    return indices


@nb.njit
def _calculate_indices_year_dmy(to_months: np.ndarray,
                                to_days: np.ndarray,
                                from_months: np.ndarray,
                                from_days: np.ndarray) -> np.ndarray:
    indices = np.full((to_months.size), fill_value=-1, dtype=np.int64)

    for to_index, (to_month, to_day) in enumerate(zip(to_months, to_days)):

        month_sel = from_months == to_month
        day_sel = from_days == to_day
        from_sel = np.logical_and(month_sel, day_sel)
        from_index = np.where(from_sel)[0][0]
        indices[to_index] = from_index

    return indices


@nb.njit
def _calculate_indices_year(to_factors: np.ndarray,
                            from_factors: np.ndarray) -> np.ndarray:
    indices = np.full((to_factors.size), fill_value=-1, dtype=np.int64)

    from_dmys_max = from_factors.max()
    from_dmys_min = from_factors.min()
    for to_index, to_factor in enumerate(to_factors):

        if to_factor > from_dmys_max:
            to_factor = from_dmys_max
        elif to_factor < from_dmys_min:
            to_factor = from_dmys_min

        from_sel = from_factors == to_factor
        from_index = np.where(from_sel)[0][0]
        indices[to_index] = from_index

    return indices


def calculate_date_indices(to_datetime: np.ndarray,
                           from_datetime: np.ndarray) -> tuple[np.ndarray, str]:

    from_years = np.array([datum.year for datum in from_datetime])
    from_months = np.array([datum.month for datum in from_datetime])
    from_days = np.array([datum.day for datum in from_datetime])

    years_u = pd.unique(from_years)
    months_u = pd.unique(from_months)
    days_u = pd.unique(from_days)

    indices = None
    frequency = ""
    if len(years_u) == 1:
        frequency += "single-year_"
        if len(days_u) == 1 and len(months_u) == 1:
            frequency += "yearly"
            to_years = np.array([datum.year for datum in to_datetime])

            indices = _calculate_indices_year(to_factors=to_years,
                                              from_factors=from_years)
        else:
            if len(days_u) > 1:
                frequency += "daily"
                to_months = np.array([datum.month for datum in to_datetime])
                to_days = np.array([datum.day for datum in to_datetime])

                indices = _calculate_indices_year_dmy(to_months=to_months,
                                                      to_days=to_days,
                                                      from_months=from_months,
                                                      from_days=from_days)

            elif len(months_u) > 1:
                frequency += "monthly"
                to_months = np.array([datum.month for datum in to_datetime])

                indices = _calculate_indices_year(to_factors=to_months,
                                                  from_factors=from_months)
            else:
                raise NotImplementedError("datetime indicies cannot be calculated: years {}, months {}, days {}".format(len(years_u),
                                                                                                                         len(months_u),
                                                                                                                         len(days_u)))
    else:
        frequency += "multi-year_"
        if len(days_u) == 1 and len(months_u) == 1:
            frequency += "yearly"
            to_years = np.array([datum.year for datum in to_datetime])

            indices = _calculate_indices_year(to_factors=to_years,
                                              from_factors=from_years)
        else:
            if len(days_u) > 1:
                frequency += "daily"
                to_years = np.array([datum.year for datum in to_datetime])
                to_months = np.array([datum.month for datum in to_datetime])
                to_days = np.array([datum.day for datum in to_datetime])

                indices = _calculate_indices_multiyear_dmy(to_years=to_years,
                                                           to_months=to_months,
                                                           to_days=to_days,
                                                           from_years=from_years,
                                                           from_months=from_months,
                                                           from_days=from_days)

            elif len(months_u) > 1:
                frequency += "monthly"
                to_years = np.array([datum.year for datum in to_datetime])
                to_months = np.array([datum.month for datum in to_datetime])

                indices = _calculate_indices_multiyear(to_years=to_years,
                                                       to_factors=to_months,
                                                       from_years=from_years,
                                                       from_factors=from_months)
            else:
                raise NotImplementedError("datetime indicies cannot be calculated: years {}, months {}, days {}".format(len(years_u),
                                                                                                                         len(months_u),
                                                                                                                         len(days_u)))

    return indices, frequency
