import numpy as np
import pandas as pd
import numba as nb


@nb.njit
def _calculate_indices_4D(to_firsts: np.ndarray,
                        to_seconds: np.ndarray,
                        to_thirds: np.ndarray,
                        to_fourths: np.ndarray,
                        from_firsts: np.ndarray,
                        from_seconds: np.ndarray,
                        from_thirds: np.ndarray,
                        from_fourths: np.ndarray,
                          extend_first: bool = True):

    indices_len = to_firsts.size
    indices = np.full(indices_len, fill_value=-1, dtype=np.int64)

    from_first_max = from_firsts.max()
    from_first_min = from_firsts.min()
    for to_index, (to_first, to_second, to_third, to_fourth) in enumerate(zip(to_firsts, to_seconds, to_thirds, to_fourths)):

        if extend_first:
            if to_first > from_first_max:
                to_first = from_first_max
            elif to_first < from_first_min:
                to_first = from_first_min

        first_sel = from_firsts == to_first
        seconds_sel = from_seconds == to_second
        third_sel = from_thirds == to_third
        fourth_sel = from_fourths == to_fourth
        from_sel = np.logical_and(np.logical_and(first_sel, seconds_sel), np.logical_and(third_sel, fourth_sel))
        from_index = np.where(from_sel)[0][0]
        indices[to_index] = from_index

    return indices


@nb.njit
def _calculate_indices_3D(to_firsts: np.ndarray,
                          to_seconds: np.ndarray,
                          to_thirds: np.ndarray,
                          from_firsts: np.ndarray,
                          from_seconds: np.ndarray,
                          from_thirds: np.ndarray,
                          extend_first: bool = True):

    indices_len = to_firsts.size
    indices = np.full(indices_len, fill_value=-1, dtype=np.int64)

    from_first_max = from_firsts.max()
    from_first_min = from_firsts.min()
    for to_index, (to_first, to_second, to_third) in enumerate(zip(to_firsts, to_seconds, to_thirds)):

        if extend_first:
            if to_first > from_first_max:
                to_first = from_first_max
            elif to_first < from_first_min:
                to_first = from_first_min

        first_sel = from_firsts == to_first
        seconds_sel = from_seconds == to_second
        third_sel = from_thirds == to_third
        from_sel = np.logical_and(first_sel, np.logical_and(seconds_sel, third_sel))
        from_index = np.where(from_sel)[0][0]
        indices[to_index] = from_index

    return indices


@nb.njit
def _calculate_indices_2D(to_firsts: np.ndarray,
                        to_seconds: np.ndarray,
                        from_firsts: np.ndarray,
                        from_seconds: np.ndarray,
                          extend_first: bool = True):

    indices_len = to_firsts.size
    indices = np.full(indices_len, fill_value=-1, dtype=np.int64)

    from_first_max = from_firsts.max()
    from_first_min = from_firsts.min()
    for to_index, (to_first, to_second) in enumerate(zip(to_firsts, to_seconds)):

        if extend_first:
            if to_first > from_first_max:
                to_first = from_first_max
            elif to_first < from_first_min:
                to_first = from_first_min

        first_sel = from_firsts == to_first
        second_sel = from_seconds == to_second
        from_sel = np.logical_and(first_sel, second_sel)
        from_index = np.where(from_sel)[0][0]
        indices[to_index] = from_index

    return indices


@nb.njit
def _calculate_indices_1D(to_firsts: np.ndarray,
                          from_firsts: np.ndarray,
                          extend_first: bool = True) -> np.ndarray:
    
    indices = np.full((to_firsts.size), fill_value=-1, dtype=np.int64)

    from_first_max = from_firsts.max()
    from_first_min = from_firsts.min()
    
    for to_index, to_first in enumerate(to_firsts):

        if extend_first:
            if to_first > from_first_max:
                to_first = from_first_max
            elif to_first < from_first_min:
                to_first = from_first_min

        from_sel = from_firsts == to_first
        from_index = np.where(from_sel)[0][0]
        indices[to_index] = from_index

    return indices


def calculate_date_indices(to_date: np.ndarray,
                           from_date: np.ndarray) -> tuple[np.ndarray, str]:

    from_years = np.array([datum.year for datum in from_date])
    from_months = np.array([datum.month for datum in from_date])
    from_days = np.array([datum.day for datum in from_date])

    years_u = pd.unique(from_years)
    months_u = pd.unique(from_months)
    days_u = pd.unique(from_days)

    indices = None
    frequency = ""
    if len(years_u) == 1:
        frequency += "single-year_"
        if len(days_u) == 1 and len(months_u) == 1:
            frequency += "yearly"
            to_years = np.array([datum.year for datum in to_date])

            indices = _calculate_indices_1D(to_firsts=to_years,
                                              from_firsts=from_years)
        else:
            if len(days_u) > 1:
                frequency += "daily"
                to_months = np.array([datum.month for datum in to_date])
                to_days = np.array([datum.day for datum in to_date])

                indices = _calculate_indices_2D(to_firsts=to_months,
                                                to_seconds=to_days,
                                                from_firsts=from_months,
                                                from_seconds=from_days,
                                                extend_first=False)

            elif len(months_u) > 1:
                frequency += "monthly"
                to_months = np.array([datum.month for datum in to_date])

                indices = _calculate_indices_1D(to_firsts=to_months,
                                                from_firsts=from_months)
            else:
                raise NotImplementedError("datetime indicies cannot be calculated: years {}, months {}, days {}".format(len(years_u),
                                                                                                                        len(
                                                                                                                            months_u),
                                                                                                                        len(days_u)))
    else:
        frequency += "multi-year_"
        if len(days_u) == 1 and len(months_u) == 1:
            frequency += "yearly"
            to_years = np.array([datum.year for datum in to_date])

            indices = _calculate_indices_1D(to_firsts=to_years,
                                              from_firsts=from_years)
        else:
            if len(days_u) > 1:
                frequency += "daily"
                to_years = np.array([datum.year for datum in to_date])
                to_months = np.array([datum.month for datum in to_date])
                to_days = np.array([datum.day for datum in to_date])

                indices = _calculate_indices_3D(to_firsts=to_years,
                                                to_seconds=to_months,
                                                to_thirds=to_days,
                                                from_firsts=from_years,
                                                from_seconds=from_months,
                                                from_thirds=from_days)

            elif len(months_u) > 1:
                frequency += "monthly"
                to_years = np.array([datum.year for datum in to_date])
                to_months = np.array([datum.month for datum in to_date])

                indices = _calculate_indices_2D(to_firsts=to_years,
                                                to_seconds=to_months,
                                                from_firsts=from_years,
                                                from_seconds=from_months)
            else:
                raise NotImplementedError("datetime indicies cannot be calculated: years {}, months {}, days {}".format(len(years_u),
                                                                                                                        len(
                                                                                                                            months_u),
                                                                                                                        len(days_u)))

    return indices, frequency
