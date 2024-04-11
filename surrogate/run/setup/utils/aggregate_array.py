import pcraster as pcr

def aggregate_array(array_map: pcr.Field,
                    aggregate: str,
                    ids: pcr.Field) -> pcr.Field:
    
    if aggregate == "none" or aggregate == "":
        return array_map
    
    if aggregate == "mean" or aggregate == "average":
        array_map = pcr.ifthenelse(ids != 0,
                                   pcr.areaaverage(array_map, ids),
                                   array_map)
    elif aggregate == "sum" or aggregate == "total":
        array_map = pcr.ifthenelse(ids != 0,
                                   pcr.areatotal(array_map, ids),
                                   array_map)
    elif aggregate == "min" or aggregate == "minimum":
        array_map = pcr.ifthenelse(ids != 0,
                                   pcr.areaminimum(array_map, ids),
                                   array_map)
    elif aggregate == "max" or aggregate == "maximum":
        array_map = pcr.ifthenelse(ids != 0,
                                   pcr.areamaximum(array_map, ids),
                                   array_map)
    elif aggregate == "count":
        array_map = pcr.ifthenelse(ids != 0,
                                   pcr.areatotal(pcr.scalar(pcr.defined(array_map)), ids),
                                   0)
    else:
        raise ValueError("Aggregation {} unknown".format(aggregate))
    
    return array_map