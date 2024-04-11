import pcraster as pcr

def convert_array(array_map: pcr.Field,
                  conversion: str,
                  area: pcr.Field) -> pcr.Field:
    
    if conversion == "none" or conversion == "":
        return array_map
    
    if conversion == "to_flux":
        array_map = array_map * (area / 86400)
    elif conversion == "from_flux":
        array_map = array_map * (86400 / area)
    else:
        raise ValueError("Conversion {} unknown".format(conversion))
    
    return array_map