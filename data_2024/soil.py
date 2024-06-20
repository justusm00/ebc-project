# thermal conductivity of water, quartz and air ( at 20 °C) in Watts per meter per Kelvin
kw = 0.57 
k_quartz = 8.8
k_clay = 2.92
n = 0.4 # porosity, literature value?


def get_soil_texture(depth):
    """Get soil texture of Goettingen Forest at specific depth according to
    H. Meesenburg, R. Brumme, C. Jacobsen, K.J. Meiwes, and J. Eichhorn - Soil Properties Ch. 3
    It is assumed that the sand and slit components refer to quartz
    TODO check this assumption

    Args:
        depth (_type_): _description_

    Returns:
        _type_: _description_
    """
    xs = 1 - n # soil content

    assert depth < 90
    
    if depth < 5:
        x_clay = xs * 0.36
        x_sand = xs * 0.64
    elif depth < 10:
        x_clay = xs * 0.41
        x_sand = xs * 0.59
    elif depth < 20:
        x_clay = xs * 0.39
        x_sand = xs * 0.61
    elif depth < 30:
        x_clay = xs * 0.54
        x_sand = xs * 0.46
    elif depth < 60:
        x_clay = xs * 0.3
        x_sand = xs * 0.7   
    else: 
        x_clay = xs * 0.23
        x_sand = xs * 0.77

    return x_clay, x_sand 


def get_thermal_conductivity(xw, depth):
    """Compute thermal conductivity according to De Vries model:
    DeVries D.A. (1963) Thermal Properties of Soils. In W.R. van Wijk (ed.) Physics of Plant Environment. North-Holland Publishing Company, Amsterdam.

    Original paper not available anywhere so see e.g. the following:
    https://www.sciencedirect.com/science/article/pii/0165232X81900410

    Args:
        water_content (_type_): _description_
    """
    # compute air content (= porosity minus water content)
    xa = n - xw
    # compute soil content (1 - porosity)
    xs = 1 - n
    # compute dry soil texture 
    x_clay, x_sand = get_soil_texture(depth)
    # compute dry soil thermal conductivity
    ks = x_clay * k_clay + x_sand * k_quartz
    # define weighting factors
    Fs = 1 / 3 * (2 / (1+ 0.125 * (ks / kw - 1)) + 1 / (1+ 0.75 * (ks / kw - 1)))
    if xw > 0.09:
        ga = 0.333 - xa / n * (0.333 - 0.035) # found conflicting info, best double check 
    else:
        ga = 0.013 + 0.944 * xw
    gc = 1 - 2 * ga
    # compute effective thermal conductivity of air due to humidity
    # ka = 0.0615 + 1.9 * xw
    ka = 0.025
    Fa = 1 / 3 * (2 / (1+ ga * (ka / kw - 1)) + 1 / (1+ gc * (ka / kw - 1)))
    
    # compute thermal conductivity
    k = (kw * xw + Fa * ka * xa + Fs * ks * xs) / (xw + Fa * xa + Fs * xs)
    return k



def fill_thermal_conductivity(df):
    """Compute thermal conductivity for all measurements at all depths

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    measure_indices = [1, 2, 3]
    depths = [5, 15, 30] # cm
    for measure_idx in measure_indices:
        for depth in depths:
            df[f"thermalConductivity_{measure_idx}_{depth}cm"] = df.apply(lambda x: get_thermal_conductivity(x[f"soilMoisture_{measure_idx}_{depth}cm"] / 100, depth), 
                        axis=1)
            
    return df


