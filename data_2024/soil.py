# thermal conductivity of water, quartz and other materials in Watts per meter per Kelvin
k_water = 0.57 
k_quartz = 8.8
k_clay = 2.92

rho_bulk = 1.2 # bulk density in g/ cm^3 (from https://doi.org/10.1016/j.foreco.2019.04.022)
rho_quartz = 2.66
rho_clay = 2.65

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

    assert depth < 90
    
    if depth < 5:
        x_clay = 0.36
        x_quartz = 0.64
    elif depth < 10:
        x_clay = 0.41
        x_quartz = 0.59
    elif depth < 20:
        x_clay = 0.39
        x_quartz = 0.61
    elif depth < 30:
        x_clay = 0.54
        x_quartz = 0.46
    elif depth < 60:
        x_clay = 0.3
        x_quartz = 0.7   
    else: 
        x_clay = 0.23
        x_quartz = 0.77

    return x_quartz, x_clay



def get_thermal_conductivity(x_water, depth):
    """Compute thermal conductivity according to De Vries model:
    DeVries D.A. (1963) Thermal Properties of Soils. In W.R. van Wijk (ed.) Physics of Plant Environment. North-Holland Publishing Company, Amsterdam.

    Original paper not available anywhere so see e.g. the following:
    https://www.sciencedirect.com/science/article/pii/0165232X81900410

    Args:
        water_content (_type_): _description_
    """

    # get dry soil texture 
    x_quartz_dry, x_clay_dry = get_soil_texture(depth)

    # compute porosity
    n = 1 - rho_bulk / (x_quartz_dry * rho_quartz + x_clay_dry * rho_clay)

    # compute soil content (fraction of wet mass that is due to soil particles)
    xs = 1 - n

    # compute wet soil texture
    x_quartz = xs * x_quartz_dry
    x_clay = xs * x_clay_dry

    # compute air content (= porosity minus water content)
    x_air = n - x_water


    # compute air thermal conductivity at 4.4 degrees celsius (maybe adjust this for actual temperature?)
    if x_water > 0.09:
        k_air = 0.059 + 0.066
    else:
        k_air = 0.059 + 0.066 * x_water / 0.03
        
    # weighting factor for quartz
    F_quartz = 1 / 3 * (2 / (1+ 0.125 * (k_quartz / k_water - 1)) + 1 / (1+ 0.75 * (k_quartz / k_water - 1)))
    # weighting factor for clay
    F_clay = 1 / 3 * (2 / (1+ 0.125 * (k_clay / k_water - 1)) + 1 / (1+ 0.75 * (k_clay / k_water - 1)))

    # weighting factor for air
    if x_water > 0.09:
        ga = 0.333 - x_air / n * (0.333 - 0.035) # found conflicting info, best double check 
    else:
        ga = 0.013 + 0.944 * x_water
    gc = 1 - 2 * ga
    F_air = 1 / 3 * (2 / (1+ ga * (k_air / k_water - 1)) + 1 / (1+ gc * (k_air / k_water - 1)))
    
    # compute thermal conductivity
    k = (k_water * x_water + F_air * k_air * x_air + F_quartz * k_quartz * x_quartz + F_clay * k_clay * x_clay) / (x_water  + F_air * x_air + F_quartz * x_quartz + F_clay * x_clay)
    return k



def fill_thermal_conductivity(df):
    """Compute thermal conductivity for all measurements at all depths

    Args:
        df (_type_): goewa meteo dataframe

    Returns:
        _type_: _description_
    """
    measure_indices = [1, 2, 3]
    depths = [5, 15, 30] # cm
    for measure_idx in measure_indices:
        for depth in depths:
            df[f"thermalConductivity_{measure_idx}_{depth}cm"] = df.apply(lambda x: get_thermal_conductivity(x[f"soilMoisture_{measure_idx}_{depth}cm"] / 100, depth), axis=1)
            
    return df



def compute_soil_heatflux(df):
    """Compute soil heatflux 
    Compute gradient at every measuring site and then average
    Use only data at 5 and 15 cm depth

    Args:
        df (_type_): _description_
    """
    measure_indices = [1, 2, 3]
    # compute gradient at every measurement site
    for measure_idx in measure_indices:
        df[f"dTdz_5cm_15cm_{measure_idx}"] = (df[f"soilTemperature_{measure_idx}_15cm"] - df[f"soilTemperature_{measure_idx}_5cm"]) / 0.1

    # compute mean thermal conductivity at 5cm
    df["thermalConductivity_5cm_mean"] = df[["thermalConductivity_1_5cm",
                                             "thermalConductivity_2_5cm",
                                             "thermalConductivity_3_5cm"]].mean(axis=1)

    
    df["dTdz_mean"] = df[["dTdz_5cm_15cm_1", "dTdz_5cm_15cm_2", "dTdz_5cm_15cm_3"]].mean(axis = 1)


    df["soilHeatflux"] = -1 * df["thermalConductivity_5cm_mean"] * df["dTdz_mean"]
    return df


def compute_soil_heatflux2(df):
    """Compute soil heatflux 
    Average temperatures over all three sites and then compute gradient
    Use only data at 5 and 15 cm depth

    Args:
        df (_type_): _description_
    """
    measure_indices = [1, 2, 3]
    # compute gradient at every measurement site
    df["soilTemperature_5cm_mean"] = df[["soilTemperature_1_5cm", "soilTemperature_2_5cm", "soilTemperature_3_5cm"]].mean(axis=1)
    df["soilTemperature_15cm_mean"] = df[["soilTemperature_1_15cm", "soilTemperature_2_15cm", "soilTemperature_3_15cm"]].mean(axis=1)

    # compute mean thermal conductivity at 5cm
    df["thermalConductivity_5cm_mean"] = df[["thermalConductivity_1_5cm",
                                             "thermalConductivity_2_5cm",
                                             "thermalConductivity_3_5cm"]].mean(axis=1)


    
    df["dTdz_mean"] = (df["soilTemperature_15cm_mean"] - df["soilTemperature_5cm_mean"]) / 0.1


    df["soilHeatflux"] = -1 * df["thermalConductivity_5cm_mean"] * df["dTdz_mean"]
    return df




