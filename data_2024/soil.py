# thermal conductivity of water, quartz and air ( at 20 °C) in Watts per meter per Kelvin
kw = 0.57 
ks = 8.8
n = 0.4 # porosity

def get_thermal_conductivity(xw):
    """Compute thermal conductivity according to De Vries model

    Args:
        water_content (_type_): _description_
    """
    # compute air content (= porosity minus water content)
    xa = n - xw
    # compute soil content (1 - porosity)
    xs = 1 - n
    # define weighting factors
    Fs = 1 / 3 * (2 / (1+ 0.125 * (ks / kw - 1)) + 1 / (1+ 0.75 * (ks / kw - 1)))
    if xw > 0.09:
        ga = 0.333 - xa / n * (0.333 - 0.038) # found conflicting info, best double check 
    else:
        ga = 0.013 + 0.944 * xw
    gc = 1 - 2 * ga
    Fa = 1 / 3 * (2 / (1+ ga * (ks / kw - 1)) + 1 / (1+ gc * (ks / kw - 1)))
    # compute effective thermal conductivity of air due to humidity
    ka = 0.0615 + 1.9 * xw
    # compute thermal conductivity
    k = (kw * xw + Fa * ka * xa + Fs * ks * xs) / (xw + Fa * xa + Fs * xs)
    return k



print(get_thermal_conductivity(0.2))