"""Règles pN simplifiées CAMELYON17 (5 ganglions). Source : sujet SPE1, tableau pN."""
def stade_pn(classes):
    """classes : liste parmi 'negative','itc','micro','macro'. Les ITC ne comptent pas comme ganglion atteint."""
    classes = list(classes)
    macro, micro, itc = classes.count("macro"), classes.count("micro"), classes.count("itc")
    atteints = macro + micro
    if atteints == 0:
        return "pN0(i+)" if itc else "pN0"
    if macro == 0:
        return "pN1mi"
    return "pN1" if atteints <= 3 else "pN2"
