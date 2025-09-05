import numpy as np


def AMcalculation(cooccurrence, X, Y, n):
    """cooccurrence = 共起頻度
    X = Unit X の頻度
    Y = Unit Y の頻度
    n = 全体の合計
    """
    o11 = cooccurrence
    r1 = X
    c1 = Y
    e11 = (r1 * c1) / n
    o12 = r1 - o11
    o21 = c1 - o11
    o22 = n - r1 - c1 + o11
    c2 = n - c1
    r2 = n - r1
    e12 = (r1 * c2) / n
    e21 = (r2 * c1) / n
    e22 = (r2 * c2) / n

    mi = np.log2(o11 / e11)  # MIの計算
    t_score = (o11 - e11) / (np.sqrt(o11))  # Tスコアの計算
    mi3 = np.log2(o11**3 / e11)  # MI3の計算
    mi2 = np.log2(o11**2 / e11)  # MI2の計算
    z_score = (o11 - e11) / np.sqrt(e11)  # Zスコアの計算
    simple_ll = 2 * (o11 * np.log(o11 / e11) - (o11 - e11))  # simple_llの計算
    dice = 2 * (o11 / (c1 + r1))  # Dice係数の計算
    log_dice = 14 + np.log2(dice)  # logDiceの計算
    # log-likelihoodの計算
    log_likelihood = (
        (2 * (o11 * np.log(o11 / e11)))
        + (2 * (o12 * np.log(o12 / e12)))
        + (2 * (o21 * np.log(o21 / e21)))
        + (2 * (o22 * np.log(o22 / e22)))
    )

    # chi-squaredの計算
    chi_squared = (
        ((o11 - e11) ** 2 / e11)
        + ((o12 - e12) ** 2 / e12)
        + ((o21 - e21) ** 2 / e21)
        + ((o22 - e22) ** 2 / e22)
    )

    if o11 < e11:
        log_likelihood = log_likelihood * -1
        chi_squared = chi_squared * -1
    else:
        pass

    deltaP_CueX = o11 / (o11 + o12) - o21 / (o21 + o22)
    deltaP_CueY = o11 / (o11 + o21) - o12 / (o12 + o22)

    AM_dict = {
        "MI": mi,
        "t-score": t_score,
        "MI3": mi3,
        "MI2": mi2,
        "z-score": z_score,
        "Dice": dice,
        "logDice": log_dice,
        "LL": log_likelihood,
        "X2": chi_squared,
        "ΔP(X->Y)": deltaP_CueX,
        "ΔP(Y->X)": deltaP_CueY,
    }

    return AM_dict


def deltaP_calculation(cooccurrence, X, Y, n):
    o11 = cooccurrence
    r1 = X
    c1 = Y
    o12 = r1 - o11
    o21 = c1 - o11
    o22 = n - r1 - c1 + o11
    deltaP_CueX = o11 / (o11 + o12) - o21 / (o21 + o22)
    deltaP_CueY = o11 / (o11 + o21) - o12 / (o12 + o22)

    return deltaP_CueX, deltaP_CueY
