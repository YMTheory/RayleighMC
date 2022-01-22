import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__" :
    df = pd.read_csv("./exp/multiangle.csv", delim_whitespace=True, names=["theta", "detang", "N", "Nerr"])
    print(df[df["theta"]==90])


