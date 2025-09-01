from ladybug.epw import EPW
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from fcns import *

epw_path = "data/PRT_Porto.085450_IWEC.epw"

df,location_dict,df_stats=parse_with_ladybug(epw_path)

plt.figure(figsize=(12,4))
plt.plot(df.index,df['dry_bulb_temperature'])
plt.xlabel("Time")
plt.ylabel("Dry Bulb Temperature (°C)")
plt.title("Hourly Dry Bulb Temperature from EPW")
plt.show()
