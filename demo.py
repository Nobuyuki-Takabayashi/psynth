from psynth import psynth
import matplotlib.pyplot as plt
import os
import importlib
importlib.reload(psynth)
os.chdir("/users/kohnobuyuki/github/psynth")
filename1 = "data/farfield/patch_on_npc-f260a"
pa1=psynth.PhasedArray(frequency=5.75,element_num_x=8, element_num_y=8, element_interval_x=36.6, element_interval_y=36.6)
### Modify array weight ###
# pa1.beam_focus(800)
# pa1.beam_steering(theta_x=20)
# pa1.gauss_window(taper_db_x=10,taper_db_y=10)
# pa1.chebyshev_window(null_angle_y=10)

### Set element factor ###
pa1.set_element(filename=filename1)

### Farfield calculation ####
pa1.plot_farfield(angle=(-180,180))
plt.show()

### Nearfield calculation ###
pa1.set_input_power(50)
pa1.plot_nearfield(xlim=(-400,400),ylim=(-400,400), receiving_area=(146,146), z=800)
plt.show()
pa1.calculate_received_power(xlim=(-73,73),ylim=(-73,73),z=800)
