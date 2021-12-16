import psynth
import matplotlib.pyplot as plt
filename1 = 'data/farfield/patch_on_npc-f260a'
pa1=psynth.PhasedArray(frequency=5.74,element_num_x=16, element_num_y=16, element_interval_x=42.8, element_interval_y=42.8)
pa1.beam_focus(800)
pa1.set_element(filename=filename1)
pa1.plot_farfield(angle=(-180,180))
plt.show()
