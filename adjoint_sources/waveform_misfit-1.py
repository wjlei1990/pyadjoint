import pyadjoint
import matplotlib.pylab as plt
fig = plt.figure(figsize=(12, 6))
obs, syn = pyadjoint.utils.get_example_data()
obs = obs.select(component="Z")[0]
syn = syn.select(component="Z")[0]
start, end = pyadjoint.utils.EXAMPLE_DATA_PDIFF
pyadjoint.calculate_adjoint_source("waveform_misfit", obs, syn, 20.0, 100.0,
                                   start, end, adjoint_src=True, plot=fig)
plt.show()