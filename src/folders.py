import os

src = os.path.dirname(os.path.abspath(__file__))
codebase = os.path.abspath(os.path.join(src, os.pardir))
data = os.path.abspath(os.path.join(codebase, 'data'))
fromParker = os.path.abspath(os.path.join(data, 'from_Parker'))
exports = os.path.abspath(os.path.join(codebase, 'exports'))
conc_trajectories = os.path.abspath(os.path.join(exports, 'concentration trajectories'))
corrmats = os.path.abspath(os.path.join(exports, 'correlation matrices'))
heatmaps = os.path.abspath(os.path.join(exports, 'heatmaps'))