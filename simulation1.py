import matplotlib.pyplot as plt
import numpy as np
import doubleml as dml
from doubleml.datasets import make_plr_CCDDHNR2018
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
learner = RandomForestRegressor(n_estimators=500, max_features=20, max_depth=5, min_samples_leaf=2)
n = 500
ml_l = clone(learner)
ml_m = clone(learner)
theta = np.zeros(500)
for i in range(500):
    np.random.seed(2000+i)
    e1 = np.random.rand(n, 1)
    e2 = np.random.rand(n, 1)
    e3 = np.random.rand(n, 1)
    e4 = np.random.rand(n, 1)
    e5 = np.random.rand(n, 1)
    e6 = np.random.rand(n, 1)
    e7 = np.random.rand(n, 1)
    u = np.random.randn(n, 1)
    r1 = 1
    r2 = 0.5
    x = ((1 - r2) * e1 + r1 * e5 + r2 * e6) / (1 + r1)
    s = (1 - r2) * e7 + r2 * e6
    z1 = (e2 + r1 * e5) / (1 + r1)
    z2 = (e3 + r1 * e5) / (1 + r1)
    z3 = (e4 + r1 * e5) / (1 + r1)
    z = np.hstack((z1, z2, z3))
    y = 5 * x + np.sin(2 * np.pi * z1) + 0.2 * np.cos(2 * np.pi * z1) + 0.3 * np.sin(
        2 * np.pi * z1) ** 2 + 0.4 * np.sin(2 * np.pi * z1) ** 3 + 0.5 * np.cos(2 * np.pi * z1) ** 3 + 0.5 * (
        2 * z2 - 1) ** 2 + np.sin(2 * np.pi * z3) / (2 - np.sin(2 * np.pi * z3)) + u + s
    y = y.flatten()
    data = dml.DoubleMLData.from_arrays(z, y, x)
    dml_obj = dml.DoubleMLPLR(data, ml_l, ml_m, dml_procedure='dml1')
    dml_obj.fit()
    theta[i] = dml_obj.coef[0]
plt.hist(theta-5, density=True, bins=50)
plt.xlabel(r'$\hat{\beta}-\beta_0$')
plt.ylabel('Probability density')
plt.savefig('2.png')
print(np.sum(theta)/500)
