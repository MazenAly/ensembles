from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.max_open_warning"] = -1
# Print options
np.set_printoptions(precision=3)
# Slideshow
from notebook.services.config import ConfigManager
cm = ConfigManager()
cm.update('livereveal', {'width': 1440, 'height': 768, 'scroll': True, 'theme': 'simple'})
# Silence warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
# Helper functions
def plot_surface(clf, X, y, xlim=(-10, 10), ylim=(-10, 10), n_steps=250,
    subplot=None, show=True , est=0 , acc=0):
    if subplot is None:
        fig = plt.figure()
    else:
        plt.subplot(*subplot)
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n_steps),
    np.linspace(ylim[0], ylim[1], n_steps))
    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.RdBu_r)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title('Decision boundaries of a random forest of estimators number: ' + str(est) + " & CV:" + str(acc))
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    if show:
        plt.show()
        

X , y  = make_moons(n_samples=200, shuffle=True, noise=None, random_state=0)

plt.scatter(X[:,0],X[:,1] , c=y)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.show()


estimators_nums = [ 1,2,4,8,16,32,64,128]
for j , est_num in enumerate(estimators_nums):
    clf = RandomForestClassifier(n_estimators=est_num)
    clf = clf.fit(X, y)
    scores = cross_val_score(clf, X, y, cv=5)
    print est_num
    plot_surface(clf, X, y , xlim=(-1.5, 2.5), ylim=(-1, 1.5)  , show=True , est =est_num , acc=scores.mean())



