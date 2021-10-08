import matplotlib.pyplot as plt

def remove_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)



def remove_axis_3d(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    #ax.spines["right"].set_visible(False)
    #ax.spines["bottom"].set_visible(False)
    #ax.spines["top"].set_visible(False)
    #ax.spines["left"].set_visible(False)
    #ax.spines["front"].set_visible(False)
    #ax.spines["back"].set_visible(False)