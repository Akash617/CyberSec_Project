from dython import nominal
import seaborn as sns
import matplotlib.pyplot as plt

def generate_heatmap(dataset):
    heatmap = nominal.associations(dataset, nom_nom_assoc="theil", compute_only=True)
    df = round(abs(heatmap["corr"]), 2)
    sns.heatmap(df, linecolor=[0,0,0], linewidth=0.5, annot=True)
    plt.show()
