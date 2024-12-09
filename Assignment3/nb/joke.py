import matplotlib.pyplot as plt

# Extract values for keys 6 and 4 from the given input
values_4 = [
    0.7142857142857143,
    0.7093912511471399,
    0.7098752919294185,
    0.7085428325834274,
    0.7089218081542841,
    0.7081999765224489,
    0.7088649452949395,
    0.7282334808295701,
    0.7266837289913638,
    0.7403952844076396,
    0.7522455584905567,
    0.7486002639739684,
    0.7212043075289462,
    0.7204143518151809,
    0.6984184743925814,
    0.6991608625004622,
    0.7046202707604178,
    0.705087519424045,
    0.7039887303130842,
    0.7042629780079563,
    0.7034647409599584,
    0.7027630246186082,
    0.7028690805977438,
    0.7033697312402032,
    0.7035426342982875,
]


values_6 = [
    -0.247557003257329,
    0.3086792844876582,
    0.44615803560797396,
    0.4770973402653025,
    0.4329093993058381,
    0.4783263030287461,
    0.4431254310962462,
    0.48752899089234303,
    0.4767257953428473,
    0.4876505008764359,
    0.4836981605496284,
    0.5046548026382134,
    0.4849841386054867,
    0.48318680217448257,
    0.5024686015697469,
    0.4933125575313567,
    0.49721973139859077,
    0.45855540352383556,
    0.48163453089255004,
    0.49898137577649404,
    0.5162651340017782,
    0.5036316841984809,
    0.508400115898791,
    0.5215539351614529,
    0.5251917777549948,
]


# Create a boxplot for comparison
plt.figure(figsize=(10, 6))
plt.boxplot(
    [values_4, values_6],
    labels=["Action 4", "Action 6"],
    patch_artist=True,
    showmeans=True,
)

# Adding titles and labels
plt.title("Convergence of actions for the first move", fontsize=14)
plt.ylabel("Mean value", fontsize=12)
plt.xlabel("Actions", fontsize=12)

# Show the plot
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("../imgs/joke.png")