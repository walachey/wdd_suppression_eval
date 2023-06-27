import matplotlib.pyplot as plt
import seaborn as sns


def plot_results_for(long_form_df, schedule_merged_df, additional_index=None):
    fig, ax = plt.subplots(figsize=(7, 3))

    hue = "is_control"
    if additional_index:
        long_form_df = long_form_df.copy()
        hue = "what"

        def to_text(row):
            t = ""
            if row[additional_index]:
                t = additional_index
            else:
                t = "other"
            if row["is_control"]:
                t += "_control"
            return t

        long_form_df[hue] = long_form_df[[additional_index, "is_control"]].apply(
            to_text, axis=1
        )

    sns.countplot(
        data=long_form_df[["interval_id", hue, "sound"]].drop_duplicates(),
        hue=hue,
        x="sound",
        orient="v",
    )
    plt.xlabel("Sound index")
    plt.ylabel("Individual intervals with observations")
    plt.show()

    sns.boxenplot(
        data=long_form_df, x="sound", y="number_of_waggles", hue=hue, orient="v"
    )
    plt.xlabel("Sound index")
    plt.ylabel("Number of individual detected waggle runs")
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 3))
    ax = sns.boxenplot(
        data=schedule_merged_df,
        x="sound",
        y="waggle_number_reduction_factor",
        orient="v",
        color="white",
        width=0.5,
        hue=additional_index,
        box_kws=dict(edgecolor="k", facecolor="white", color="black"),
        flier_kws=dict(color="k"),
        line_kws=dict(color="k"),
    )

    for i, box in enumerate(ax.patches):
        box.set_edgecolor("black")
        box.set_facecolor("white")

    plt.setp(ax.artists, edgecolor="k", facecolor="w")
    plt.setp(ax.lines, color="k")

    plt.axhline(1.0, linestyle=":", color="gray", alpha=0.5)
    plt.ylim(0, 2)
    plt.xlabel("Sound index")
    plt.ylabel("{\#$waggles_{suppression}$} / {\#$waggles_{control}$}")
    plt.show()
