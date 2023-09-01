import matplotlib.transforms
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
import pandas


def plot_results_for(long_form_df, schedule_merged_df, additional_index=None):
    fig, ax = plt.subplots(figsize=(7, 3))

    hue = "is_control"
    if additional_index:
        long_form_df = long_form_df.copy()
        hue = "what"

        def to_text(row):
            t = ""
            value = row[additional_index]
            is_boolean = isinstance(value, bool) or isinstance(value, np.bool_)

            if is_boolean:
                if value:
                    t = additional_index
                else:
                    t = "other"
            else:
                t = value

            if row["is_control"]:
                t += "_control"
            return t

        long_form_df[hue] = long_form_df[[additional_index] + ["is_control"]].apply(
            to_text, axis=1
        )

    black_and_white = (
        additional_index is None or long_form_df[additional_index].dtype is bool
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

    additional_kws = {}
    if not black_and_white:
        additional_kws = dict(hue_order=sorted(set(long_form_df[hue].values)))
    sns.boxenplot(
        data=long_form_df,
        x="sound",
        y="number_of_waggles",
        hue=hue,
        orient="v",
        **additional_kws
    )
    plt.xlabel("Sound index")
    plt.ylabel("Number of individual detected waggle runs")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 3))

    additional_kws = {}
    if black_and_white:
        additional_kws = dict(
            color="white",
            box_kws=dict(edgecolor="k", facecolor="white", color="black"),
            flier_kws=dict(color="k"),
            line_kws=dict(color="k"),
        )
    else:
        additional_kws = dict(
            hue_order=sorted(set(schedule_merged_df[additional_index].values))
        )

    schedule_merged_df_filtered = schedule_merged_df[
        schedule_merged_df.too_few_data_points_in_control == False
    ]
    ax = sns.boxenplot(
        data=schedule_merged_df_filtered,
        x="sound",
        y="waggle_number_reduction_factor",
        orient="v",
        width=0.5,
        hue=additional_index,
        **additional_kws
    )

    if black_and_white:
        for i, box in enumerate(ax.patches):
            box.set_edgecolor("black")
            box.set_facecolor("white")

        plt.setp(ax.artists, edgecolor="k", facecolor="w")
        plt.setp(ax.lines, color="k")

    plt.axhline(1.0, linestyle=":", color="gray", alpha=0.5)
    plt.ylim(0, 2)
    plt.xlabel("Sound index")
    plt.ylabel("{\#$waggles_{suppression}$} / {\#$waggles_{control}$}")
    plt.tight_layout()
    plt.show()

    # rank test
    boxes_width = 0.5
    ax = sns.boxenplot(
        data=schedule_merged_df,
        x="sound",
        y="waggle_number_change",
        orient="v",
        width=boxes_width,
        hue=additional_index,
        **additional_kws
    )

    if black_and_white:
        for i, box in enumerate(ax.patches):
            box.set_edgecolor("black")
            box.set_facecolor("white")

        plt.setp(ax.artists, edgecolor="k", facecolor="w")
        plt.setp(ax.lines, color="k")

    if "hue_order" in additional_kws:
        additional_values = []
        for sound_index, (sound, sound_df) in enumerate(
            schedule_merged_df[["sound", additional_index]]
            .drop_duplicates()
            .groupby("sound")
        ):
            n_hues = sound_df.shape[0]
            hue_step = boxes_width / len(additional_kws["hue_order"])
            hue_offset = -n_hues / 2 * hue_step
            for hue in additional_kws["hue_order"]:
                hue_df = schedule_merged_df[schedule_merged_df.sound == sound]
                hue_df = hue_df[hue_df[additional_index] == hue]
                if hue_df.empty:
                    continue
                vals = hue_df.waggle_number_change_sign.values
                vals = vals[vals != 0]
                n_trials = vals.shape[0]
                n_successes = (vals < 0).sum()
                p_value = scipy.stats.binom_test(n_successes, n_trials)

                axis_coordinate = sound_index + hue_offset + hue_step / 2.0
                hue_offset += hue_step
                additional_values.append((axis_coordinate, sound, hue, p_value))
        additional_values = pandas.DataFrame(
            additional_values, columns=["x", "sound", additional_index, "p_value"]
        )

        for x, p in additional_values[["x", "p_value"]].itertuples(index=False):
            text = ""
            if p < 0.1:
                text = "*"
            elif p < 0.05:
                text = "**"
            elif p < 0.01:
                text = "***"
            else:
                continue

            trans = matplotlib.transforms.blended_transform_factory(
                ax.transData, ax.transAxes
            )
            ax.text(x, 1.0, s=text, transform=trans, horizontalalignment="center")
            ax.axvline(x, linewidth=0.5, linestyle=":", color="gray", zorder=-100)
    print(additional_values)

    plt.axhline(0.0, linestyle=":", color="gray", alpha=0.5)
    plt.yscale("symlog")
    plt.xlabel("Sound index")
    plt.ylabel("Difference of number of waggles from control phase")
    plt.tight_layout()
    plt.show()
