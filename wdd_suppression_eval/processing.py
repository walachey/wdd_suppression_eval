import pandas
import numpy as np


def aggregate_waggles_per_interval(
    waggles_df,
    indices=["interval_id", "was_clustered_to_dance", "was_suppression_target"],
):
    waggles_per_interval = waggles_df.pivot_table(
        index=indices, values="waggle_id", aggfunc="count"
    ).reset_index(level=(0, 1, 2))
    waggles_per_interval.columns = indices + ["number_of_waggles"]
    waggles_per_interval.head()

    return waggles_per_interval


def count_number_of_waggles_for_schedules(
    schedule_df, waggle_counts_df, additional_index=[]
):
    waggle_counts_df = waggle_counts_df.pivot_table(
        index=["interval_id"] + additional_index,
        values="number_of_waggles",
        aggfunc="sum",
    )
    waggle_counts_df = waggle_counts_df.reset_index(
        level=list(range(1 + len(additional_index)))
    )

    schedule_merged_df = pandas.merge(
        schedule_df, waggle_counts_df, how="left", on="interval_id"
    )
    controls = schedule_merged_df[schedule_merged_df.is_control][
        ["interval_id", "control_for", "number_of_waggles"] + additional_index
    ]
    long_form_df = schedule_merged_df.copy()

    schedule_merged_df = schedule_merged_df[~schedule_merged_df.is_control]
    print(
        "{} control intervals and {} regular intervals".format(
            controls.shape[0], schedule_merged_df.shape[0]
        )
    )

    schedule_merged_df = pandas.merge(
        schedule_merged_df,
        controls,
        left_on=["interval_id"] + additional_index,
        right_on=["control_for"] + additional_index,
        how="inner",
        suffixes=("", "_control"),
    )
    schedule_merged_df.drop("control_for_control", axis=1, inplace=True)

    to_drop = pandas.isnull(schedule_merged_df.number_of_waggles) | pandas.isnull(
        schedule_merged_df.number_of_waggles_control
    )
    print("Dropping {} / {} rows.".format(to_drop.sum(), schedule_merged_df.shape[0]))
    schedule_merged_df = schedule_merged_df[~to_drop]

    schedule_merged_df["waggle_number_reduction_factor"] = (
        schedule_merged_df["number_of_waggles"]
        / schedule_merged_df["number_of_waggles_control"]
    )

    if False:
        fig, ax = plt.subplots(figsize=(10, 1))
        sns.histplot(schedule_merged_df.number_of_waggles, binwidth=10)
        sns.histplot(schedule_merged_df.number_of_waggles_control, binwidth=10)
        plt.show()

    threshold_perc, threshold_perc2 = np.percentile(
        schedule_merged_df.number_of_waggles_control.values, (2.5, 5)
    )
    to_drop = schedule_merged_df.number_of_waggles_control < threshold_perc
    print(
        "Dropping {} / {} rows with too few waggles. Control percentiles: {}".format(
            to_drop.sum(), schedule_merged_df.shape[0], threshold_perc
        )
    )

    valid_intervals = set(schedule_merged_df.interval_id) | set(
        schedule_merged_df.interval_id_control
    )
    to_drop = ~long_form_df.interval_id.isin(valid_intervals)
    print(
        "Dropping {} / {} rows from long form.".format(
            to_drop.sum(), long_form_df.shape[0]
        )
    )
    long_form_df = long_form_df[~to_drop]

    return long_form_df, schedule_merged_df
