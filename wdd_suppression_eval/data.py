import datetime
import json
import pytz
import numpy as np
import pandas
import pathlib
import tqdm
import bb_behavior.waggledance.worldmapping
import os


def load_experimental_config(comb_config_path, interval_duration):
    with open(comb_config_path, "r") as f:
        comb_config_data = json.load(f)

    interval_delta = datetime.timedelta(minutes=interval_duration)
    raw_schedule = comb_config_data["experiment"]["timeslots"]
    experiment_angular_tolerance = int(comb_config_data["experiment"]["tolerance_deg"])

    interval_id = 0

    schedule = []
    for entry in raw_schedule:
        from_dt = datetime.datetime.fromisoformat(entry["from"]).astimezone(pytz.UTC)
        to_dt = datetime.datetime.fromisoformat(entry["to"]).astimezone(pytz.UTC)

        duration = to_dt - from_dt
        duration_minutes = duration.total_seconds() // 60
        if duration_minutes != interval_duration or not ("angle_deg" in entry):
            print("Skipping entry: {}".format(entry))
            continue

        interval_id += 1

        is_active = entry["rule"] == "vibrate"

        result = dict(
            interval_id=interval_id,
            begin=from_dt,
            end=to_dt,
            is_active=is_active,
            is_control=False,
            control_for=-1,
            angle_deg=entry["angle_deg"],
            angle_tolerance=experiment_angular_tolerance,
        )
        if "current_sound_index" in entry:
            result["sound"] = entry["current_sound_index"]

        schedule.append(result)

    # add controls
    for entry in [e for e in schedule]:
        interval_id += 1

        row = dict(
            interval_id=interval_id,
            begin=entry["begin"] - interval_delta,
            end=entry["end"] - interval_delta,
            is_active=False,
            is_control=True,
            control_for=entry["interval_id"],
            angle_deg=entry["angle_deg"],
            angle_tolerance=entry["angle_tolerance"],
        )
        if "sound" in entry:
            row["sound"] = entry["sound"]

        schedule.append(row)

    schedule = pandas.DataFrame(schedule)
    schedule["date"] = schedule.begin.apply(lambda dt: dt.date())

    return schedule


def load_bridge_output(filepath):
    labels = []
    all_lines = []
    with open(filepath, "r") as f:
        for line in tqdm.auto.tqdm(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(str(e), line)
                continue

            data["log_timestamp"] = pytz.UTC.localize(
                datetime.datetime.fromisoformat(data["log_timestamp"])
            )

            all_lines.append(data)
            labels.append(data["message"])

    import re

    world_angle_regex = re.compile(r"Dance for .+? \(([-]?.*?)°\)[,]? .*")

    def extract_world_angle(line):
        text = line["text"]
        match = world_angle_regex.match(text)
        if match is None:
            return np.nan, np.nan
        azimuth = float(text.split("az. ")[1].split("°")[0])
        return float(match.group(1)), azimuth

    mapped_dances = []
    for line in all_lines:
        if line["message"] != "log":
            continue
        world_angle, azimuth = extract_world_angle(line)
        if pandas.isnull(world_angle):
            continue
        mapped_dances.append((line["log_timestamp"], world_angle, azimuth))
    mapped_dances_df = pandas.DataFrame(
        mapped_dances, columns=["timestamp", "world_angle", "azimuth"]
    )

    from collections import defaultdict

    all_dances_to_waggles = defaultdict(set)
    waggle_ids_to_dance_angle = dict()
    for line in all_lines:
        if line["message"] != "detected dance":
            continue
        waggles = line["waggle_ids"]

        dance_id = waggles[0]
        for w in waggles:
            all_dances_to_waggles[dance_id].add(w)

            if w not in waggle_ids_to_dance_angle:
                waggle_ids_to_dance_angle[w] = line["dance_angle"]
    all_waggles_to_dances = dict()
    for dance_index, (dance_id, waggles) in enumerate(all_dances_to_waggles.items()):
        for w in waggles:
            all_waggles_to_dances[w] = dance_index

    return (
        mapped_dances_df,
        all_dances_to_waggles,
        all_waggles_to_dances,
        waggle_ids_to_dance_angle,
    )


def load_waggle_detection_metdata(
    filepath,
    waggle_id_to_dance_id_map=dict(),
    waggle_id_to_dance_angle_map=dict(),
    filter_class="waggle",
):
    with open(filepath, "r") as f:
        waggle_metadata = json.load(f)

    # The ID is a globally unique identifier for each waggle.
    waggle_id = 0
    if "waggle_id" in waggle_metadata:
        waggle_id = waggle_metadata["waggle_id"]

    # Throw away everything that was not classified as a waggle.
    label = waggle_metadata["predicted_class_label"]
    if filter_class and label != filter_class:
        return None

    dance_id = waggle_id_to_dance_id_map.get(waggle_id, -1)
    dance_angle = np.nan
    if dance_id != -1:
        dance_angle = waggle_id_to_dance_angle_map[waggle_id]

    meta = dict(
        waggle_id=waggle_id,
        dance_id=dance_id,
        waggle_angle=waggle_metadata["waggle_angle"],
        dance_angle=dance_angle,
        # Note that the duration here is not accurate, as it comes from the classification model which does not see the whole waggle.
        # A more accurate duration is likely the count of camera_timestamps (where the initial flickering detection triggered).
        waggle_duration=waggle_metadata["waggle_duration"],
        cam_id=int(waggle_metadata["cam_id"][3]),
        timestamp_begin=datetime.datetime.fromisoformat(
            waggle_metadata["camera_timestamps"][0]
        ),
        timestamp_end=datetime.datetime.fromisoformat(
            waggle_metadata["camera_timestamps"][-1]
        ),
        # The coordinates contain all locations where flickering was detected.
        # Here, I take the last one, but you could also take the median.
        wdd_x=waggle_metadata["x_coordinates"][-1],
        wdd_y=waggle_metadata["y_coordinates"][-1],
        x_median=np.median(waggle_metadata["x_coordinates"]),
        y_median=np.median(waggle_metadata["y_coordinates"]),
        # A rough direction of motion during the waggle - the 'waggle_angle' is more accurate, though.
        dir_x=waggle_metadata["x_coordinates"][-1]
        - waggle_metadata["x_coordinates"][0],
        dir_y=waggle_metadata["y_coordinates"][-1]
        - waggle_metadata["y_coordinates"][0],
        label=label,
        label_confidence=waggle_metadata["predicted_class_confidence"],
    )
    return meta


def parse_waggle_metadata_path(root_path: pathlib.Path, path: pathlib.Path):
    subdir = path.relative_to(root_path)
    parts = str(subdir).replace("\\", "/").split("/")

    idx, minute, hour, day, month, year = map(int, parts[::-1][1:7])

    dt = pytz.UTC.localize(datetime.datetime(year, month, day, hour, minute, 0))
    return dt, idx


def iterate_all_waggle_metadata_files(root_path: str):
    root_path = pathlib.Path(root_path)

    all_json_files = []
    for root, dirs, files in tqdm.auto.tqdm(os.walk(root_path)):
        for f in files:
            if not f.endswith("waggle.json"):
                continue

            full_path = os.path.join(root, f)
            full_path = pathlib.Path(full_path)

            dt, idx = parse_waggle_metadata_path(root_path, full_path)
            yield dict(metadata_path=full_path, metadata_timestamp=dt, dir_index=idx)


def iterate_waggle_metadata_for_schedule(root_path: str, schedule):
    for meta in iterate_all_waggle_metadata_files(root_path):
        timestamp = meta["metadata_timestamp"]
        fit = schedule[(schedule.begin <= timestamp) & (schedule.end > timestamp)]

        if fit.empty:
            continue

        assert fit.shape[0] == 1
        interval_id = fit.interval_id.iloc[0]

        meta["interval_id"] = interval_id

        yield meta


def load_and_parse_all_waggle_metadata(path_generator, **kwargs):
    for meta in path_generator:
        path = meta["metadata_path"]
        del meta["metadata_path"]
        waggle_metadata = load_waggle_detection_metdata(
            path, filter_class="waggle", **kwargs
        )
        if waggle_metadata is None:
            continue

        yield {**waggle_metadata, **meta}


def load_waggle_metadata_dataframe(
    path_generator, latitude, longitude, schedule_df=None, **kwargs
):
    all_waggle_metadata = list(
        load_and_parse_all_waggle_metadata(path_generator, **kwargs)
    )

    waggles_df = pandas.DataFrame(all_waggle_metadata)

    waggles_df["end_x"] = waggles_df.wdd_x + waggles_df.dir_x
    waggles_df["end_y"] = waggles_df.wdd_y + waggles_df.dir_y
    waggles_df["date"] = waggles_df.timestamp_begin.apply(lambda ts: ts.date())
    waggles_df["timestamp_duration"] = (
        waggles_df.timestamp_end - waggles_df.timestamp_begin
    )
    waggles_df["timestamp_duration"] = waggles_df["timestamp_duration"].apply(
        lambda dt: dt.total_seconds()
    )

    world_angles = []
    azimuths = []
    dance_world_angles = []
    for hive_angle, dance_hive_angle, timestamp in tqdm.auto.tqdm(
        waggles_df[["waggle_angle", "dance_angle", "timestamp_begin"]].itertuples(
            index=False
        ),
        total=waggles_df.shape[0],
    ):
        hive_angle = hive_angle - np.pi / 2
        dance_world_angle = np.nan
        if not pandas.isnull(dance_hive_angle):
            dance_hive_angle = dance_hive_angle - np.pi / 2
            (
                dance_world_angle,
                _x,
                _y,
                _azimuth,
            ) = bb_behavior.waggledance.worldmapping.decode_waggle_dance_angle(
                dance_hive_angle,
                timestamp,
                distance=10.0,
                latitude=latitude,
                longitude=longitude,
            )
        dance_world_angles.append(dance_world_angle)

        (
            world_angle,
            x,
            y,
            azimuth,
        ) = bb_behavior.waggledance.worldmapping.decode_waggle_dance_angle(
            hive_angle, timestamp, distance=10.0, latitude=latitude, longitude=longitude
        )
        world_angles.append(world_angle)
        azimuths.append(azimuth)

    waggles_df["azimuth"] = azimuths
    waggles_df["world_angle"] = world_angles
    waggles_df["dance_world_angle"] = dance_world_angles
    waggles_df["world_angle_deg"] = waggles_df.world_angle.values * 180 / np.pi
    waggles_df["dance_world_angle_deg"] = waggles_df.dance_world_angle * 180 / np.pi
    waggles_df["was_clustered_to_dance"] = waggles_df.dance_id != -1

    if schedule_df is not None:
        angles_df = schedule_df[["interval_id", "angle_deg", "angle_tolerance"]]
        angles_df.columns = [
            "interval_id",
            "suppression_angle_deg",
            "suppression_angle_tolerance_deg",
        ]
        waggles_df = pandas.merge(waggles_df, angles_df, how="left", on="interval_id")
        assert waggles_df[pandas.isnull(waggles_df.suppression_angle_deg)].empty

        angular_distance = (
            np.abs(waggles_df.world_angle_deg - waggles_df.suppression_angle_deg) % 360
        )
        angular_distance2 = np.abs(360 - angular_distance) % 360
        angular_distance = np.min(
            np.stack([angular_distance, angular_distance2], axis=1), axis=1
        )
        waggles_df["was_suppression_target"] = (
            angular_distance <= waggles_df.suppression_angle_tolerance_deg.values
        )
    return waggles_df
