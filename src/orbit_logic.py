from skyfield.api import load, EarthSatellite
from datetime import datetime, timedelta
import simplekml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import re
from typing import List, Tuple, Optional

# Launch sites and sample intervals
launch_sites = {
    "SLC-4E, Vandenberg SFB, CA, USA": (34.6328, -120.6107),
    "LC-36, Cape Canaveral, FL, USA": (28.4707, -80.5406),
    "LC-39A, Kennedy Space Center, FL, USA": (28.6082, -80.6041),
    "LC-39B, Kennedy Space Center, FL, USA": (28.6271, -80.6209),
    "SLC-40, Cape Canaveral, FL, USA": (28.5621, -80.5772),
    "LC-41, Cape Canaveral, FL, USA": (28.5834, -80.5830),
    "Launch Complex 1, Mahia Peninsula, New Zealand": (-39.2620, 177.8649),
    "Launch Complex 2, Wallops Island, VA, USA": (37.8337, -75.4881),
    "Custom": (0.0, 0.0)
}
sample_intervals = [1, 2, 5, 10]

# Validation and utility functions (unchanged from original)
def parse_datetime(date_str: str, time_str: str) -> tuple[Optional[datetime], Optional[str]]:
    try:
        dt_str = f"{date_str} {time_str}"
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S"), None
    except ValueError:
        return None, "Invalid date/time format. Use YYYY-MM-DD and HH:MM:SS."

def validate_duration(duration_str: str) -> tuple[Optional[float], Optional[str]]:
    try:
        duration = float(duration_str)
        if duration <= 0:
            raise ValueError
        return duration, None
    except ValueError:
        return None, "Duration must be a positive number (in minutes)."

def validate_index(index_str: str, max_index: int, field_name: str = "Index") -> tuple[Optional[int], Optional[str]]:
    try:
        index = int(index_str)
        if index < 0 or index > max_index:
            raise ValueError
        return index, None
    except ValueError:
        return None, f"{field_name} must be a non-negative integer between 0 and {max_index}."

def validate_altitude(alt_str: str, field_name: str) -> tuple[Optional[float], Optional[str]]:
    try:
        alt = float(alt_str)
        if alt < 0:
            raise ValueError
        return alt, None
    except ValueError:
        return None, f"{field_name} must be a non-negative number (in km)."

def validate_dms_coordinate(coord_str: str, field_name: str, is_latitude: bool = True) -> tuple[Optional[float], Optional[str]]:
    pattern = (
        r'^\s*(\d{1,2})°\s*(\d{1,2})\'\s*(\d{1,2}(?:\.\d+)?)?\s*"?\s*([NS])\s*$' if is_latitude
        else r'^\s*(\d{1,3})°\s*(\d{1,2})\'\s*(\d{1,2}(?:\.\d+)?)?\s*"?\s*([EW])\s*$'
    )
    match = re.match(pattern, coord_str.strip(), re.IGNORECASE)
    if not match:
        return None, f"Invalid {field_name} format. Use DD°MM'SS\"N/S or DDD°MM'SS\"E/W."
    degrees, minutes, seconds, direction = match.groups()
    degrees, minutes = int(degrees), int(minutes)
    seconds = float(seconds or 0)
    max_degrees = 90 if is_latitude else 180
    if degrees > max_degrees or minutes > 59 or seconds >= 60:
        return None, f"{field_name} values out of range."
    valid_direction = 'NS' if is_latitude else 'EW'
    if direction.upper() not in valid_direction:
        return None, f"Invalid direction in {field_name}. Use {'N/S' if is_latitude else 'E/W'}."
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction.upper() in ('S', 'W'):
        decimal = -decimal
    return decimal, None

def validate_time_duration(time_str: str, field_name: str = "Time Duration") -> tuple[Optional[float], Optional[str]]:
    try:
        time = float(time_str)
        if time <= 0:
            raise ValueError
        return time, None
    except ValueError:
        return None, f"{field_name} must be a positive number (in minutes)."

def parse_tle_epoch(tle_line1: str) -> tuple[Optional[str], Optional[str]]:
    if len(tle_line1) < 32:
        return None, None
    try:
        epoch_str = tle_line1[18:32].strip()
        year = int(epoch_str[:2])
        year = 2000 + year if year < 57 else 1900 + year
        day_of_year = float(epoch_str[2:])
        epoch_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        date_str = epoch_date.strftime("%Y-%m-%d")
        time_str = epoch_date.strftime("%H:%M:%S")
        return date_str, time_str
    except (ValueError, IndexError):
        return None, None

def great_circle_interpolate(start_lat: float, start_lon: float, end_lat: float, end_lon: float, fraction: float) -> Tuple[float, float]:
    lat1, lon1, lat2, lon2 = map(math.radians, [start_lat, start_lon, end_lat, end_lon])
    d = 2 * math.asin(math.sqrt(
        math.sin((lat2 - lat1) / 2) ** 2 +
        math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    ))
    if d < 1e-10:
        return start_lat, start_lon
    a = math.sin((1 - fraction) * d) / math.sin(d)
    b = math.sin(fraction * d) / math.sin(d)
    x = a * math.cos(lat1) * math.cos(lon1) + b * math.cos(lat2) * math.cos(lon2)
    y = a * math.cos(lat1) * math.sin(lon1) + b * math.cos(lat2) * math.sin(lon2)
    z = a * math.sin(lat1) + b * math.sin(lat2)
    lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def find_fraction_for_distance(start_lat: float, start_lon: float, end_lat: float, end_lon: float, target_distance: float) -> float:
    total_distance = calculate_distance(start_lat, start_lon, end_lat, end_lon)
    if total_distance < 1e-6:
        return 0.0
    if target_distance > total_distance:
        return 1.0
    low, high = 0.0, 1.0
    tolerance = 1e-3
    max_iterations = 100
    for _ in range(max_iterations):
        mid = (low + high) / 2
        lat, lon = great_circle_interpolate(start_lat, start_lon, end_lat, end_lon, mid)
        distance = calculate_distance(start_lat, start_lon, lat, lon)
        if abs(distance - target_distance) < tolerance:
            return mid
        elif distance < target_distance:
            low = mid
        else:
            high = mid
    return (low + high) / 2

def apply_altitude_transition(
    coords: List[Tuple[float, float, float]],
    selected_indices: Tuple[int, ...],
    start_transition_index: int,
    end_transition_index: int,
    start_alt: float,
    target_alt: float,
    is_deorbit: bool = True
) -> List[Tuple[float, float, float]]:
    if not selected_indices or start_transition_index not in selected_indices or end_transition_index not in selected_indices:
        return coords
    start_idx = selected_indices.index(start_transition_index)
    end_idx = selected_indices.index(end_transition_index)
    if start_idx > end_idx:
        return coords
    new_coords = coords.copy()
    num_points = end_idx - start_idx + 1
    for i, idx in enumerate(selected_indices[start_idx:end_idx + 1]):
        t = i / (num_points - 1) if num_points > 1 else 0
        new_alt = start_alt + (target_alt - start_alt) * t
        lat, lon = new_coords[idx][:2]
        new_coords[idx] = (lat, lon, new_alt)
    return new_coords

def generate_orbit(form_data: dict) -> dict:
    generated_coords = []
    generated_coords_second = []
    selected_indices = []
    error = None

    # Validate inputs
    launch_dt, err = parse_datetime(form_data['launch_date'], form_data['launch_time'])
    if err:
        return {'error': err, 'generated_coords': [], 'generated_coords_second': [], 'selected_indices': []}
    epoch_dt, err = parse_datetime(form_data['epoch_date'], form_data['epoch_time'])
    if err:
        return {'error': err, 'generated_coords': [], 'generated_coords_second': [], 'selected_indices': []}
    duration, err = validate_duration(form_data['duration'])
    if err:
        return {'error': err, 'generated_coords': [], 'generated_coords_second': [], 'selected_indices': []}
    sample_interval = float(form_data['sample_interval'])
    launch_site = form_data['launch_site']
    if launch_site not in launch_sites:
        return {'error': 'Invalid launch site.', 'generated_coords': [], 'generated_coords_second': [], 'selected_indices': []}

    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        load = load.FileLoader(data_dir)
        ts = load.timescale()

        # First TLE
        satellite = EarthSatellite(form_data['tle1'], form_data['tle2'], "Satellite", ts)
        launch_time_ts = ts.utc(
            launch_dt.year, launch_dt.month, launch_dt.day,
            launch_dt.hour, launch_dt.minute, launch_dt.second
        )
        end_time = launch_time_ts + duration / (24 * 60)
        time_step = timedelta(minutes=sample_interval)

        launch_lat, launch_lon = launch_sites[launch_site]
        generated_coords.append((launch_lat, launch_lon, 0.0))
        current_time = launch_time_ts
        while current_time.tt <= end_time.tt:
            geocentric = satellite.at(current_time)
            subpoint = geocentric.subpoint()
            latitude = subpoint.latitude.degrees
            longitude = subpoint.longitude.degrees
            altitude = subpoint.elevation.km
            generated_coords.append((latitude, longitude, altitude))
            current_time += time_step

        first_max_index = len(generated_coords) - 1
        start_index, err = validate_index(form_data['orbit_index_start'], first_max_index, "First TLE Start Index") if form_data['orbit_index_start'] else (None, None)
        end_index, err_end = validate_index(form_data['orbit_index_end'], first_max_index, "First TLE End Index") if form_data['orbit_index_end'] else (None, None)
        if start_index is not None and end_index is not None and start_index <= end_index:
            selected_indices.extend(range(start_index, end_index + 1))
        else:
            selected_indices.extend(range(first_max_index + 1))
            if err or err_end:
                error = err or err_end

        # Second TLE
        if form_data['include_second_orbit']:
            epoch_dt_second, err = parse_datetime(form_data['epoch_date_second'], form_data['epoch_time_second'])
            if err:
                return {'error': err, 'generated_coords': generated_coords, 'generated_coords_second': [], 'selected_indices': selected_indices}
            satellite_second = EarthSatellite(form_data['tle1_second'], form_data['tle2_second'], "Second Satellite", ts)
            launch_time_ts_second = ts.utc(
                epoch_dt_second.year, epoch_dt_second.month, epoch_dt_second.day,
                epoch_dt_second.hour, epoch_dt_second.minute, epoch_dt_second.second
            )
            end_time_second = launch_time_ts_second + duration / (24 * 60)

            for idx, current_time in enumerate(np.arange(0, duration * 60, sample_interval * 60), 0):
                time = launch_time_ts_second + timedelta(seconds=current_time)
                geocentric = satellite_second.at(time)
                subpoint = geocentric.subpoint()
                latitude = subpoint.latitude.degrees
                longitude = subpoint.longitude.degrees
                altitude = subpoint.elevation.km
                generated_coords_second.append((latitude, longitude, altitude))

            second_max_index = len(generated_coords_second) - 1
            second_start_index, err = validate_index(form_data['second_orbit_index_start'], second_max_index, "Second TLE Start Index") if form_data['second_orbit_index_start'] else (None, None)
            second_end_index, err_end = validate_index(form_data['second_orbit_index_end'], second_max_index, "Second TLE End Index") if form_data['second_orbit_index_end'] else (None, None)
            if second_start_index is not None and second_end_index is not None and second_start_index <= second_end_index:
                selected_indices.extend(idx + len(generated_coords) for idx in range(second_start_index, second_end_index + 1))
            else:
                selected_indices.extend(idx + len(generated_coords) for idx in range(second_max_index + 1))
                if err or err_end:
                    error = err or err_end

    except Exception as e:
        error = f"Failed to generate orbit: {str(e)}"

    return {
        'generated_coords': generated_coords,
        'generated_coords_second': generated_coords_second,
        'selected_indices': selected_indices,
        'error': error
    }

def save_plot(generated_coords, generated_coords_second, selected_indices, form_data, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    if not generated_coords:
        ax.text(0.5, 0.5, 'No coordinates generated', ha='center', va='center')
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    if not selected_indices:
        ax.text(0.5, 0.5, 'No coordinates selected', ha='center', va='center')
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    launch_site = form_data['launch_site']
    launch_lat, launch_lon = launch_sites[launch_site]
    first_orbital_index = min(selected_indices)
    first_orbital_coord = (
        generated_coords[first_orbital_index] if first_orbital_index < len(generated_coords)
        else generated_coords_second[first_orbital_index - len(generated_coords)]
    )
    first_orbital_lat, first_orbital_lon, first_orbital_alt = first_orbital_coord

    max_q_alt, err = validate_altitude(form_data['max_q_alt'], "Max Q Altitude")
    max_q_downrange, err = validate_altitude(form_data['max_q_downrange'], "Max Q Downrange") if not err else (None, err)
    stage_sep_alt, err = validate_altitude(form_data['stage_sep_alt'], "Stage Separation Altitude") if not err else (None, err)
    stage_sep_downrange, err = validate_altitude(form_data['stage_sep_downrange'], "Stage Separation Downrange") if not err else (None, err)
    orbit_insertion_alt, err = validate_altitude(form_data['orbit_insertion_altitude'], "Orbit Insertion Altitude") if not err else (None, err)
    if err:
        ax.text(0.5, 0.5, err, ha='center', va='center')
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    # Ascent trajectory
    total_distance = calculate_distance(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon)
    total_distance = max(total_distance, 0.001)
    max_q_fraction = find_fraction_for_distance(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon, max_q_downrange)
    stage_sep_fraction = find_fraction_for_distance(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon, stage_sep_downrange)

    key_points = [
        (0.0, launch_lat, launch_lon, 0.0),
        (max_q_fraction, *great_circle_interpolate(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon, max_q_fraction), max_q_alt),
        (stage_sep_fraction, *great_circle_interpolate(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon, stage_sep_fraction), stage_sep_alt),
        (1.0, first_orbital_lat, first_orbital_lon, orbit_insertion_alt)
    ]

    ascent_lons, ascent_lats = [], []
    num_segments = len(key_points) - 1
    total_points = 100
    points_per_segment = [total_points // num_segments] * num_segments
    remainder = total_points % num_segments
    for i in range(remainder):
        points_per_segment[i] += 1

    for i in range(num_segments):
        start_fraction, start_lat, start_lon, start_alt = key_points[i]
        end_fraction, end_lat, end_lon, end_alt = key_points[i + 1]
        num_points = points_per_segment[i]
        for j in range(num_points + 1):
            t = j / num_points if num_points > 1 else 0
            lat, lon = great_circle_interpolate(start_lat, start_lon, end_lat, end_lon, t)
            ascent_lats.append(lat)
            ascent_lons.append(lon)

    ax.plot(ascent_lons, ascent_lats, color='green', label='Ascent Trajectory', linewidth=4.0)
    for _, lat, lon, _ in key_points:
        ax.scatter([lon], [lat], color='green', marker='o', s=100)

    # Orbital path
    orbital_lats, orbital_lons = [], []
    start_orbital_idx = first_orbital_index
    max_index = len(generated_coords) - 1
    if form_data['include_second_orbit']:
        max_index += len(generated_coords_second)

    if form_data['include_coast_phase']:
        coast_index, err = validate_index(form_data['orbit_index_start'], max_index, "Coast Start Index")
        coast_time, err = validate_time_duration(form_data['coast_time'], "Coast Time") if not err else (None, err)
        if err:
            ax.text(0.5, 0.5, err, ha='center', va='center')
            fig.savefig(output_path)
            plt.close(fig)
            return output_path

        if coast_index in selected_indices:
            num_points = int(coast_time / sample_interval) + 1
            target_index = min(coast_index + num_points - 1, max_index)
            start_orbital_idx = target_index if target_index in selected_indices else first_orbital_index

            coords = generated_coords if coast_index < len(generated_coords) else generated_coords_second
            offset = 0 if coast_index < len(generated_coords) else len(generated_coords)
            target_alt = coords[target_index - offset][2] if target_index < len(generated_coords) else generated_coords_second[target_index - len(generated_coords)][2]
            coast_coords = apply_altitude_transition(
                coords,
                tuple(idx - offset for idx in selected_indices if idx >= offset),
                coast_index - offset,
                target_index - offset,
                orbit_insertion_alt,
                target_alt,
                is_deorbit=False
            )
            coast_lats, coast_lons = [], []
            for idx in range(coast_index, target_index + 1):
                if idx < len(generated_coords):
                    lat, lon, _ = coast_coords[idx]
                else:
                    lat, lon, _ = coast_coords[idx - len(generated_coords)]
                coast_lats.append(lat)
                coast_lons.append(lon)
            ax.plot(coast_lons, coast_lats, color='purple', label='Coast Phase', linewidth=4.0)
            ax.scatter([coast_lons[-1]], [coast_lats[-1]], color='purple', marker='^', s=100, label='Second Burn')

    # De-orbit transition
    plot_coords = generated_coords.copy()
    if form_data['include_second_orbit']:
        plot_coords.extend(generated_coords_second)
    
    deorbit_burn_index = None
    reentry_index, err = validate_index(form_data['reentry_point'], max_index, "Re-Entry Point") if form_data['reentry_point'] else (None, None)
    deorbit_offset, err = validate_time_duration(form_data['de_orbit_burn_offset'], "De-Orbit Burn Offset") if form_data['de_orbit_burn_offset'] and not err else (None, err)
    if reentry_index is not None and deorbit_offset is not None and reentry_index in selected_indices:
        offset_points = int(deorbit_offset / sample_interval)
        deorbit_burn_index = max(min(reentry_index - offset_points, max_index), 0)
        if deorbit_burn_index in selected_indices:
            target_alt, err = validate_altitude(form_data['de_orbit_final_altitude'], "De-Orbit Final Altitude")
            if err:
                ax.text(0.5, 0.5, err, ha='center', va='center')
                fig.savefig(output_path)
                plt.close(fig)
                return output_path
            start_alt = plot_coords[deorbit_burn_index][2]
            plot_coords = apply_altitude_transition(
                plot_coords,
                tuple(selected_indices),
                deorbit_burn_index,
                reentry_index,
                start_alt,
                target_alt,
                is_deorbit=True
            )

    for idx in sorted(selected_indices):
        if idx < start_orbital_idx:
            continue
        if idx < len(plot_coords):
            lat, lon, _ = plot_coords[idx]
            orbital_lats.append(lat)
            orbital_lons.append(lon)
    if orbital_lats:
        ax.plot(orbital_lons, orbital_lats, color='red', label='Orbital Path', linewidth=4.0)

    # Booster return
    if form_data['include_booster']:
        booster_apogee, err = validate_altitude(form_data['booster_apogee'], "Booster Apogee")
        booster_land_lat, err = validate_dms_coordinate(form_data['booster_land_lat'], "Landing Latitude", is_latitude=True) if not err else (None, err)
        booster_land_lon, err = validate_dms_coordinate(form_data['booster_land_lon'], "Landing Longitude", is_latitude=False) if not err else (None, err)
        if err:
            ax.text(0.5, 0.5, err, ha='center', va='center')
            fig.savefig(output_path)
            plt.close(fig)
            return output_path
        stage_sep_lat, stage_sep_lon = great_circle_interpolate(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon, stage_sep_fraction)
        apogee_lat, apogee_lon = great_circle_interpolate(stage_sep_lat, stage_sep_lon, booster_land_lat, booster_land_lon, 0.5)
        booster_lats = [stage_sep_lat, apogee_lat, booster_land_lat]
        booster_lons = [stage_sep_lon, apogee_lon, booster_land_lon]
        ax.plot(booster_lons, booster_lats, color='blue', label='Booster Return', linewidth=4.0)
        ax.scatter([booster_land_lon], [booster_land_lat], color='blue', marker='x', s=100, label='Booster Landing')

    # De-orbit burn marker
    if deorbit_burn_index is not None and deorbit_burn_index in selected_indices and deorbit_burn_index < len(plot_coords):
        lat, lon, _ = plot_coords[deorbit_burn_index]
        if lat is not None and lon is not None:
            ax.scatter([lon], [lat], color='orange', marker='*', s=100, label='De-Orbit Burn')

    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.legend()
    ax.grid(True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path

def save_kml(generated_coords, generated_coords_second, selected_indices, form_data, output_path):
    if not selected_indices:
        return output_path
    if not generated_coords:
        return output_path

    max_index = len(generated_coords) - 1
    if form_data['include_second_orbit']:
        max_index += len(generated_coords_second)

    # Validate inputs
    reentry_index = None
    target_alt = 0.0
    if form_data['reentry_point'] and form_data['de_orbit_final_altitude']:
        reentry_index, err = validate_index(form_data['reentry_point'], max_index, "Re-Entry Point")
        if err:
            return output_path
        if reentry_index not in selected_indices:
            return output_path
        target_alt, err = validate_altitude(form_data['de_orbit_final_altitude'], "De-Orbit Final Altitude")
        if err:
            return output_path

    coast_index = None
    coast_time = None
    orbit_insertion_alt = None
    if form_data['include_coast_phase'] and all([form_data['orbit_index_start'], form_data['coast_time'], form_data['orbit_insertion_altitude']]):
        coast_index, err = validate_index(form_data['orbit_index_start'], max_index, "Coast Start Index")
        coast_time, err = validate_time_duration(form_data['coast_time'], "Coast Time") if not err else (None, err)
        orbit_insertion_alt, err = validate_altitude(form_data['orbit_insertion_altitude'], "Orbit Insertion Altitude") if not err else (None, err)
        if err:
            return output_path
        if coast_index not in selected_indices:
            return output_path

    deorbit_burn_index = None
    if form_data['de_orbit_burn_offset'] and reentry_index is not None:
        deorbit_offset, err = validate_time_duration(form_data['de_orbit_burn_offset'], "De-Orbit Burn Offset")
        if err:
            return output_path
        sample_interval = float(form_data['sample_interval'])
        offset_points = int(deorbit_offset / sample_interval)
        deorbit_burn_index = max(min(reentry_index - offset_points, max_index), 0)
        if deorbit_burn_index not in selected_indices:
            return output_path

    max_q_alt, err = validate_altitude(form_data['max_q_alt'], "Max Q Altitude")
    max_q_downrange, err = validate_altitude(form_data['max_q_downrange'], "Max Q Downrange") if not err else (None, err)
    stage_sep_alt, err = validate_altitude(form_data['stage_sep_alt'], "Stage Separation Altitude") if not err else (None, err)
    stage_sep_downrange, err = validate_altitude(form_data['stage_sep_downrange'], "Stage Separation Downrange") if not err else (None, err)
    if err:
        return output_path
    if max_q_downrange >= stage_sep_downrange:
        return output_path

    booster_apogee = None
    booster_land_lat = None
    booster_land_lon = None
    if form_data['include_booster']:
        booster_apogee, err = validate_altitude(form_data['booster_apogee'], "Booster Apogee")
        booster_land_lat, err = validate_dms_coordinate(form_data['booster_land_lat'], "Landing Latitude", is_latitude=True) if not err else (None, err)
        booster_land_lon, err = validate_dms_coordinate(form_data['booster_land_lon'], "Landing Longitude", is_latitude=False) if not err else (None, err)
        if err:
            return output_path

    launch_site = form_data['launch_site']
    launch_lat, launch_lon = launch_sites[launch_site]
    coords = generated_coords.copy()
    if form_data['include_second_orbit']:
        coords.extend(generated_coords_second)

    # Apply transitions
    if reentry_index is not None and deorbit_burn_index is not None:
        start_alt = coords[deorbit_burn_index][2]
        coords = apply_altitude_transition(
            coords,
            tuple(selected_indices),
            deorbit_burn_index,
            reentry_index,
            start_alt,
            target_alt,
            is_deorbit=True
        )

    if form_data['include_coast_phase'] and coast_index is not None and coast_time is not None and orbit_insertion_alt is not None:
        sample_interval = float(form_data['sample_interval'])
        num_points = int(coast_time / sample_interval) + 1
        target_index = min(coast_index + num_points - 1, max_index)
        if target_index in selected_indices:
            target_coords = coords
            target_alt = target_coords[target_index][2]
            coords = apply_altitude_transition(
                target_coords,
                tuple(selected_indices),
                coast_index,
                target_index,
                orbit_insertion_alt,
                target_alt,
                is_deorbit=False
            )

    first_orbital_index = min(selected_indices)
    first_orbital_coord = coords[first_orbital_index]
    first_orbital_lat, first_orbital_lon, first_orbital_alt = first_orbital_coord

    total_distance = calculate_distance(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon)
    total_distance = max(total_distance, 0.001)
    max_q_fraction = find_fraction_for_distance(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon, max_q_downrange)
    stage_sep_fraction = find_fraction_for_distance(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon, stage_sep_downrange)
    if max_q_fraction >= stage_sep_fraction:
        return output_path

    kml = simplekml.Kml()
    ascent_line = kml.newlinestring(name="Ascent Trajectory")
    ascent_line.altitudemode = simplekml.AltitudeMode.absolute
    ascent_line.style.linestyle.color = simplekml.Color.darkgreen
    ascent_line.style.linestyle.width = 4

    key_points = [
        ("Launch Site", 0.0, launch_lat, launch_lon, 0.0),
        ("Max Q", max_q_fraction, *great_circle_interpolate(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon, max_q_fraction), max_q_alt),
        ("Stage Separation", stage_sep_fraction, *great_circle_interpolate(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon, stage_sep_fraction), stage_sep_alt),
        ("Orbit Insertion", 1.0, first_orbital_lat, first_orbital_lon, orbit_insertion_alt)
    ]

    ascent_coords = []
    num_segments = len(key_points) - 1
    total_points = 100
    points_per_segment = [total_points // num_segments] * num_segments
    remainder = total_points % num_segments
    for i in range(remainder):
        points_per_segment[i] += 1

    for i in range(num_segments):
        _, start_fraction, start_lat, start_lon, start_alt = key_points[i]
        _, end_fraction, end_lat, end_lon, end_alt = key_points[i + 1]
        num_points = points_per_segment[i]
        for j in range(num_points + 1):
            t = j / num_points if num_points > 1 else 0
            lat, lon = great_circle_interpolate(start_lat, start_lon, end_lat, end_lon, t)
            alt = start_alt + t * (end_alt - start_alt)
            ascent_coords.append((lon, lat, alt * 1000))
    ascent_line.coords = ascent_coords

    for name, _, lat, lon, alt in key_points:
        pm = kml.newpoint(name=name)
        pm.coords = [(lon, lat, alt * 1000)]
        pm.altitudemode = simplekml.AltitudeMode.absolute

    start_orbital_idx = first_orbital_index
    if form_data['include_coast_phase'] and coast_index is not None and coast_time is not None and orbit_insertion_alt is not None:
        sample_interval = float(form_data['sample_interval'])
        num_points = int(coast_time / sample_interval) + 1
        target_index = min(coast_index + num_points - 1, max_index)
        if target_index in selected_indices:
            start_orbital_idx = target_index
            coast_line = kml.newlinestring(name="Coast Phase")
            coast_line.altitudemode = simplekml.AltitudeMode.absolute
            coast_line.style.linestyle.color = simplekml.Color.white
            coast_line.style.linestyle.width = 4
            coast_coords = []
            for idx in range(coast_index, target_index + 1):
                lat, lon, alt = coords[idx]
                coast_coords.append((lon, lat, alt * 1000))
            coast_line.coords = coast_coords
            pm = kml.newpoint(name="Second Burn")
            pm.coords = [(coast_coords[-1][0], coast_coords[-1][1], coast_coords[-1][2])]
            pm.altitudemode = simplekml.AltitudeMode.absolute

    orbital_line = kml.newlinestring(name="Orbital Path")
    orbital_line.altitudemode = simplekml.AltitudeMode.absolute
    orbital_line.style.linestyle.color = simplekml.Color.red
    orbital_line.style.linestyle.width = 4
    orbital_coords = []
    last_coord = None
    for idx in sorted(selected_indices):
        if idx < start_orbital_idx:
            continue
        lat, lon, alt = coords[idx]
        orbital_coords.append((lon, lat, alt * 1000))
        current_coord = (lat, lon, alt)
        if last_coord and form_data['include_second_orbit'] and idx == len(generated_coords):
            distance = calculate_distance(last_coord[0], last_coord[1], current_coord[0], current_coord[1])
            if distance > 0.1:
                connect_line = kml.newlinestring(name="Connection Path")
                connect_line.altitudemode = simplekml.AltitudeMode.absolute
                connect_line.style.linestyle.color = simplekml.Color.yellow
                connect_line.style.linestyle.width = 4
                connect_line.coords = [
                    (last_coord[1], last_coord[0], last_coord[2] * 1000),
                    (current_coord[1], current_coord[0], current_coord[2] * 1000)
                ]
        last_coord = current_coord
    if orbital_coords:
        orbital_line.coords = orbital_coords

    if deorbit_burn_index is not None and deorbit_burn_index in selected_indices and deorbit_burn_index < len(coords):
        lat, lon, alt = coords[deorbit_burn_index]
        if lat is not None and lon is not None:
            pm = kml.newpoint(name="De-Orbit Burn")
            pm.coords = [(lon, lat, alt * 1000)]
            pm.altitudemode = simplekml.AltitudeMode.absolute
            pm.style.iconstyle.color = simplekml.Color.orange
            pm.style.iconstyle.scale = 1.0
            pm.style.labelstyle.scale = 1.0

    if form_data['include_booster'] and all([booster_apogee, booster_land_lat, booster_land_lon]):
        stage_sep_lat, stage_sep_lon = great_circle_interpolate(launch_lat, launch_lon, first_orbital_lat, first_orbital_lon, stage_sep_fraction)
        booster_line = kml.newlinestring(name="Booster Return Trajectory")
        booster_line.altitudemode = simplekml.AltitudeMode.absolute
        booster_line.style.linestyle.color = simplekml.Color.blue
        booster_line.style.linestyle.width = 4
        apogee_lat, apogee_lon = great_circle_interpolate(stage_sep_lat, stage_sep_lon, booster_land_lat, booster_land_lon, 0.5)
        booster_coords = [
            (stage_sep_lon, stage_sep_lat, stage_sep_alt * 1000),
            (apogee_lon, apogee_lat, booster_apogee * 1000),
            (booster_land_lon, booster_land_lat, 0)
        ]
        booster_line.coords = booster_coords
        pm = kml.newpoint(name="Booster Apogee")
        pm.coords = [(apogee_lon, apogee_lat, booster_apogee * 1000)]
        pm.altitudemode = simplekml.AltitudeMode.absolute
        pm = kml.newpoint(name="Booster Landing")
        pm.coords = [(booster_land_lon, booster_land_lat, 0)]
        pm.altitudemode = simplekml.AltitudeMode.absolute

    kml.save(output_path)
    return output_path