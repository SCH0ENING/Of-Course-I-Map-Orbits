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

# Launch sites dictionary
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

def parse_datetime(date_str: str, time_str: str) -> tuple[datetime, str]:
    try:
        dt_str = f"{date_str} {time_str}"
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S"), None
    except ValueError:
        return None, "Invalid date/time format. Use YYYY-MM-DD and HH:MM:SS."

def validate_duration(duration_str: str) -> tuple[float, str]:
    try:
        duration = float(duration_str)
        if duration <= 0:
            raise ValueError
        return duration, None
    except ValueError:
        return None, "Duration must be a positive number (in minutes)."

def validate_dms_coordinate(coord_str: str, field_name: str, is_latitude: bool = True) -> tuple[float, str]:
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

def generate_orbit(tle1, tle2, launch_date, launch_time, epoch_date, epoch_time, launch_site, duration, sample_interval):
    launch_dt, error = parse_datetime(launch_date, launch_time)
    if error:
        return None, error
    epoch_dt, error = parse_datetime(epoch_date, epoch_time)
    if error:
        return None, error
    duration, error = validate_duration(duration)
    if error:
        return None, error
    if launch_site not in launch_sites:
        return None, "Invalid launch site."
    
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        load = load.FileLoader(data_dir)
        ts = load.timescale()
        satellite = EarthSatellite(tle1, tle2, "Satellite", ts)
        launch_time_ts = ts.utc(
            launch_dt.year, launch_dt.month, launch_dt.day,
            launch_dt.hour, launch_dt.minute, launch_dt.second
        )
        end_time = launch_time_ts + duration / (24 * 60)
        time_step = timedelta(minutes=sample_interval)

        coords = [launch_sites[launch_site] + (0.0,)]
        current_time = launch_time_ts
        while current_time.tt <= end_time.tt:
            geocentric = satellite.at(current_time)
            subpoint = geocentric.subpoint()
            latitude = subpoint.latitude.degrees
            longitude = subpoint.longitude.degrees
            altitude = subpoint.elevation.km
            coords.append((latitude, longitude, altitude))
            current_time += time_step
        return coords, None
    except Exception as e:
        return None, f"Failed to generate orbit: {str(e)}"

def save_plot(coords, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    if not coords:
        ax.text(0.5, 0.5, 'No coordinates generated', ha='center', va='center')
    else:
        lons, lats = zip(*[(c[1], c[0]) for c in coords])
        ax.plot(lons, lats, color='red', label='Orbital Path')
        ax.scatter(lons[0], lats[0], color='green', marker='o', s=100, label='Launch Site')
        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        ax.legend()
        ax.grid(True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path

def save_kml(coords, output_path):
    kml = simplekml.Kml()
    orbital_line = kml.newlinestring(name="Orbital Path")
    orbital_line.altitudemode = simplekml.AltitudeMode.absolute
    orbital_line.style.linestyle.color = simplekml.Color.red
    orbital_line.style.linestyle.width = 4
    kml_coords = [(lon, lat, alt * 1000) for lat, lon, alt in coords]
    orbital_line.coords = kml_coords
    kml.save(output_path)
    return output_path