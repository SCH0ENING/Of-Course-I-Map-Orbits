from flask import jsonify
from flask import Flask, request, send_file, render_template
from orbit_logic import (
    launch_sites, sample_intervals, parse_datetime, validate_duration, validate_dms_coordinate,
    validate_altitude, validate_time_duration, validate_index, parse_tle_epoch,
    great_circle_interpolate, calculate_distance, find_fraction_for_distance, apply_altitude_transition,
    generate_orbit, save_plot, save_kml
)
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    form_data = {}
    if request.method == 'POST':
        # Collect all form inputs
        form_data = {
            'tle1': request.form.get('tle1', ''),
            'tle2': request.form.get('tle2', ''),
            'launch_date': request.form.get('launch_date', ''),
            'launch_time': request.form.get('launch_time', ''),
            'epoch_date': request.form.get('epoch_date', ''),
            'epoch_time': request.form.get('epoch_time', ''),
            'launch_site': request.form.get('launch_site', ''),
            'custom_lat': request.form.get('custom_lat', ''),
            'custom_lon': request.form.get('custom_lon', ''),
            'sample_interval': request.form.get('sample_interval', '1'),
            'duration': request.form.get('duration', '120'),
            'orbit_insertion_altitude': request.form.get('orbit_insertion_altitude', '200'),
            'include_coast_phase': 'include_coast_phase' in request.form,
            'orbit_index_start': request.form.get('orbit_index_start', ''),
            'orbit_index_end': request.form.get('orbit_index_end', ''),
            'coast_time': request.form.get('coast_time', '45'),
            'max_q_alt': request.form.get('max_q_alt', '11'),
            'max_q_downrange': request.form.get('max_q_downrange', '1.5'),
            'stage_sep_alt': request.form.get('stage_sep_alt', '65'),
            'stage_sep_downrange': request.form.get('stage_sep_downrange', '60'),
            'reentry_point': request.form.get('reentry_point', ''),
            'de_orbit_final_altitude': request.form.get('de_orbit_final_altitude', '0'),
            'de_orbit_burn_offset': request.form.get('de_orbit_burn_offset', '40'),
            'include_booster': 'include_booster' in request.form,
            'booster_apogee': request.form.get('booster_apogee', '115'),
            'booster_land_lat': request.form.get('booster_land_lat', ''),
            'booster_land_lon': request.form.get('booster_land_lon', ''),
            'include_second_orbit': 'include_second_orbit' in request.form,
            'tle1_second': request.form.get('tle1_second', ''),
            'tle2_second': request.form.get('tle2_second', ''),
            'epoch_date_second': request.form.get('epoch_date_second', ''),
            'epoch_time_second': request.form.get('epoch_time_second', ''),
            'second_orbit_index_start': request.form.get('second_orbit_index_start', ''),
            'second_orbit_index_end': request.form.get('second_orbit_index_end', '')
        }

        # Basic validation
        required_fields = ['tle1', 'tle2', 'launch_date', 'launch_time', 'epoch_date', 'epoch_time', 'launch_site', 'duration']
        if form_data['include_second_orbit']:
            required_fields.extend(['tle1_second', 'tle2_second', 'epoch_date_second', 'epoch_time_second'])
        if not all(form_data.get(field) for field in required_fields):
            error = "All required fields must be filled."
        elif form_data['launch_site'] == 'Custom' and not (form_data['custom_lat'] and form_data['custom_lon']):
            error = "Custom latitude and longitude must be provided."
        elif form_data['include_booster'] and not (form_data['booster_apogee'] and form_data['booster_land_lat'] and form_data['booster_land_lon']):
            error = "All booster fields must be filled when including booster return."
        else:
            # Validate and process inputs
            if form_data['launch_site'] == 'Custom':
                lat, lat_error = validate_dms_coordinate(form_data['custom_lat'], "Custom Latitude", is_latitude=True)
                lon, lon_error = validate_dms_coordinate(form_data['custom_lon'], "Custom Longitude", is_latitude=False)
                if lat_error or lon_error:
                    error = lat_error or lon_error
                else:
                    launch_sites['Custom'] = (lat, lon)
            
            if not error:
                # Generate orbits
                result = generate_orbit(form_data)
                if result['error']:
                    error = result['error']
                else:
                    # Save plot and KML
                    plot_path = save_plot(
                        result['generated_coords'], result['generated_coords_second'], result['selected_indices'],
                        form_data, os.path.join(app.config['UPLOAD_FOLDER'], 'plot.png')
                    )
                    kml_path = save_kml(
                        result['generated_coords'], result['generated_coords_second'], result['selected_indices'],
                        form_data, os.path.join(app.config['UPLOAD_FOLDER'], 'orbit.kml')
                    )
                    return render_template(
                        'result.html', plot_url='/plot', kml_url='/kml', error=None,
                        coords=result['generated_coords'], coords_second=result['generated_coords_second'],
                        selected_indices=result['selected_indices']
                    )
    
    # Prepopulate epoch fields from TLE if available
    tle1 = form_data.get('tle1', '')
    tle1_second = form_data.get('tle1_second', '')
    epoch_date, epoch_time = parse_tle_epoch(tle1) if tle1 else (form_data.get('epoch_date', ''), form_data.get('epoch_time', ''))
    epoch_date_second, epoch_time_second = parse_tle_epoch(tle1_second) if tle1_second else (form_data.get('epoch_date_second', ''), form_data.get('epoch_time_second', ''))
    form_data['epoch_date'] = epoch_date or form_data.get('epoch_date', '')
    form_data['epoch_time'] = epoch_time or form_data.get('epoch_time', '')
    form_data['epoch_date_second'] = epoch_date_second or form_data.get('epoch_date_second', '')
    form_data['epoch_time_second'] = epoch_time_second or form_data.get('epoch_time_second', '')

    return render_template(
        'index.html', launch_sites=launch_sites.keys(), sample_intervals=sample_intervals,
        error=error, form_data=form_data
    )

@app.route('/plot')
def serve_plot():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'plot.png'))

@app.route('/kml')
def serve_kml():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'orbit.kml'), as_attachment=True, download_name='orbit.kml')

@app.route('/parse_tle_epoch', methods=['POST'])
def parse_tle_epoch_route():
    data = request.get_json()
    tle1 = data.get('tle1', '')
    epoch_date, epoch_time = parse_tle_epoch(tle1)
    return jsonify({'epoch_date': epoch_date or '', 'epoch_time': epoch_time or ''})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)