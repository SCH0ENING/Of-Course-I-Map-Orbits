from flask import jsonify
from flask import Flask, request, send_file, render_template
from orbit_logic import launch_sites, parse_datetime, validate_duration, validate_dms_coordinate, generate_orbit, save_plot, save_kml
import os

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    if request.method == 'POST':
        tle1 = request.form.get('tle1')
        tle2 = request.form.get('tle2')
        launch_date = request.form.get('launch_date')
        launch_time = request.form.get('launch_time')
        epoch_date = request.form.get('epoch_date')
        epoch_time = request.form.get('epoch_time')
        launch_site = request.form.get('launch_site')
        duration = request.form.get('duration', '120')
        sample_interval = float(request.form.get('sample_interval', '1'))

        if not all([tle1, tle2, launch_date, launch_time, epoch_date, epoch_time, launch_site]):
            error = "All fields must be filled."
        else:
            if launch_site == "Custom":
                custom_lat = request.form.get('custom_lat')
                custom_lon = request.form.get('custom_lon')
                if not (custom_lat and custom_lon):
                    error = "Custom latitude and longitude must be provided."
                else:
                    lat = validate_dms_coordinate(custom_lat, "Custom Latitude", is_latitude=True)
                    lon = validate_dms_coordinate(custom_lon, "Custom Longitude", is_latitude=False)
                    if lat is None or lon is None:
                        error = "Invalid custom coordinates."
                    else:
                        launch_sites['Custom'] = (lat, lon)
            if not error:
                coords, error = generate_orbit(tle1, tle2, launch_date, launch_time, epoch_date, epoch_time, launch_site, duration, sample_interval)
                if coords:
                    plot_path = save_plot(coords, os.path.join(app.config['UPLOAD_FOLDER'], 'plot.png'))
                    kml_path = save_kml(coords, os.path.join(app.config['UPLOAD_FOLDER'], 'orbit.kml'))
                    return render_template('result.html', plot_url='/plot', kml_url='/kml', error=None)
    
    return render_template('index.html', launch_sites=launch_sites.keys(), sample_intervals=[1, 2, 5, 10], error=error)

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