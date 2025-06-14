# Map Orbits

A web-based application to generate orbital trajectories and KML files from TLE data, with full support for ascent, coast phase, booster return, de-orbit, and second TLE.

## Local Setup
1. Install Python 3.10 and Git.
2. Clone the repository.
3. Install dependencies: `pip install -r requirements.txt`
4. Ensure `src/data/de421.bsp` exists.
5. Run: `python src/app.py`
6. Access at `http://localhost:5000`

## Deployment
Deploy on Render by connecting this repository and setting the start command to `gunicorn --workers 4 --bind 0.0.0.0:$PORT src.app:app`.