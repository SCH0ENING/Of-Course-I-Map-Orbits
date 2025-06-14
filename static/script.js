function toggleCustomFields() {
    const launchSite = document.querySelector('select[name="launch_site"]').value;
    document.getElementById('custom_coords').style.display = launchSite === 'Custom' ? 'block' : 'none';
}

function toggleBoosterFields() {
    const includeBooster = document.getElementById('include_booster').checked;
    document.getElementById('booster_fields').style.display = includeBooster ? 'block' : 'none';
}

function toggleSecondOrbitFields() {
    const includeSecondOrbit = document.getElementById('include_second_orbit').checked;
    document.getElementById('second_orbit_fields').style.display = includeSecondOrbit ? 'block' : 'none';
}

function updateEpochFields() {
    const tle1 = document.querySelector('input[name="tle1"]').value;
    if (tle1) {
        fetch('/parse_tle_epoch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tle1: tle1 })
        })
        .then(response => response.json())
        .then(data => {
            if (data.epoch_date && data.epoch_time) {
                document.getElementById('epoch_date').value = data.epoch_date;
                document.getElementById('epoch_time').value = data.epoch_time;
            }
        });
    }
}

function updateSecondEpochFields() {
    const tle1_second = document.querySelector('input[name="tle1_second"]').value;
    if (tle1_second) {
        fetch('/parse_tle_epoch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tle1: tle1_second })
        })
        .then(response => response.json())
        .then(data => {
            if (data.epoch_date && data.epoch_time) {
                document.getElementById('epoch_date_second').value = data.epoch_date;
                document.getElementById('epoch_time_second').value = data.epoch_time;
            }
        });
    }
}

// Initialize form state on load
window.onload = function() {
    toggleCustomFields();
    toggleBoosterFields();
    toggleSecondOrbitFields();
};