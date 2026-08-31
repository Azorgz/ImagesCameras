import json
import os

from .base import Sensor


def load_json_sensors():
    with open(os.path.join(os.path.dirname(__file__), 'Sensors.json')) as sensors_file:
        sensors = json.load(sensors_file)
    return sensors


def load_sensor(sensor):
    assert sensor in sensor_list, "The given Sensor is not implemented"
    sensor_caracteristics = sensor_list[sensor]
    sensor_caracteristics['pixelSize'] = sensor_caracteristics['pixelSize'] * 1e-6
    sensor_caracteristics['size'] = [s * 1e-3 for s in sensor_caracteristics['size']]
    return Sensor(**sensor_caracteristics)


sensor_list = {k: v for k, v in load_json_sensors().items()}
