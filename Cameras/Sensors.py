from .base import Sensor


def load_sensor(sensor):
    assert sensor in sensor_list, "The given Sensor is not implemented"
    return sensor_list[sensor]()


def smartIR640():
    return Sensor(16.4, (0.480 * 16.4, 0.640 * 16.4), (480, 640), 'Infrared',
                  sensor_name='SmartIR640')


def RGBLynred():
    return Sensor(3.45, (0.960 * 3.45, 0.1280 * 3.45), (960, 1280), 'Visible',
                  sensor_name='RGBLynred')


sensor_list = {'SmartIR640': smartIR640, 'RGBLynred': RGBLynred}
