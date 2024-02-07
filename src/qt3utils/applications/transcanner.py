import numpy as np
import time
import logging
import nidaqmx
import ctypes

logger = logging.getLogger(__name__)


class Lockin:
    def __init__(self, device_name: str, signal_channel: str, timeout: float) -> None:
        self.device_name = device_name
        self.signal_channel = signal_channel
        self.sample_number = 20
        self.rate = 20.0
        self.timeout = timeout

    def read(self) -> np.ndarray:
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(self.device_name + '/' + self.signal_channel, max_val=10)
            task.timing.cfg_samp_clk_timing(rate=self.rate, samps_per_chan=self.sample_number)
            c = task.read(number_of_samples_per_channel=self.sample_number, timeout=self.timeout)
        return np.array(c)

class Wavemeter:
    def __init__(self,port,dll_library) -> None:
        self.port = port
        self._wavemeter= ctypes.cdll.LoadLibrary(dll_library)
        self._wavemeter.CLGetLambdaReading.restype = ctypes.c_double

    def connect(self) -> int:
        dh = self._wavemeter.CLOpenUSBSerialDevice(self.port)
        if dh == -1:
            raise Exception("fail to connect to the instrument")
        else:
            return dh

    def read(self, device_handle):
        if device_handle != -1:
            b = self._wavemeter.CLGetLambdaReading(device_handle)
        else:
            b=0
        return b

    def disconnect(self, device_handle):
        if device_handle != -1:
            ending = self._wavemeter.CLCloseDevice(device_handle)
            if ending == -1:
                raise Exception("fail to properly close the device")

class TransmissionScanner:

    def __init__(self, wavemeter, signal, wavelength_controller) -> None:
        self.running = False
        self.current_t = 0
        self.tmax = 100
        self.vmin = float(wavelength_controller.minimum_allowed_position)
        self.vmax = float(wavelength_controller.maximum_allowed_position)
        self._step_size = 0.5
        self.raster_line_pause = 0.150  # wait 150ms for the voltage to settle before a line scan

        self.scanned_signal = []
        self.scanned_wavelength = []
        self.scanned_current = []
        self.scanned_voltage = []

        self.wavelength_controller = wavelength_controller
        self.wavemeter = wavemeter
        self.signal = signal


    def stop(self) -> None:
        self.running = False

    def start(self) -> None:
        self.running = True

    def set_to_starting_position(self) -> None:
        self.current_v = self.vmin
        if self.wavelength_controller:
            self.wavelength_controller.go_to_voltage(v=self.vmin)

    def signal_read(self) -> np.ndarray:
        return self.signal.read()

    def current_read(self) -> float:
        return self.wavelength_controller.get_current_voltage()

    def set_scan_range(self, vmin, vmax) -> None:
        if self.wavelength_controller:
            self.wavelength_controller.check_allowed_position(vmin)
            self.wavelength_controller.check_allowed_position(vmax)
        self.vmin = vmin
        self.vmax = vmax

    def get_scan_range(self) -> tuple:
        """
        Returns a tuple of the full scan range
        :return: xmin, xmax, ymin, ymax
        """
        return self.vmin, self.vmax,

    def get_completed_scan_range(self) -> tuple:
        """
        Returns a tuple of the scan range that has been completed
        :return: xmin, xmax, ymin, current_y
        """
        return self.vmin, self.vmax, self.current_v

    def move_v(self) -> None:
        self.current_v += self._step_size
        if self.wavelength_controller and self.current_v <= self.vmax:
            try:
                self.wavelength_controller.go_to_position(v=self.current_v)
            except ValueError as e:
                logger.info(f'out of range\n\n{e}')

    def go_to_v(self, desired_voltage) -> None:
        if self.wavelength_controller and desired_voltage <= self.vmax and desired_voltage >= self.vmin:
            try:
                self.wavelength_controller.go_to_voltage(v=desired_voltage)
            except ValueError as e:
                logger.info(f'out of range\n\n{e}')

    def go_to_v_slow(self, desired_voltage) -> None:
        if self.wavelength_controller and desired_voltage <= self.vmax and desired_voltage >= self.vmin:
            try:
                self.wavelength_controller.go_to_voltage_slowly(v=desired_voltage)
            except ValueError as e:
                logger.info(f'out of range\n\n{e}')

    def scan_v(self) -> None:
        """
        Scans the wavelengths from vmin to vmax in steps of step_size.

        Stores results in self.scanned_raw_counts and self.scanned_count_rate.
        """
        scan_for_axis = self.scan_axis('v', self.vmin, self.vmax, self._step_size)
        self.scanned_signal = np.append(self.scanned_signal,scan_for_axis[0],1)
        self.scanned_wavelength.append(scan_for_axis[1])
        self.scanned_current.append(scan_for_axis[2])
        self.current_t = self.current_t + 1

    def scan_axis(self, axis, min, max, step_size) -> tuple[list,list,list]:
        """
        Moves the stage along the specified axis from min to max in steps of step_size.
        Returns a list of raw counts from the scan in the shape
        [[[counts, clock_samples]], [[counts, clock_samples]], ...] where each [[counts, clock_samples]] is the
        result of a single call to signal_read at each scan position along the axis.
        """
        signal_sample = []
        wavelength = []
        LD_current = []
        self.scanned_voltage= []

        self.wavelength_controller.go_to_voltage(**{axis: min})
        time.sleep(self.raster_line_pause)
        device_handle = self.wavemeter.connect()
        for val in np.arange(min, max, step_size):
            if self.wavelength_controller:
                logger.info(f'go to voltage {axis}: {val:.2f}')
                self.wavelength_controller.go_to_voltage(**{axis: val})
            self.scanned_voltage.append(val)
            _current_read = self.current_read()
            LD_current.append(_current_read)
            _signal_read = self.signal_read()
            signal_sample.append(_signal_read)
            _wavelength_read = self.wavemeter.read(device_handle)
            wavelength.append(_wavelength_read)
            logger.info(f'raw counts, total clock samples: {_signal_read}')
            if self.wavelength_controller:
                logger.info(f'current voltage: {self.wavelength_controller.get_current_voltage()}')

        self.wavemeter.disconnect(device_handle)

        return signal_sample, wavelength, LD_current

    def reset(self) -> None:
        _scan_number = int(round((self.vmax-self.vmin)/self.step_size))
        self.scanned_signal = np.empty((_scan_number,0))
        self.scanned_wavelength = []
        self.scanned_current = []

    def still_scanning(self) -> None:
        if self.running == False:  # this allows external process to stop scan
            return False

        if self.current_t <= self.tmax:  # stops scan when reaches final position
            return True
        else:
            self.running = False
            return False

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, val):
        self._step_size = val
