import argparse
import tkinter as tk
import logging
from threading import Thread

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
import h5py
import nidaqmx
from qt3utils.nidaq.customcontrollers import VControlV
from qt3utils.datagenerators import transcanner

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='NI DAQ (PCIx 6363) / Transmission Scanner',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--daq-name', default='Silicon_DAQ', type=str, metavar='daq_name',
                    help='NI DAQ Device Name')
parser.add_argument('-to', '--rwtimeout', metavar='seconds', default=10, type=int,
                    help='NI DAQ read/write timeout in seconds for single value.')
parser.add_argument('--wavelength-write-channel', metavar='channel', default='ao3', type=str,
                    help='Analog output channel used to control the wavelength of the laser')
parser.add_argument('--wavelength-read-channel', metavar='channel', default='ai0', type=str,
                    help='Analog input channels used to read the instantaneous wavelength')
parser.add_argument('-sig', '--lockin-read-channel', metavar='signal', default='ai21', type=str,
                    help='Analog input channels used to read the analog signal from lock-in')
parser.add_argument('-lmin', '--wavelength-min-position', metavar='voltage', default=0, type=float,
                    help='sets min allowed voltage on wavelength controller.')
parser.add_argument('-lmax', '--wavelength-max-position', metavar='voltage', default=8.5, type=float,
                    help='sets min allowed voltage on wavelength controller.')
parser.add_argument('-q', '--quiet', action='store_true',
                    help='When true,logger level will be set to warning. Otherwise, set to "info".')
parser.add_argument('-dll', '--dll-dict', metavar='dictionary', default='C:\\Users\\Fulab\\Downloads\\00392-2-06-A_CustomerCD621\\00392-2-06-A_Customer CD 621\\Programming Interface\\x64\\CLDevIFace.dll',
                    help='Directory of the windll function for direct communication with the wavemeter.')
parser.add_argument('-LD', '--LD-max-current', metavar='mA', default=200.0, type=float,
                    help='Thorlabs LDC202C max current for laser driver')
parser.add_argument('-tos', '--timeout', metavar='voltage', default=20, type=int,
                    help='timeout value for sampling the lock in amplifier signal')
parser.add_argument('-cmap', metavar='<MPL color>', default='gray',
                    help='Set the MatplotLib colormap scale')

args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig()

if args.quiet is False:
    logger.setLevel(logging.INFO)


class ScanImage:
    def __init__(self, mplcolormap='gray') -> None:
        self.fig, self.axes = plt.subplots(3,1)
        self.cbar = None
        self.cmap = mplcolormap
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.axes[0].set_xlabel('Voltage')
        self.log_data = False
        self.onclick_callback = None

    def update_image(self, model) -> None:

        if self.log_data:
            data = np.log10(model.scanned_signal)
            data[np.isinf(data)] = 0  # protect against +-inf
        else:
            data = model.scanned_signal
        data = np.array(data)#.T.tolist()

        self.artist = self.axes[0].imshow(data, cmap=self.cmap, extent=[model.current_t + model.raster_line_pause,
                                                                   0,
                                                                   model.vmax + model.step_size, model.vmin
                                                                   ])
        if self.cbar is None:
            self.cbar = self.fig.colorbar(self.artist, ax=self.axes[0])
        else:
            self.cbar.update_normal(self.artist)

        if self.log_data is False:
            self.cbar.formatter.set_powerlimits((0, 3))

        self.axes[0].set_xlabel('Pixels')
        self.axes[0].set_ylabel('Voltage (V)')

    def update_plot(self, model) -> None:

        data_x = model.scanned_current
        data_y1 = model.scanned_wavelength
        data_y2 = model.scanned_signal
        sample_number = model.signal.sample_number
        t = model.current_t

        x = np.mean(np.array(data_x), axis=0) / 10 * args.LD_max_current
        y1 = np.mean(np.array(data_y1), axis=0)
        y2 = np.mean(np.array(data_y2), axis=1)

        i = 0
        while i < t:
            x_current = np.array(data_x[i]) / 10 * args.LD_max_current
            y_current = data_y2[:, 0 + i * sample_number:0 + (i + 1) * sample_number - 1]
            y_current_avg = np.mean(y_current, axis=1)
            self.axes[1].plot(x_current, data_y1[i], linewidth=1.5)
            self.axes[2].plot(x_current, y_current_avg, linewidth=1.5)
            i += 1

        self.axes[1].plot(x, y1, 'k', linewidth=1.5)
        self.axes[2].plot(x, y2,  'k', linewidth=1.5)

    def reset(self) -> None:
        self.axes[0].cla()
        self.axes[1].cla()
        self.axes[2].cla()

    def set_onclick_callback(self, f) -> None:
        self.onclick_callback = f

    def onclick(self, event) -> None:
        pass


class SidePanel():
    def __init__(self, root, scan_range) -> None:
        frame = tk.Frame(root)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        row = 0
        tk.Label(frame, text="Scan Settings", font='Helvetica 16').grid(row=row, column=0, pady=10)
        row += 1
        self.startButton = tk.Button(frame, text="Start Scan")
        self.startButton.grid(row=row, column=0)
        self.stopButton = tk.Button(frame, text="Stop Scan")
        self.stopButton.grid(row=row, column=1)
        self.saveScanButton = tk.Button(frame, text="Save Scan")
        self.saveScanButton.grid(row=row, column=2)
        row += 1
        tk.Label(frame, text="Voltage Range (V)").grid(row=row, column=0)
        self.v_min_entry = tk.Entry(frame, width=10)
        self.v_max_entry = tk.Entry(frame, width=10)
        self.v_min_entry.insert(10, scan_range[0])
        self.v_max_entry.insert(10, scan_range[1])
        self.v_min_entry.grid(row=row, column=1)
        self.v_max_entry.grid(row=row, column=2)

        row += 1
        tk.Label(frame, text="Number of scanned voltages").grid(row=row, column=0)
        self.num_pixels = tk.Entry(frame, width=10)
        self.num_pixels.insert(10, 150)
        self.num_pixels.grid(row=row, column=1)

        row += 1
        tk.Label(frame, text="sample number").grid(row=row, column=0)
        self.sample_number_entry = tk.Entry(frame, width=10)
        self.sample_number_entry.insert(10, 20)
        self.sample_number_entry.grid(row=row, column=1)

        tk.Label(frame, text="sample rate").grid(row=row, column=2)
        self.sample_rate_entry = tk.Entry(frame, width=10)
        self.sample_rate_entry.insert(10, 200.0)
        self.sample_rate_entry.grid(row=row, column=3)

        row += 1
        tk.Label(frame, text="Wait Time").grid(row=row, column=0)
        self.wait_time_entry = tk.Entry(frame, width=10)
        self.wait_time_entry.insert(10, 0.1)
        self.wait_time_entry.grid(row=row, column=1)

        row += 1
        tk.Label(frame, text="DAQ Settings", font='Helvetica 16').grid(row=row, column=0, pady=10)

        row += 1
        self.GotoButton = tk.Button(frame, text="Go To Voltage")
        self.GotoButton.grid(row=row, column=0)
        self.GotoSlowButton = tk.Button(frame, text="Slowly Back to Voltage")
        self.GotoSlowButton.grid(row=row, column=1)
        row += 1
        tk.Label(frame, text="Voltage (V)").grid(row=row, column=0)
        self.v_entry = tk.Entry(frame, width=10)
        self.v_entry.insert(10, 0)
        self.v_entry.grid(row=row, column=1)

        row += 1
        self.GetButton = tk.Button(frame, text="Get current Voltage")
        self.GetButton.grid(row=row, column=0)
        self.voltage_show=tk.Label(frame, text='None')
        self.voltage_show.grid(row=row, column=2)

        row += 1
        tk.Label(frame, text="Voltage Limits (V)").grid(row=row, column=0)
        self.v_lmin_entry = tk.Entry(frame, width=10)
        self.v_lmax_entry = tk.Entry(frame, width=10)
        self.v_lmin_entry.insert(10, float(args.wavelength_min_position))
        self.v_lmax_entry.insert(10, float(args.wavelength_max_position))
        self.v_lmin_entry.grid(row=row, column=1)
        self.v_lmax_entry.grid(row=row, column=2)


class MainApplicationView():
    def __init__(self, main_frame, scan_range=[0, 5]) -> None:
        frame = tk.Frame(main_frame)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scan_view = ScanImage(args.cmap)
        self.sidepanel = SidePanel(main_frame, scan_range)

        self.canvas = FigureCanvasTkAgg(self.scan_view.fig, master=frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, frame)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.draw()


class MainTkApplication():

    def __init__(self, transmission_scanner) -> None:
        self.transmission_scanner = transmission_scanner
        self.root = tk.Tk()
        self.view = MainApplicationView(self.root)
        self.view.sidepanel.startButton.bind("<Button>", self.start_scan)
        self.view.sidepanel.saveScanButton.bind("<Button>", self.save_scan)
        self.view.sidepanel.stopButton.bind("<Button>", self.stop_scan)
        self.view.sidepanel.GotoButton.bind("<Button>", self.go_to_voltage)
        self.view.sidepanel.GotoSlowButton.bind("<Button>", self.go_to_voltage_slowly)
        self.view.sidepanel.GetButton.bind("<Button>", self.update_voltage_show)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def go_to_voltage(self, event=None) -> None:
        self.view.sidepanel.startButton['state'] = 'disabled'
        self.view.sidepanel.GotoSlowButton['state'] = 'disabled'
        self.view.sidepanel.GotoButton['state'] = 'disabled'
        self.transmission_scanner.go_to_v(float(self.view.sidepanel.v_entry.get()))
        self.view.sidepanel.startButton['state'] = 'normal'
        self.view.sidepanel.GotoButton['state'] = 'normal'
        self.view.sidepanel.GotoSlowButton['state'] = 'normal'

    def go_to_voltage_slowly(self, event=None) -> None:
        self.view.sidepanel.startButton['state'] = 'disabled'
        self.view.sidepanel.GotoSlowButton['state'] = 'disabled'
        self.view.sidepanel.GotoButton['state'] = 'disabled'
        self.transmission_scanner.go_to_v_slow(float(self.view.sidepanel.v_entry.get()))
        self.view.sidepanel.startButton['state'] = 'normal'
        self.view.sidepanel.GotoSlowButton['state'] = 'normal'
        self.view.sidepanel.GotoButton['state'] = 'normal'

    def update_voltage_show(self, event=None) -> None:
        read=self.transmission_scanner.wavelength_controller.last_write_value
        l=self.view.sidepanel.voltage_show
        l.config(text=read)

    def start_scan(self, event=None) -> None:
        self.view.sidepanel.startButton['state'] = 'disabled'
        self.view.sidepanel.GotoButton['state'] = 'disabled'

        n_sample_size = int(self.view.sidepanel.num_pixels.get())
        vmin = float(self.view.sidepanel.v_min_entry.get())
        vmax = float(self.view.sidepanel.v_max_entry.get())
        step_size = (vmax - vmin) / float(n_sample_size)
        args = [vmin, vmax]
        args.append(step_size)

        settling_time =  float(self.view.sidepanel.wait_time_entry.get())
        self.transmission_scanner.wavelength_controller.settling_time_in_seconds = settling_time
        sample_number = int(self.view.sidepanel.sample_number_entry.get())
        self.transmission_scanner.signal.sample_number = sample_number
        sample_rate = float(self.view.sidepanel.sample_rate_entry.get())
        self.transmission_scanner.signal.rate = sample_rate
        self.scan_thread = Thread(target=self.scan_thread_function, args=args)
        self.scan_thread.start()


    def run(self) -> None:
        self.root.title("QT3transmission: Run transmission scan")
        self.root.deiconify()
        self.root.mainloop()

    def stop_scan(self, event=None) -> None:
        self.transmission_scanner.stop()

    def on_closing(self) -> None:
        try:
            self.stop_scan()
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            logger.debug(e)
            pass

    def scan_thread_function(self, vmin, vmax, step_size) -> None:

        self.transmission_scanner.set_scan_range(vmin, vmax)
        self.transmission_scanner.step_size = step_size

        try:
            self.transmission_scanner.reset()  # clears the data
            self.transmission_scanner.start()  # starts the DAQ
            self.transmission_scanner.set_to_starting_position()  # moves the stage to starting position

            while self.transmission_scanner.still_scanning():
                self.transmission_scanner.scan_v()
                self.view.scan_view.reset()
                self.view.scan_view.update_image(self.transmission_scanner)
                self.view.scan_view.update_plot(self.transmission_scanner)
                self.view.canvas.draw()

            self.transmission_scanner.stop()

        except nidaqmx.errors.DaqError as e:
            logger.info(e)
            logger.info(
                'Check for other applications using resources. If not, you may need to restart the application.')

        self.view.sidepanel.startButton['state'] = 'normal'
        self.view.sidepanel.GotoButton['state'] = 'normal'

    def save_scan(self, event = None):
        myformats = [('Compressed Numpy MultiArray', '*.npz'), ('Numpy Array (count rate only)', '*.npy'), ('HDF5', '*.h5')]
        afile = tk.filedialog.asksaveasfilename(filetypes=myformats, defaultextension='.npz')
        logger.info(afile)
        file_type = afile.split('.')[-1]
        if afile is None or afile == '':
            return # selection was canceled.

        data = dict(
            lockin_signal=self.transmission_scanner.scanned_signal,
            wavelength=np.array(self.transmission_scanner.scanned_wavelength).T,
            LD_current=np.array(self.transmission_scanner.scanned_current).T,
            voltages=self.transmission_scanner.scanned_voltage
        )

        if file_type == 'npy':
            np.save(afile, data['count_rate'])

        if file_type == 'npz':
            np.savez_compressed(afile, **data)

        elif file_type == 'h5':
            h5file = h5py.File(afile, 'w')
            for key, value in data.items():
                h5file.create_dataset(key, data=value)
            h5file.close()


def build_data_scanner():

    wavemeter = transcanner.Wavemeter(port=4,
                                      dll_library=args.dll_dict)

    signal= transcanner.Lockin(device_name=args.daq_name,
                               signal_channel=args.lockin_read_channel,
                               timeout=args.timeout)

    voltage_controller = VControlV(device_name=args.daq_name,
                                  write_channel=args.wavelength_write_channel,
                                  read_channel=args.wavelength_read_channel,
                                  min_position=args.wavelength_min_position,
                                  max_position=args.wavelength_max_position)

    scanner = transcanner.TransmissionScanner(wavemeter, signal, voltage_controller)

    return scanner


def main() -> None:
    tkapp = MainTkApplication(build_data_scanner())
    tkapp.run()


if __name__ == '__main__':
    main()

