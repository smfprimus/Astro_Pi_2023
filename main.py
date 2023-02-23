##########################
#    Astro Pi 2022/23    #
#      Team PRIMUS       #
# Sasha-Mercedes Fischer #
#    Nita Schierwater    #
##########################


####################
# IMPORT LIBRARIES #
####################

# Basic libraries for logging, timekeeping, exceptions, threading
import os
import logging
import sys
import signal
import time
from threading import Thread, Event
from datetime import timedelta, datetime
from PIL import Image, ImageStat
import pandas as pd
import numpy as np

# I/O libraries
from csv import writer

# Astro Pi experiment
from sense_hat import SenseHat
from picamera import PiCamera
from orbit import ISS


########
# INIT #
########

#sys.stdout = open('consoleoutput.log', 'a') #redirect console output to file

print('=== START ===')

starttime = datetime.now() # record time the experiment was started
allocated_time = 10500 # for 3h experiment window set this to 10000 (300 seconds less than 3h)
allocated_space = 2900000000 # size limit of 2.9 Gb (100mb less than 3 Gb) 2,900,000,000

signal.alarm(allocated_time) # set timer and send signal ALARM when time is up

# Check if all necessary folders exist, and create them if they don't
os.makedirs('log', exist_ok=True) # folder for logs
os.makedirs('data', exist_ok=True) # folder for data
os.makedirs('img', exist_ok=True) # folder for images
os.makedirs('lowimg', exist_ok=True) # folder for low resolution images

logfile = 'log/threadlog.log'
maglog = 'log/maglog.log'
camlog = 'log/camlog.log'

with open(logfile, 'a') as f:
    f.write('===============================================\n')
    f.write('Script started at ' + str(starttime)+'\n')
    f.write('===============================================\n')

with open(maglog, 'a') as f:
    f.write('===============================================\n')
    f.write('Script started at ' + str(starttime)+'\n')
    f.write('===============================================\n')

with open(camlog, 'a') as f:
    f.write('===============================================\n')
    f.write('Script started at ' + str(starttime)+'\n')
    f.write('===============================================\n')

cam = PiCamera() # assign camera to variable
mag = SenseHat() # assign sensehat to variable

# define two resolutions:
# highres is used for capturing full resolution images for detailed analysis
# lowres is just used to quickly determine e.g. average brightness 
highres = (4056, 3040)
lowres = (640, 480)

# Measurements will be taken in regular intervals.
# define intervals for camera and magnetometer

caminterval = timedelta(seconds=20) # every 20 seconds, take a picture
camtmoffset = timedelta(seconds=1)
maginterval = timedelta(seconds=5) # every 5 seconds, read mag field
magtmoffset = timedelta(seconds=2)

checksizein = timedelta(seconds=60) # every 60 seconds, check size of data gathered
checksizeof = timedelta(seconds=0)

df_brightness = pd.DataFrame(columns = ['nr', 'name', 'path', 'brightness'])


##################
# MULTITHREADING #
##################

class IntervalTicker():
    """
    Custom class that keeps track of active threads (jobs) and triggers
    them periodically. Ticker is kept running by a continious sleep(1) loop.
    Idea comes from a thread on StackExchange started by HelderDaniel.
    """
    def __init__(self):
        self.jobs = [] # create empty list to keep track of active jobs
        logger = logging.getLogger('intervalticker') # define logging var
        logoutput = logging.FileHandler(logfile) # define handler to write log to file
        logoutput.setLevel(logging.INFO) # set logging level to INFO
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        logoutput.setFormatter(formatter) # assign format to text for logfile
        logger.addHandler(logoutput) # attach handler to logger
        logger.setLevel(logging.DEBUG) # set logging level to DEBUG (second call prevents misformatted log entries that originate outside ch)
        self.logger = logger # attach logger to IntervalTicker

    def _add_job(self, func, interval, *args, **kwargs): # adds new jobs to class
        j = Job(interval, func, *args, **kwargs)
        self.jobs.append(j)

    def _block_main_thread(self): # main ticker loop, that runs continuously
        signal.signal(signal.SIGTERM, service_shutdown) # catch termination signals
        signal.signal(signal.SIGINT, service_shutdown) # catch interupt signals
        signal.signal(signal.SIGALRM, service_timeout) # catch timeout
        signal.signal(signal.SIGUSR1, service_sizelimit) # catch size limit

        while True: # run ticker continuously
            try:
                time.sleep(1) # wait and keep the main IntervalTicker active
            except NukeItFromOrbit: # custom catch all exception
                print('Stopping all threads...')
                self.stop() # stop main thread after exception
                break

    def _start_jobs(self, block): # function for starting new jobs
        for j in self.jobs: # go through al jobs in IntervalTicker
            j.daemon = not block # keep all jobs in separate threads 
            j.start() # start job and assigned function
            self.logger.info('Started new thread {}'.format(j.execute)) #log thread in logfile

    def _stop_jobs(self): # function for stopping active threads
        for j in self.jobs: # go through al jobs in IntervalTicker
            self.logger.info('Stopping thread {}'.format(j.execute)) # log thread shutdown
            j.stop() # shutdown job and close thread

    # this function is used with a decorator to assign functions to threads
    def job(self, interval: timedelta, num_executions, initial_delay: timedelta = None): # job (Thread)
        if initial_delay is None:
            initial_delay = interval # if no delay is set, jobs start after <interval> seconds
        def decorator(f): # wrapper function of the decorator
            self._add_job(f, interval, num_executions=num_executions, initial_delay=initial_delay) # call _add_job and pass function and interval
            return f
        return decorator

    def stop(self): # function to stop threads
        self._stop_jobs()
        self.logger.info('IntervalTicker shut down.')

    def start(self, block=False): # function to start threads
        self.logger.info('Starting IntervalTicker...')
        self._start_jobs(block=block)
        self.logger.info('IntervalTicker now started.')
        if block:
            self._block_main_thread()

class NukeItFromOrbit(Exception):
    """
    Custom exception that triggers clean exit of all threads and main program.
    It's the only way to make sure.
    """
    pass

def service_shutdown(signum, frame): # function to shut down all threads and main
    print("Emergency Shutdown!")
    raise NukeItFromOrbit # raise custom exception

def service_timeout(signum, frame): # function called when timer runs out
    print("Time is up. Shut it all down!")
    raise NukeItFromOrbit

def service_sizelimit(signum, frame): # function called when size limit is reached
    print("Data limit reached.")
    raise NukeItFromOrbit

class Job(Thread): # job class
    """
    Job class that runs a specified function in a separate thread.
    Each thread is run at regular intervals specified by the decorator that created
    that function. The decorator also passes variables, to control an initial 
    delay timer and an optional maximum number of executions. (The latter is
    not used by any job in this experiment. The functionality was left in
    the code to enable easier testing.)
    """
    UNLIMITED_EXECUTION = None # defaut number of executions of jobs (no limit set)
    NUM_EXEC_KEYWORD = "num_executions" # kwarg for number of executions of job
    INITIAL_DELAY_KEYWORD = "initial_delay" # kwarg for delay of job

    def __init__(self, interval: timedelta, execute, *args, **kwargs):
        Thread.__init__(self) # assign job to thread
        self.stopped = Event() # define self.stopped to be an event object
        self.interval = interval # assign interval (set via decorator) to job
        self.execute = execute # execute holds function to be run by the job (this is the actual code)
        self.args = args # additional arguments (tuples)
        if Job.NUM_EXEC_KEYWORD in kwargs.keys():
            self.num_executions = kwargs.pop(Job.NUM_EXEC_KEYWORD)
        else:
            self.num_executions = Job.UNLIMITED_EXECUTION
        if Job.INITIAL_DELAY_KEYWORD in kwargs.keys():
            self.initial_delay = kwargs.pop(Job.INITIAL_DELAY_KEYWORD)
        else:
            self.initial_delay = interval
        self.kwargs = kwargs # additional keyword arguments (dictionaries)
        self.__counter = 0 # count number of executions

    def stop(self): # called to close job
        self.stopped.set() # set event to true
        self.join() # close thread
    
    def _set_check_counter_and_stop(self):
        self.__counter += 1 # count number of times a job has been executed
        if self.num_executions and self.__counter >= self.num_executions: # strop job if executions reach set limit
            self.stopped.set()

    def _adjust_interval(self):
        """
        Check the size of the images folder and compare it with the maximum
        allowed space and the time passed. Calculate the expected size of all
        data gathered by the end of the experiment. If it looks like the file
        size will exceede the allowed space, adjust the interval images are
        taken to keep within the limit.
        """
        pass

    def run(self):
        """
        Method of job cass, that runs the function associated with the job (e.g.
        capturing images). The function is executed after <interval> seconds.
        To compensate for the runtime of the function itself, this method 
        automatically adjusts the interval by subtracting the runtime from it.

        Note: We are aware that the way we integrated the initial wait time (delay)
        this results in the time between the second and third execution of a job
        to become (close to) zero. As this does not negatively impact the experiment
        we chose not to add extra code to prevent that behaviour.
        """
        nperiod = self.interval.total_seconds()
        ntime = time.time()
        if not self.stopped.wait(self.initial_delay.total_seconds()): # wait for initial_delay
            self.execute(*self.args, **self.kwargs)
            self._set_check_counter_and_stop() # check if number of executions has been reached
        while not self.stopped.wait(nperiod): # run job, when it's done waiting
            self.execute(*self.args, **self.kwargs) # run function assigned to job (may pass arguments)
            self._set_check_counter_and_stop() # check if number of executions has been reached
            ntime += self.interval.total_seconds()
            nperiod = ntime - time.time() # adjust next interval by runtime of job capture


################################
# ASTRO PI: PREPARE EXPERIMENT #
################################

def get_mag_data():
    mag_data = [] # create empty list
    t = datetime.now() # record current time
    try:
        mag_temp = mag.get_compass_raw() # read data from magnetometer
    except Exception as error:
            with open(maglog, 'a') as f:
                print('ERROR reading mag  : {}'.format(datetime.now()) + ' Error: ' + str(error))
                f.write('ERROR reading mag  : {}'.format(datetime.now()) + ' Error: ' + str(error) + '\n')
            mag_temp = {'x': 0, 'y': 0, 'z': 0} # assign 0 to measurements to keep program running
    # reformating the collected data and time into columns
    mag_data.append(t)
    mag_data.append(mag_temp['x'])
    mag_data.append(mag_temp['y'])
    mag_data.append(mag_temp['z'])
    return(mag_data)

def get_ISS_position(): # get location of ISS
    ISS_location = ISS.coordinates()
    return(ISS_location)

# read magnetometer n times to 'prime' it
def mag_cal(n):
    """
    We found that it takes quite a few readings from the magnetometer to
    produce accurate data. during the first few measurements values are significantly
    too low. The loop below is therefor used to 'prime' the system with a few measurements
    before logging the values.
    We do not know if this behaviour is to be expected, or if it is a bug in our 
    development environment. In any case, this should not negatively
    impact the experiment.
    """
    i=0
    while i<n:
        get_mag_data()
        i=i+1
    print('Read magnetometer values ' + str(n) + ' times to prime system.')

# calculate size of folder by summing up size of files in that folder
def get_folder_size(folder_path): 
    size = 0
    for path, dirs, files in os.walk(folder_path):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)
    return(size)

# count the number of files in a directory and its subdirectories
def get_nr_of_files(folder_path):
    count = 0
    for path, dirs, files in os.walk(folder_path):
        count += 1
    return(count)

# convert size in byte to Megabyte
def bytes_to_mb(s):
    s = round(s/1024/1024,2)
    return(s)

# calculate average brightness of an image
def calc_brightness( im_file ):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]


############################
# ASTRO PI: RUN EXPERIMENT #
############################

IT = IntervalTicker()

# create thread for magnetometer measurements
@IT.job(interval=maginterval, num_executions=None, initial_delay=magtmoffset)
def read_mag_data():
    # write log entry
    try:
        with open(maglog, 'a') as f:
            print('reading mag field  : {}'.format(datetime.now()))
            f.write('reading mag field  : {}'.format(datetime.now()) + '\n')
    except Exception as error:
        print('Error writing to maglog: ' + str(error))
        pass
    
    # get mag data and write it to csv
    # recorded data is in the form of:
    # 'date', 'mag_x', 'mag_y', 'mag_z', 'lat', 'lon', 'alt'
    # Strength and mag_z/Strength is calculated later in the analysis phase
    try:
        with open('data/mag_data.csv', 'a', buffering = 1, newline = '') as f:
            data=[]
            data_writer = writer(f)
            mdata = get_mag_data()
            ldata = get_ISS_position()
            data = mdata
            data.append(ldata.latitude.degrees)
            data.append(ldata.longitude.degrees)
            data.append(ldata.elevation.km)
            data_writer.writerow(data)
    except Exception as error:
        print('Error recording magnetometer: ' + str(error))

# create thread for image capture and gathering of camera data
@IT.job(interval=caminterval, num_executions=None, initial_delay=camtmoffset)
def take_picture():
    # take time and picture number and write those in camlog
    ta = datetime.now()
    c = take_picture.counter
    try:
        with open(camlog, 'a') as f:
            print('taking picture ' + format(c,'04d') + ': {}'.format(ta))
            f.write('taking picture ' + format(c,'04d') + ': {}'.format(ta) + '\n')
    except Exception as error:
        print('Error writing to camlog: ' + str(error))

    # capture image in high resolution and save in img folder
    try:
        cam.resolution = highres
        cam.capture('img/image%s.jpg' % c)
    except Exception as error:
        print('Error taking high-resolution picture: ' + str(error))

    # capture image in low resolution and save in lowimg folder
    try:
        cam.resolution = lowres
        cam.capture('lowimg/image%s.jpg' % c)
    except Exception as error:
        print('Error taking low-resolution picture: ' + str(error))

    # calculate brightness based on lowres image
    # This is done in preparation for the post-experiment analysis
    file = 'image%s.jpg' % c
    path = 'lowimg' + '/' + file
    try:
        b = calc_brightness(path)
        df_brightness.loc[len(df_brightness)] = [c, file, path, b]
    except Exception as error:
        print('Error calculating image brightness: ' + str(error))

    # advance image counter and measure the time it took to take the images
    take_picture.counter +=1
    tb = datetime.now()
    tt = tb-ta # time taken

    # write image meta data to cam_data.csv
    try:
        with open('data/cam_data.csv', 'a', buffering = 1, newline = '') as f:
            data=[]
            data_writer = writer(f)
            pdata = get_ISS_position()
            data.append(tb)
            data.append(tt)
            data.append(take_picture.counter)
            data.append(pdata.latitude.degrees)
            data.append(pdata.longitude.degrees)
            data.append(pdata.elevation.km)
            data.append(cam.resolution)
            data.append(cam.exposure_speed)
            data.append(cam.exposure_mode)
            data.append(cam.exposure_compensation)
            data.append(cam.iso)
            data.append(cam.analog_gain)
            data.append(cam.meter_mode)
            data.append(cam.digital_gain)
            data.append(cam.awb_gains)
            data.append(cam.awb_mode)
            data.append(cam.brightness)
            data.append(cam.contrast)
            data.append(cam.saturation)
            data.append(cam.sensor_mode)
            data.append(cam.sharpness)
            data.append(cam.shutter_speed)
            data_writer.writerow(data)
    except Exception as error:
        print('Error writing camera data to csv: ' + str(error))

    # show how long it took to take one image (including both resolutions,
    # image brightness calculation and log file writing)
    try:
        print('image capture took : ' + str(tt))
    except Exception as error:
        print('Error: ' + str(error))

take_picture.counter = 0 # set image counter to 0

# periodically check size of gathered data
# checks all files in folders 'img', 'lowimg', 'data', 'log'
@IT.job(interval=checksizein, num_executions=None, initial_delay=checksizeof)
def check_size():
    """
    get size of all files in all subfolders to make sure it stays
    below the set limit of 3 Gb.
    """
    try:
        img_size = get_folder_size('img')
        lowimg_size = get_folder_size('lowimg')
        data_size = get_folder_size('data')
        log_size = get_folder_size('log')
        nr_of_images = get_nr_of_files('img')
    except Exception as error:
        print('Error checking filesize: ' + str(error))
    total_size = img_size + lowimg_size + data_size + log_size
    print('current size of data gathered: ' + str(bytes_to_mb(total_size)) + ' MB')
    print('allowed size of data         : ' + str(bytes_to_mb(allocated_space)) + ' MB')
    if total_size > allocated_space * 0.9:
        print('Size limit warning: Data exceeding 90 percent of allocated space')
    if total_size > allocated_space: # if size limit is reached, raise a signal
        print('Size imit reached - triggering signal')
        signal.raise_signal(signal.SIGUSR1) # raise signal to stop experiment


if __name__ == '__main__': # main function
    try:
        mag_cal(100) # function that reads magnetometer data 100 times
    except Exception as error:
        print('Error priming magnetometer: ' + str(error))
    IT.start(block=True) # start all jobs

"""
CODE BELOW IS RUN AFTER EXPERIMENT FINISHES
"""

endtime = datetime.now()
runtime = (endtime - starttime).total_seconds()
t_error = round(float(runtime) - float(allocated_time),2)

print('\n============================================')
print('\nMain and all threads shut down successfully!')
print('Time taken for experiment: ' + str(round(float(runtime),2)) + ' sec')
print('Time that was allocated  : ' + str(round(float(allocated_time),2)) + ' sec')
print('\n============================================')


#####################
# ASTRO PI: CLEANUP #
#####################

cam.close()