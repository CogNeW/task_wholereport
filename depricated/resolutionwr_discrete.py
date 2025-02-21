"""An estimation whole report experiment.

Author - Colin Quirk (cquirk@uchicago.edu)

Repo: https://github.com/colinquirk/PsychopyResolutionWR

This is a working memory paradigm adapted from Adam, Vogel, Awh (2017) with minor differences.
This code can either be used directly or it can be inherited and extended.
If this file is run directly the defaults at the top of the page will be
used. To make simple changes, you can adjust any of these files.
For more in depth changes you will need to overwrite the methods yourself.

Note: this code relies on my templateexperiments module. You can get
it from https://github.com/colinquirk/templateexperiments and either put it in the
same folder as this code or give the path to psychopy in the preferences.

Classes:
ResolutionWR -- The class that runs the experiment.
    See 'print ResolutionWR.__doc__' for simple class docs or help(ResolutionWR) for everything.
"""


import copy
import errno
import json
import math
import os
import random
import sys

import numpy as np

import psychopy

import template as template

from psychopy import visual
from psychopy_visionscience.radial import RadialStim
from psychopy.tools.monitorunittools import pix2deg

# Things you probably want to change
set_sizes = [6]
trials_per_set_size = 30  # per block
number_of_blocks = 2

iti_time = 1
cue_time = 1.1
sample_time = .25
delay_time = 1.3
monitor_distance = 120

colors = {
    "Cyan": (-1, 1, 1),
    "Red": (1, -1, -1),
    "Green": (-1, 1, -1),
    "Orange": (1, 0, -1),
    "Black": (-1, -1, -1),
    "White": (1, 1, 1),
    "Yellow": (1, 1, -1),
    "Blue": (-1, -1, 1),
    "Magenta": (1, -1, 1),
}

experiment_name = 'ResolutionWR_Discrete'

data_directory = os.path.join(
    os.path.expanduser('~'), 'Desktop', experiment_name, 'Data')

instruct_text = [
    ('Welcome to the experiment. Press space to begin.'),
    ('In this experiment you will be remembering colors.\n\n'
     'Each trial will start with a blank screen.\n'
     'Then, a number of circles with different colors will appear.\n'
     'Remember as many colors as you can.\n\n'
     'After a short delay, color wheels will appear.\n\n'
     'Match the color wheel to the color that appeared in that position.\n'
     'Click the mouse button until the wheel disappears.\n'
     'If you are not sure, just take your best guess.\n\n'
     'You will get breaks in between blocks.\n\n'
     'Press space to start.'),
]

# Things you probably don't need to change, but can if you want to
stim_size = 1.2/3  # visual degrees

data_fields = [
    'Subject',
    'Session',
    'Block',
    'Trial',
    'LocationNumber',
    'ClickNumber',
    'TS_ITI',
    'TS_Cue',
    'TS_Stim',
    'TS_Delay',
    'TS_Resp',
    'Timestamp',
    'SetSize',
    'LocationX',
    'LocationY',
    'ColorIndex',
    'TrueColor',
    'RespColor',
    'InCuedSet',
    'TrueColorName',
    'RespColorName',
    'Accuracy',
    'RT',
]

gender_options = [
    'Male',
    'Female',
    'Other/Choose Not To Respond',
]

hispanic_options = [
    'Yes, Hispanic or Latino/a',
    'No, not Hispanic or Latino/a',
    'Choose Not To Respond',
]

race_options = [
    'American Indian or Alaskan Native',
    'Asian',
    'Pacific Islander',
    'Black or African American',
    'White / Caucasian',
    'More Than One Race',
    'Choose Not To Respond',
]

# Add additional questions here
questionaire_dict = {
    'Session': 1,
    'Age': 0,
    'Gender': gender_options,
    'Hispanic:': hispanic_options,
    'Race': race_options,
}

# This is the logic that runs the experiment
# Change anything below this comment at your own risk
psychopy.logging.console.setLevel(psychopy.logging.CRITICAL)  # Avoid error output

from psychopy import visual, core, event
import socket

#server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#host = '127.0.0.1'  # The server's hostname or IP address
#port = 8000        # The port used by the server

#server_socket.bind((host, port))

# Listen for incoming connections (only one client expected)
#server_socket.listen(1)
#server_socket.settimeout(100)
#print("Waiting for a connection")
#client_socket, addr = server_socket.accept()  # This line will block until a client connects
#print("Connected by", addr)

#client_socket.settimeout(500)

class ResolutionWR(template.BaseExperiment):
    """
    The class that runs the whole report estimation experiment.

    Parameters:
    data_directory -- Where the data should be saved.
    delay_time -- The number of seconds between the stimuli display and test.
    distance_from_fixation -- A number describing how far from fixation stimuli will
        appear in visual degrees.
    instruct_text -- The text to be displayed to the participant at the beginning of the experiment.
    iti_time -- The number of seconds in between a response and the next trial.
    min_color_dist -- The minimum number of degrees in color space between display items.
    number_of_blocks -- The number of blocks in the experiment.
    questionaire_dict -- Questions to be included in the dialog.
    sample_time -- The number of seconds the stimuli are on the screen for.
    set_sizes -- A list of all the set sizes.
        An equal number of trials will be shown for each set size.
    stim_size -- The size of the stimuli in visual degrees.
    trials_per_set_size -- The number of trials per set size per block.

    Methods:
    calculate_locations -- Calculates locations for the upcoming trial with random jitter.
    calculate_error -- Calculates error in a response compared to the true color value.
    chdir -- Changes the directory to where the data will be saved.
    display_blank -- Displays a blank screen.
    display_break -- Displays a screen during the break between blocks.
    display_stimuli -- Displays the stimuli.
    get_response -- Manages getting responses for all color wheels.
    make_block -- Creates a list of trials to be run.
    make_trial -- Creates a single trial dictionary.
    run -- Runs the entire experiment including optional hooks.
    run_trial -- Runs a single trial.
    send_data -- Updates the experiment data with the information from the last trial.
    """
    def __init__(self, set_sizes=set_sizes, trials_per_set_size=trials_per_set_size,
                 number_of_blocks=number_of_blocks, stim_size=stim_size,
                 iti_time=iti_time, sample_time=sample_time, delay_time=delay_time,
                 data_directory=data_directory, questionaire_dict=questionaire_dict,
                 instruct_text=instruct_text, **kwargs):

        self.set_sizes = set_sizes
        self.trials_per_set_size = trials_per_set_size
        self.number_of_blocks = number_of_blocks

        self.stim_size = stim_size

        self.questionaire_dict = questionaire_dict
        self.data_directory = data_directory
        self.instruct_text = instruct_text

        self.iti_time = iti_time
        self.cue_time = cue_time
        self.sample_time = sample_time
        self.delay_time = delay_time

        self.mouse = None

        super().__init__(**kwargs)

    def save_experiment_info(self, filename=None):
        """Writes the info from the dialog box to a json file.

        This method overwrites the base method in order to include the session number in the filename.

        Parameters:
            filename -- a string defining the filename with no extension
        """

        ext = '.json'

        if filename is None:
            filename = (self.experiment_name + '_' +
                        self.experiment_info['Subject Number'].zfill(3) + '_' +
                        str(self.experiment_info['Session']).zfill(3) +
                        '_info')
        elif filename[-5:] == ext:
            filename = filename[:-5]

        if os.path.isfile(filename + ext):
            if self.overwrite_ok is None:
                self.overwrite_ok = self._confirm_overwrite()
            if not self.overwrite_ok:
                # If the file exists make a new filename
                i = 1
                new_filename = filename + '(' + str(i) + ')'
                while os.path.isfile(new_filename + ext):
                    i += 1
                    new_filename = filename + '(' + str(i) + ')'
                filename = new_filename

        filename = filename + ext

        with open(filename, 'w') as info_file:
            info_file.write(json.dumps(self.experiment_info))

    def open_csv_data_file(self, data_filename=None):
        """Opens the csv file and writes the header.

        This method overwrites the base method in order to include the session number in the filename.

        Parameters:
            data_filename -- name of the csv file with no extension
                (defaults to experimentname_subjectnumber).
        """

        if data_filename is None:
            data_filename = (self.experiment_name + '_' +
                             self.experiment_info['Subject Number'].zfill(3) + '_' +
                             str(self.experiment_info['Session']).zfill(3))
        elif data_filename[-4:] == '.csv':
            data_filename = data_filename[:-4]

        if os.path.isfile(data_filename + '.csv'):
            if self.overwrite_ok is None:
                self.overwrite_ok = self._confirm_overwrite()
            if not self.overwrite_ok:
                # If the file exists and we can't overwrite make a new filename
                i = 1
                new_filename = data_filename + '(' + str(i) + ')'
                while os.path.isfile(new_filename + '.csv'):
                    i += 1
                    new_filename = data_filename + '(' + str(i) + ')'
                data_filename = new_filename

        self.experiment_data_filename = data_filename + '.csv'

        # Write the header
        with open(self.experiment_data_filename, 'w+') as data_file:
            for field in self.data_fields:
                data_file.write('"')
                data_file.write(field)
                data_file.write('"')
                if field != self.data_fields[-1]:
                    data_file.write(',')
            data_file.write('\n')

    def chdir(self):
        """Changes the directory to where the data will be saved."""
        try:
            os.makedirs(self.data_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        os.chdir(self.data_directory)
            
            
    def calculate_locations(self, set_size):
        """
        Generate random screen locations with specified constraints.

        Parameters:
        - set_size: Number of locations to generate on each side

        Returns:
        - List of (x, y) coordinate tuples for locations
        """
        min_distance = self.stim_size * 4.5
        regions = [(-7, 0-min_distance/2), (0+min_distance/2, 7)]
        region_locations = [[], []]  # List to store locations for each region

        for region_index, region in enumerate(regions): 
            region_locations[region_index] = []  # Initialize the current region's locations list 
            attempts = 0
            max_attempts = 1000

            while len(region_locations[region_index]) < set_size: 
                attempts += 1
                if attempts > max_attempts:
                    raise ValueError(f"Could not find valid positions for all stimuli in region {region_index} after {max_attempts} attempts.")

                x = np.random.uniform(region[0], region[1])
                y = np.random.uniform(-5.2, 5.2)

                valid = True
                for existing_loc in region_locations[region_index]: 
                    if np.sqrt((x - existing_loc[0])**2 + (y - existing_loc[1])**2) < min_distance:
                        valid = False
                        break

                if valid:
                    region_locations[region_index].append((x, y))

        return region_locations

    def make_trial(self, set_size, colors):
        """Creates a single trial dictionary."""

        color_values_left = {name: colors[name] for name in random.sample(list(colors), set_size)}
        color_values_right = {name: colors[name] for name in random.sample(list(colors), set_size)}

        locations = self.calculate_locations(set_size)

        color_values = [list(color_values_left.values()), list(color_values_right.values())] #Create the list of lists

        trial = {
            'set_size': set_size,
            'color_values': color_values, #Use the new list of lists
            'locations': locations
        }

        return trial

    def make_block(self, colors):
        """Makes a block of trials.

        Returns a shuffled list of trials created by self.make_trial.
        """

        trial_list = []

        for set_size in self.set_sizes:
            for _ in range(self.trials_per_set_size):
                trial = self.make_trial(set_size, colors)
                trial_list.append(trial)

        random.shuffle(trial_list)

        return trial_list
        

    def display_blank(self, wait_time):
        """
        Displays a blank screen.

        Parameters:
            wait_time -- The number of seconds to display the blank for.
        """
        # Display fixation
        fixation = visual.Circle(self.experiment_window, radius=0.06, fillColor='black')
        
        fixation.draw()
        ts = self.experiment_window.flip()

        psychopy.core.wait(wait_time)
        
        return ts
        
        
    def display_cue(self, cue_time):
        
        # Display fixation
        fixation = visual.Circle(self.experiment_window, radius=0.06, fillColor='black')
        
        # Display cue
        cue_dir = random.choice([-1, 1])  # Choose left (-1) or right (+1) randomly
        
        tsz = 0.2
        tshft = 0.4
        
        arrow_purple = (0.5, 0, 0.5) #(0.6667, 0.3922, 0.6667) 
        arrow_green = (0, 0.5, 0) #(0.6667, 0.3922, 0.6667)
        
        if cue_dir == -1:
            leftColor = arrow_green
            rightColor = arrow_purple
        elif cue_dir == 1:
            rightColor = arrow_green
            leftColor = arrow_purple
        
        # Create two triangles
        triangle_left = visual.ShapeStim(
            win=self.experiment_window, 
            vertices=((0, tsz+tshft), (0, -tsz+tshft), (-2*tsz, tshft)),
            lineWidth=1.0, 
            closeShape=True,
            fillColor=leftColor,
            pos=(0, 0) 
        )

        triangle_right = visual.ShapeStim(
            win=self.experiment_window, 
            vertices=((0, tsz+tshft), (0, -tsz+tshft), (2*tsz, tshft)),
            lineWidth=1.0, 
            closeShape=True,
            fillColor=rightColor,
            pos=(0, 0) 
        )

        # Draw the shapes
        fixation.draw()
        triangle_left.draw()
        triangle_right.draw()
        
        ts = self.experiment_window.flip()
        
#   Code used in Berger task to trigger closedloop and bookkeeping indicating when it happened
#        time_elapsed = t
#
#        # Check and act upon the 500ms threshold
#        if not flag_500ms_triggered and t >= 0.5:
#            print("First message at 500ms")
#            thisExp.addData('Flag500ms', str(t))
#            flag_500ms_triggered = True  # Set the flag to prevent re-triggering
#            client_socket.sendall(b'ready')
#
#        # Check and act upon the 1500ms threshold
#        if not flag_1500ms_triggered and t >= 1.5:
#            print("Second message at 1500ms")
#            thisExp.addData('Flag1500ms', str(t))
#            flag_1500ms_triggered = True  # Set the flag to prevent re-triggering
#            client_socket.sendall(b'done')

#        client_socket.sendall(b'ready')

#        try:
#            data = client_socket.recv(1024)
#        except socket.timeout:
#            print("Cue stimulation timed out")
#            data = None
#        except BlockingIOError:
#            data = None
#        except ConnectionResetError:
#            data = None
#        finally:
#            print("data: " + str(data))

        psychopy.core.wait(cue_time)  # Display cue for 500 ms

#        client_socket.sendall(b'done')

        return cue_dir, ts
    

    def display_stimuli(self, coordinates, colors):
        """Displays the stimuli.

        Parameters:
            coordinates -- A list of lists of (x, y) tuples in visual degrees.
            colors -- A list of lists of -1 to 1 rgb color lists.
        """
        
        for region_coords, region_colors in zip(coordinates, colors): #Iterate through the regions
            for pos, color in zip(region_coords, region_colors): #Iterate through the coords and colors of each region
                psychopy.visual.Rect(
                    self.experiment_window, width=self.stim_size*3, height=self.stim_size*3, pos=pos, fillColor=color,
                    units='deg', lineColor=None).draw()
                    
        # Display fixation
        fixation = visual.Circle(self.experiment_window, radius=0.06, fillColor='black')
        fixation.draw()
        
        ts = self.experiment_window.flip()
        psychopy.core.wait(self.sample_time)
        
        return ts


    def draw_color_grid(self, coordinates, colors):
        """
        Draws a 3x3 grid of predefined color patches at stimuli locations,
        with the center square left empty.

        Parameters:
            coordinates -- A list of (x, y) tuples, where each tuple represents a location on the screen
        """
        
        # Coordinates for the grid arrangement
        grid_positions = [
            (-1, 1), (0, 1), (1, 1),
            (-1, 0), (0, 0), (1, 0),
            (-1, -1), (0, -1), (1, -1)
        ]
        
        color_values = list(colors.values())
        
        for region_coords in coordinates:
            for pos in region_coords:
                # Place each of the 8 colors in a 3x3 grid, leaving the center empty
                color_index = 0  # Start with the first color
                for i, (dx, dy) in enumerate(grid_positions):

                    color_patch = visual.Rect(
                        self.experiment_window, 
                        width=self.stim_size, 
                        height=self.stim_size, 
                        fillColor=color_values[color_index], 
                        lineColor=None, 
                        pos=(pos[0] + dx * self.stim_size, pos[1] + dy * self.stim_size)
                    )
                    color_patch.draw()

                    color_index += 1  # Increment the color index



    def _calc_mouse_color(self, mouse_pos):
        """
        Calculates the color of the pixel the mouse is hovering over.

        Parameters:
            mouse_pos -- A position returned by mouse.getPos()
        """
        frame = np.array(self.experiment_window._getFrame())  # Uses psychopy internal function

        x_correction = self.experiment_window.size[0] / 2
        y_correction = self.experiment_window.size[1] / 2

        x = int(psychopy.tools.monitorunittools.deg2pix(mouse_pos[0], self.experiment_monitor) + x_correction)
        y = (self.experiment_window.size[1] -
             int(psychopy.tools.monitorunittools.deg2pix(mouse_pos[1], self.experiment_monitor) + y_correction))

        try:
            color = frame[y, x, :]
        except IndexError:
            color = None

        return color

    def _calc_mouse_position(self, coordinates, mouse_pos):
        """
        Determines which position is closest to the mouse in order to display the hover preview.

        Parameters:
            coordinates -- A list of (x, y) tuples
            mouse_pos -- A position returned by mouse.getPos()
        """
        dists = [np.linalg.norm(np.array(i) - np.array(mouse_pos)) for i in coordinates]
        closest_dist = min(dists)

        if closest_dist < 4:
            return coordinates[np.argmin(dists)]
        else:
            return None

    def _response_loop(self, coordinates, cue_dir):
        """
        Handles the hover updating and response clicks for coordinates as a list of lists.

        Parameters:
            coordinates -- A list of lists of (x, y) tuples
        """
        # Flatten coordinates for processing
        flat_coordinates = [coord for region in coordinates for coord in region]
        
        region_map = {0: [], 1: []}

        for i, coord in enumerate(flat_coordinates):
            region_map[0 if coord[0] < 0 else 1].append(coord)
        
        # Display fixation
        fixation = visual.Circle(self.experiment_window, radius=0.06, fillColor='black')

        
        # Track selected positions instead of removing from list
        selected_positions = set()

        resp_colors = [0] * len(flat_coordinates)
        rts = [0] * len(flat_coordinates)
        click_order = [0] * len(flat_coordinates)

        click = 1

        self.mouse.clickReset()

        self.draw_color_grid(coordinates, colors)
        ts = self.experiment_window.flip()
        
        cue_side_positions = region_map[0 if cue_dir == -1 else 1]

        while True:
            if psychopy.event.getKeys(keyList=['q']):
                self.quit_experiment()

            (lclick, _, _), (rt, _, _) = self.mouse.getPressed(getTime=True)
            
            if lclick:
                mouse_pos = self.mouse.getPos()
                px_color = self._calc_mouse_color(mouse_pos)

                if px_color is not None and not np.array_equal(px_color, np.array([127, 127, 127])):
                    preview_pos = self._calc_mouse_position(flat_coordinates, mouse_pos)

                    if preview_pos and preview_pos not in selected_positions:
                        pos_index = flat_coordinates.index(preview_pos)
                        resp_colors[pos_index] = px_color
                        rts[pos_index] = rt
                        click_order[pos_index] = click
                        click += 1
                        selected_positions.add(preview_pos)

                    # Check if all cues on cue_dir are completed
                    if all(pos in selected_positions for pos in cue_side_positions):
                        return resp_colors, rts, click_order, ts

                    #if len(selected_positions) == len(flat_coordinates):
                        #return resp_colors, rts, click_order


            temp_coordinates = [
                [pos for pos in region if pos not in selected_positions] 
                for region in coordinates
            ]
            
            self.draw_color_grid(temp_coordinates, colors)
            fixation.draw()
            self.experiment_window.flip()
            

    def get_response(self, coordinates, cue_dir):
        """
        Manages getting responses for all color wheels.

        Parameters:
            coordinates -- A list of (x, y) tuples
            wheel_rotations -- A list of 0:359 ints describing how much each wheel
                should be rotated.
                
        """
        
        if not self.mouse:
            self.mouse = psychopy.event.Mouse(visible=False, win=self.experiment_window)

        self.mouse.setVisible(1)
        psychopy.event.clearEvents()

        resp_colors, rts, click_order,  ts = self._response_loop(coordinates, cue_dir)

        self.mouse.setVisible(0)

        return resp_colors, rts, click_order, ts

    def send_data(self, data):
        """Updates the experiment data with the information from the last trial.

        This function is seperated from run_trial to allow additional information to be added
        afterwards.

        Parameters:
            data -- A dict where keys exist in data_fields and values are to be saved.
        """
        self.update_experiment_data(data)
        
    def run_trial(self, trial, block_num, trial_num):
        """
        Runs a single trial.

        Parameters:
            trial -- A dictionary returned by make_trial().
            block_num -- The block number to be saved in the output csv.
            trial_num -- The trial number to be saved in the output csv.
        """
        ts_iti = self.display_blank(self.iti_time)
        cue_dir, ts_cue = self.display_cue(self.cue_time)
        ts_stim = self.display_stimuli(trial['locations'], trial['color_values'])
        ts_delay = self.display_blank(self.delay_time)
        resp_colors, rts, click_order, ts_resp = self.get_response(trial['locations'], cue_dir)

        data = []
        timestamp = psychopy.core.getAbsTime()

        # Flatten locations and color values for iteration
        all_locations = [loc for region in trial['locations'] for loc in region]
        all_colors = [color for region in trial['color_values'] for color in region]
        color_names = list(colors.keys())
        
        for i, (pos, true_color) in enumerate(zip(all_locations, all_colors)):
            thisRespColor = resp_colors[i]
            if not np.array_equal(thisRespColor, 0):
                thisRespColor = tuple(template.convert_color_value(thisRespColor))
                
                # Find the response color name
                resp_color_name = next(
                    (name for name, color in colors.items() if color == thisRespColor), "Unknown"
                )
                
                accuracy = int(true_color == thisRespColor)
                
            else:  # No response, which occurs if all items in cued set were answered
                thisRespColor = (float('nan'), float('nan'), float('nan'))
                resp_color_name = 'none'
                rts[i] = float('nan')
                accuracy = float('nan')
                click_order[i] = float('nan')
                
            in_cued_set = (cue_dir == -1 and pos[0] < 0) or (cue_dir == 1 and pos[0] > 0)
            
            # Determine which region the location belongs to
            region_index = 0 if pos[0] < 0 else 1
            
            data.append({
                'Subject': self.experiment_info['Subject Number'],
                'Session': self.experiment_info['Session'],
                'Block': block_num+1, # the +1 makes it 1 indexed in the record
                'Trial': trial_num+1, # the +1 makes it 1 indexed in the record
                'LocationNumber': i + 1,
                'ClickNumber': click_order[i],
                'TS_ITI': ts_iti,
                'TS_Cue': ts_cue,
                'TS_Stim': ts_stim,
                'TS_Delay': ts_delay,
                'TS_Resp': ts_resp,
                'Timestamp': timestamp,
                'SetSize': trial['set_size'],
                'LocationX': pos[0],
                'LocationY': pos[1],
                'ColorIndex': color_names.index(list(colors.keys())[list(colors.values()).index(true_color)]),
                'TrueColor': true_color,
                'InCuedSet': in_cued_set,
                'TrueColorName': list(colors.keys())[list(colors.values()).index(true_color)],
                'RespColor': thisRespColor,
                'RespColorName': resp_color_name,
                'Accuracy': accuracy,
                'RT': rts[i],
            })

        return data
        


    def display_break(self):
        """Displays a break screen in between blocks."""

        break_text = 'Please take a short break. Press space to continue.'
        self.display_text_screen(text=break_text, bg_color=[204, 255, 204])

    def run(self, setup_hook=None, before_first_trial_hook=None, pre_block_hook=None,
            pre_trial_hook=None, post_trial_hook=None, post_block_hook=None,
            end_experiment_hook=None):
        """Runs the entire experiment.

        This function takes a number of hooks that allow you to alter behavior of the experiment
        without having to completely rewrite the run function. While large changes will still
        require you to create a subclass, small changes like adding a practice block or
        performance feedback screen can be implimented using these hooks. All hooks take in the
        experiment object as the first argument. See below for other parameters sent to hooks.

        Parameters:
            setup_hook -- takes self, executed once the window is open.
            before_first_trial_hook -- takes self, executed after instructions are displayed.
            pre_block_hook -- takes self, block list, and block num
                Executed immediately before block start.
                Can optionally return an altered block list.
            pre_trial_hook -- takes self, trial dict, block num, and trial num
                Executed immediately before trial start.
                Can optionally return an altered trial dict.
            post_trial_hook -- takes self and the trial data, executed immediately after trial end.
                Can optionally return altered trial data to be stored.
            post_block_hook -- takes self, executed at end of block before break screen (including
                last block).
            end_experiment_hook -- takes self, executed immediately before end experiment screen.
        """
        self.chdir()

        ok = self.get_experiment_info_from_dialog(self.questionaire_dict)

        if not ok:
            print('Experiment has been terminated.')
            sys.exit(1)

        self.save_experiment_info()
        self.open_csv_data_file()
        self.open_window(screen=1)
        self.display_text_screen('Loading...', wait_for_input=False)

        if setup_hook is not None:
            setup_hook(self)

        for instruction in self.instruct_text:
            self.display_text_screen(text=instruction)

        if before_first_trial_hook is not None:
            before_first_trial_hook(self)

        for block_num in range(self.number_of_blocks):
            block = self.make_block(colors)

            if pre_block_hook is not None:
                tmp = pre_block_hook(self, block, block_num)
                if tmp is not None:
                    block = tmp

            for trial_num, trial in enumerate(block):
                if pre_trial_hook is not None:
                    tmp = pre_trial_hook(self, trial, block_num, trial_num)
                    if tmp is not None:
                        trial = tmp

                data = self.run_trial(trial, block_num, trial_num)

                if post_trial_hook is not None:
                    tmp = post_trial_hook(self, data)
                    if tmp is not None:
                        data = tmp

                self.send_data(data)

            self.save_data_to_csv()

            if post_block_hook is not None:
                post_block_hook(self)

            if block_num + 1 != self.number_of_blocks:
                self.display_break()

        if end_experiment_hook is not None:
            end_experiment_hook(self)

        self.display_text_screen(
            'The experiment is now over, please get your experimenter.',
            bg_color=[0, 0, 255], text_color=[255, 255, 255])

        self.quit_experiment()


# If you call this script directly, the task will run with your defaults
if __name__ == '__main__':
    exp = ResolutionWR(
        # BaseExperiment parameters
        experiment_name=experiment_name,
        data_fields=data_fields,
        monitor_distance=monitor_distance,
        monitor_name = 'LG Monitor (Subject Monitor)'
        # Custom parameters go here
    )

    exp.run()
