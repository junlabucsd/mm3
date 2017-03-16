









if platform == "linux" or platform == "linux2":
    # linux
    import gevent_inotifyx as inotify
    from gevent.queue import Queue as gQueue
    use_inotify = True
elif platform == "darwin":
    # OS X
    from watchdog.observers import Observer
    from watchdog.events import PatternMatchingEventHandler
    use_watchdog = True



# class which responds to file addition events and saves their file names
class tif_sentinal(PatternMatchingEventHandler):
    '''tif_sentinal will react to file additions that have .tif at the end.
    It only looks at file creations, not modifications or deletions.
    It inherits from the pattern match event handler in watchdog.
    There are two class attributes, events_filenames and events_buffer.
    events_filenames is the name of all added .tif files. Use the
    self.get_filenames method to return all events.
    events_buffer are just the filenames since last time the buffer was returned
    use self.get_buffer to return the buffer (which also clears the buffer).
    '''
    patterns = ["*.tif"]

    def __init__(self):
        # Makes sure to do the init function from the parent (super) class first
        super(tif_sentinal, self).__init__()
        self.events_filenames = [] # keeps all event filenames
        self.events_buffer = [] # keeps just event filenames since last call to buf

    def process(self, event):
        """
        called when an event (file change) occurs

        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_pquitath
            path/to/observed/file
        """
        filename = event.src_path.split('/')[-1] # get just the file name
        # append the file name to the class attributes for later retrieving
        self.events_filenames.append(filename)
        self.events_buffer.append(filename)

    def get_filenames(self):
        '''returns all event file names'''
        return self.events_filenames

    def get_buffer(self):
        '''returns all events since last call to print buffer'''
        # make a temporary buffer for printing
        buf = [x for x in self.events_buffer]
        self.events_buffer = [] # reset buffer
        return buf

    # could use this to do something on file modifications
    # def on_modified(self, event):
    #     self.process(event)

    def on_created(self, event):
        self.process(event)


if __name__ == "__main__":


    # get switches and parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:")
        param_file_path = ''
    except getopt.GetoptError:
        print('No arguments detected (-f).')

    # set parameters
    for opt, arg in opts:
        if opt == '-f':
            param_file_path = arg # parameter file path

    # Load the project parameters file
    if len(param_file_path) == 0:
        raise ValueError("A parameter file must be specified (-f <filename>).")
    information ('Loading experiment parameters.')
    with open(param_file_path, 'r') as param_file:
        p = yaml.safe_load(param_file) # load parameters into dictionary

    # Load the channel_masks file

    # Load specs file


    ### Pre watching loop

    # intialize known files list and files to be analyzed list

    # first determine known files
    # you can do this by looping through the HDF5 files and looking at the list 'filenames'
    # you could alternatively load the pickle file TIFF_metdata, though it is quite large.

    # Find the current files in the TIFF folder

    # Filter for known and unknown files
    # Add known files to the list

    # Add unknown files to the to be analyzed list

    ### Now begin loop
    # Start with analysis of unknown files.
    # Organize images by FOV.

    # Send that batch of images to a function for processing.
    # This should be the multiprocessing step.

    # Within that function.

    # Get the metadata (but skip finding channels)
    # --> get_tif_params in mm3_Compile. maybe move to mm3_Helpers

    # Slice out the channels from the TIFFs
    # must have been sent the channel_masks file.
    # could also use the specs file to only analyze channels that have cells in them...

    # Send them emtpy channels for averaging

    # Subtrat the averaged empty from the others

    # Segment them.

    # All of that should have been done in a pool for all FOVs, wait for that pool to finish.

    # move the analyzed files, if the results were successful, to the analyzed list.

    # Check to see if new files have been added to the TIFF folder and put them in the known files
    # list
