Graphical user interface
************************

The user interface is a convenient and easy way to fit an observation, generate backplanes and map the observed data.

Starting the user interface
===========================
The easiest way to run the PlanetMapper user interface is to simply type `planetmapper` in the command line. This will launch a new interactive window where you can choose observation files to open and fit. 

You can create a user interface from a :class:`planetmapper.Observation` object using the  :func:`planetmapper.Observation.run_gui` function. This is mainly useful if you want to combine using a user interface to fit the observation with some Python code to e.g. run some additional analysis. 

It is also possible to start a user interface directly using :func:`planetmapper.gui.GUI.run`, although the other two methods are generally more useful.


Example: fitting an observation
===============================
To start, type `planetmapper` into a command line and press enter. This will open a window where you can choose a file to open:
 
.. image:: images/gui_open.png
    :width: 600
    :alt: User interface window with options to choose which file to open.

If your data is a FITS file, PlanetMapper will attempt to automatically fill the target, date and observer fields for you with information from the FITS header (but it's worth double checking that the values are what you expect). The date should be in a format which `can be understood by SPICE <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/utc2et_c.html#Examples>`_ (such as `YYYY-mm-ddTHH:MM:SS`) and should be in UTC. You can also specify a list of :ref:`SPICE kernels` to load here - if you're unsure then the default values will probably work.

Once you click OK, the full fitting window should open. If you get any error messages, then double check the target, date and observer fields for any typos.

.. image:: images/gui_fitting_initial.png
    :width: 600
    :alt: Screenshot of the fitting window before the disc is fit.

This window allows you to fit the observation, so that the fitted disc (the white circle) overlaps nicely with the observed disc. You can use the buttons on the left hand side to move the disc around, or input specific values in the text boxes (for example, you may know the plate scale in arcsec/pixel of the telescope you are using). You can also find the keyboard shortcut for each button by hovering over it and reading the hint at the bottom of the window.

.. image:: images/gui_fitting.png
    :width: 600
    :alt: Screenshot of the fitting window after the disc is fit.

Once the disc is fit, it should look something like the image above. If you want more fine control from clicking the buttons, then you can adjust the step size. It can often be useful to start with a large step size, then decrease it for the final fine alignment.

.. image:: images/gui_customisation.png
    :width: 600
    :alt: Screenshot of the customisation options.

You can also fully customise the appearance of the plot on the right to make fitting easier (or if you just fancy a more exciting colour scheme). In the settings tab, you can toggle the visibility of different plotted elements, and you can click on Edit to customise them further. It can be particularly useful to customise the colour scale and brightness of the observed image to increase the contrast around the limb.

You can also use the settings tab to mark points of interest to help with fitting. For example:

- You can mark a specific location (e.g. a distinctive impact crater) on the surface of the target with a lon/lat POI.
- You can mark a specific sky coordinate (e.g. a background star) with a RA/Dec POI.
- You can mark the location of other bodies (e.g. if you are fitting an observation of Jupiter, you may want to mark the positions of any of its moons which are also in shot). 

.. image:: images/gui_saving.png
    :width: 600
    :alt: Screenshot of the saving options window.

Once you are happy with the fitting result, click Save at the top of the Controls tab. This will open a window where you can choose which files to output. You can customise which files to output (with the 'Save navigated observation' and 'Save mapped observation' checkboxes) and choose the filepath where these files will be saved.

- The navigated observation is similar to the input file, with additional 'FITS backplanes' containing useful information such as the longitude/latitude coordinates for each pixel in the image.
- The mapped observation produces a FITS file which contains (as the name suggests...) a mapped version of the observation. This map file will also contain the various useful backplanes. The degree interval option allows you to customise the size of the output map (e.g. degree interval=1 produces a map which is 180x360, degree interval=10 produces a map which is 18x36).

Once you click Save, your requested files will be generated and saved. Note that for larger files, this can take around a minute to complete as some of the coordinate conversion calculations are relatively complex.

Example: running the UI from Python
===================================
This simple example shows how you could use :func:`planetmapper.Observation.run_gui` from a Python script to fit multiple observations, then run some custom code on each of them: ::

    import glob
    import planetmapper

    for path in sorted(glob.glob('data/*.fits')):
        observation = planetmapper.Observation(path)

        # Run some custom setup
        observation.add_other_bodies_of_interest('Io', 'Europa', 'Ganymede', 'Callisto')
        observation.set_plate_scale_arcsec(42)

        # Run the GUI to fit the observation interactively
        # this will open a GUI window every loop
        observation.run_gui()

        # More custom code can go here to use the fitted observation...