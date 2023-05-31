PlanetMapper
************
A Python module for visualising, navigating and mapping Solar System observations


Key features
============
:ref:`Fit and map astronimical observations using a full featured user interface <gui examples>`
------------------------------------------------------------------------------------------------
.. image:: images/gui_fitting.png
    :width: 600
    :alt: Screenshot of the fitting window after the disc is fit.


:ref:`Easily visualise solar system observations with just a few lines of code <wireframes>`
--------------------------------------------------------------------------------------------
::

   body = planetmapper.Body('saturn', '2020-01-01')
   body.plot_wireframe_radec()
   plt.show()

.. image:: images/saturn_wireframe_radec.png
    :width: 600
    :alt: Plot of Saturn

:ref:`Convert coordinates, generate backplanes and project maps of telescope observations <python examples>`
------------------------------------------------------------------------------------------------------------
.. image:: images/jupiter_mapped.png
    :width: 800
    :alt: Plot of a mapped Jupiter observation


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   spice_kernels

.. toctree::
   :maxdepth: 2
   :caption: Examples
   
   user_interface
   general_python_api

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   documentation
   base
   gui
   utils
   data_loader
   kernel_downloader


.. toctree::
   :maxdepth: 2
   :caption: Appendix

   default_backplanes
   common_issues
   credits
