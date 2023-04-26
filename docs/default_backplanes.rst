..
   THIS CONTENT IS AUTOMATICALLY GENERATED

.. _default backplanes:

Default backplanes
******************

This page lists the backplanes which are automatically registered to every instance of :class:`planetmapper.BodyXY`.

------------

`LON-GRAPHIC` Planetographic longitude, positive W [deg]

- Image function: :func:`planetmapper.BodyXY.get_lon_img`
- Map function: :func:`planetmapper.BodyXY.get_lon_map`

------------

`LAT-GRAPHIC` Planetographic latitude [deg]

- Image function: :func:`planetmapper.BodyXY.get_lat_img`
- Map function: :func:`planetmapper.BodyXY.get_lat_map`

------------

`LON-CENTRIC` Planetocentric longitude [deg]

- Image function: :func:`planetmapper.BodyXY.get_lon_centric_img`
- Map function: :func:`planetmapper.BodyXY.get_lon_centric_map`

------------

`LAT-CENTRIC` Planetocentric latitude [deg]

- Image function: :func:`planetmapper.BodyXY.get_lat_centric_img`
- Map function: :func:`planetmapper.BodyXY.get_lat_centric_map`

------------

`RA` Right ascension [deg]

- Image function: :func:`planetmapper.BodyXY.get_ra_img`
- Map function: :func:`planetmapper.BodyXY.get_ra_map`

------------

`DEC` Declination [deg]

- Image function: :func:`planetmapper.BodyXY.get_dec_img`
- Map function: :func:`planetmapper.BodyXY.get_dec_map`

------------

`PIXEL-X` Observation x pixel coordinate [pixels]

- Image function: :func:`planetmapper.BodyXY.get_x_img`
- Map function: :func:`planetmapper.BodyXY.get_x_map`

------------

`PIXEL-Y` Observation y pixel coordinate [pixels]

- Image function: :func:`planetmapper.BodyXY.get_y_img`
- Map function: :func:`planetmapper.BodyXY.get_y_map`

------------

`KM-X` East-West distance in target plane [km]

- Image function: :func:`planetmapper.BodyXY.get_km_x_img`
- Map function: :func:`planetmapper.BodyXY.get_km_x_map`

------------

`KM-Y` North-South distance in target plane [km]

- Image function: :func:`planetmapper.BodyXY.get_km_y_img`
- Map function: :func:`planetmapper.BodyXY.get_km_y_map`

------------

`PHASE` Phase angle [deg]

- Image function: :func:`planetmapper.BodyXY.get_phase_angle_img`
- Map function: :func:`planetmapper.BodyXY.get_phase_angle_map`

------------

`INCIDENCE` Incidence angle [deg]

- Image function: :func:`planetmapper.BodyXY.get_incidence_angle_img`
- Map function: :func:`planetmapper.BodyXY.get_incidence_angle_map`

------------

`EMISSION` Emission angle [deg]

- Image function: :func:`planetmapper.BodyXY.get_emission_angle_img`
- Map function: :func:`planetmapper.BodyXY.get_emission_angle_map`

------------

`AZIMUTH` Azimuth angle [deg]

- Image function: :func:`planetmapper.BodyXY.get_azimuth_angle_img`
- Map function: :func:`planetmapper.BodyXY.get_azimuth_angle_map`

------------

`DISTANCE` Distance to observer [km]

- Image function: :func:`planetmapper.BodyXY.get_distance_img`
- Map function: :func:`planetmapper.BodyXY.get_distance_map`

------------

`RADIAL-VELOCITY` Radial velocity away from observer [km/s]

- Image function: :func:`planetmapper.BodyXY.get_radial_velocity_img`
- Map function: :func:`planetmapper.BodyXY.get_radial_velocity_map`

------------

`DOPPLER` Doppler factor, sqrt((1 + v/c)/(1 - v/c)) where v is radial velocity

- Image function: :func:`planetmapper.BodyXY.get_doppler_img`
- Map function: :func:`planetmapper.BodyXY.get_doppler_map`

------------

`RING-RADIUS` Equatorial (ring) plane radius [km]

- Image function: :func:`planetmapper.BodyXY.get_ring_plane_radius_img`
- Map function: :func:`planetmapper.BodyXY.get_ring_plane_radius_map`

------------

`RING-LON-GRAPHIC` Equatorial (ring) plane planetographic longitude [deg]

- Image function: :func:`planetmapper.BodyXY.get_ring_plane_longitude_img`
- Map function: :func:`planetmapper.BodyXY.get_ring_plane_longitude_map`

------------

`RING-DISTANCE` Equatorial (ring) plane distance to observer [km]

- Image function: :func:`planetmapper.BodyXY.get_ring_plane_distance_img`
- Map function: :func:`planetmapper.BodyXY.get_ring_plane_distance_map`

------------

Wireframe images
================

In addition to the above backplanes, a `WIREFRAME` backplane is also included by default in saved FITS files. This backplane contains a "wireframe" image of the body, which shows latitude/longitude gridlines, labels poles, displays the body's limb etc. These wireframe images can be used to help orient the observations, and can be used as an overlay if you are creating figures from the FITS files.

The wireframe images are a graphical guide rather than containing any scientific data, so they are not registered like the other backplanes. Note that the wireframe images have a fixed size, so they will not be the same size as the data/mapped data (although the aspect ratio will be the same).

- Image function: :func:`planetmapper.BodyXY.get_wireframe_overlay_img`
- Map function: :func:`planetmapper.BodyXY.get_wireframe_overlay_map`
