# Secure Server for Connected Vehicles

This is a demo website for our UM-Ford research project. It supports the following functions:
1. encryption (AES, Base64)
2. k-anonymity (for demo use, based on adult.data from [Mondrian](https://github.com/qiyuangong/Mondrian))
3. top location inference and defense 
4. fingerprinting attacks and defense

The website is based on Django web framework and requires some Python libraries (e.g., sklearn for machine learning, numpy/scipy/pandas for processing data matrix, MySQLdb for connecting MySQL database).

Note: main entry for the demo is queryNewData.py file (which means queryData.py is out of date), so after running the sever in your local terminal, you can input "http://localhost:8000/query-new-data" to see the interface, Or "http://[host]:[port]/query-new-data" in a general form by replacing your own host and port in the URL repsectively. More about Django please see [Django CookBook](https://code.djangoproject.com/wiki/CookBook).

## Top location inference and defense
Based on new Safty Pilot data downloaded from Globus, please find the data format at [here](https://github.com/caocscar/ConnectedVehicleDocs/blob/master/BSMdocumentation.md). I just import each individual's data as it is into MySQL database, and the code will extract GPS data and run clustering algorithms to calculate centroid as inferred top locations. It has functions to compute top location for an individual based on [all queried data | daily data in queried data | weekly data in queried data], please find details in code and comments. My experiments and codes are based on data in April 2016.

## Fingerprinting attacks and defense
Based on old data which I queried from UMTRI's server directly in the late 2016, so the data format is different from the data on Globus. These data are in the "data_for_fingerprinting.zip" file at: https://umich.box.com/s/gctyig0s8p0crcxbowlcvui2cuh1gm4i  and it totally contains data for four road segments (annbor.csv, plymonth.csv, vertical.csv, straight.csv). Each road contains multiple drivers and each drivers drove multiple times through this road. So you need to import these files into tables and you can query one table and conduct fingerprinting attacks on drivers on this road. In the code, I select a subset of drivers (frequently driving drivers) to conduct fingerprinting to avoid bias due to insufficient driving data on this road. So you need to modify the commented code to select corresponding drivers filters in the experiments. For example, if you conduct fingerprinting on the file "annbor.csv
, then you query data from the table containing data from this file, and uncomment the code under "high support for annbor" and comment codes for other roads in fingerprinting functions.

The format is 
`Device	Trip	Time	AccelPedal	Altitude	AvailableLeft	AvailableRight	Ax	Ay	BoundaryLeft	BoundaryRight	Brake	CruiseEngaged	Distance	GpsHeading	Latitude	Longitude	NumTargets	Range	RangeRate	Speed	TurnSignal	YawRate`

Some explainations for these data:

|Data|Type|Unit|Explaination|
|----|----|----|----|
|Time|Long Integer|csec|Time in centiseconds since das started|
|AccelPedal|Single Float|%|Accelerator pedal|
|Altitude|Single Float|m|Height above the ellipsoid|
|AvailableLeft|Byte|none|MobilEye left LDW availability|
|AvailableRight|Byte|none|MobilEye right LDW availability|
|Ax|Single Float|m/sec2|Longitudinal accel from Conti IMU|
|Ay|Single Float|m/sec2|Lateral accel from Conti IMU|
|BoundaryLeft|Byte|none|MobilEye left lane type|
|BoundaryRight|Byte|none|MobilEye right lane type|
|Brake|Byte|none|Brake light active|
|CruiseEngaged|Byte|none|Cruise control active|
|Distance|Single Float|m|Trip distance|
|GpsHeading|Single Float|deg|Gps heading from Ublox Gps|
|Latitude|Double Float|deg|Latitude from Ublox Gps|
|Longitude|Double Float|deg|Longitude from Ublox Gps|
|NumTargets|Byte|none|MobilEye number of obstacles|
|Range|Single Float|m|MobilEye Object 1 longitudinal position relative to the reference point.|
|RangeRate|Single Float|m/sec|MobilEye relative longitudinal velocity of object1|
|Speed|Single Float|m/sec|Vehicle speed from transmission|
|TurnSignal|Byte|none|Turn signal|
|YawRate|Single Float|deg/sec|Yawrate from Conti IMU|
