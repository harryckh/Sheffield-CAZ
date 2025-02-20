# SUFO Traffic

# Series of functions to read and deal with traffic data from SUFo

def locate_traffic_sites(lat, long, radius,start_date,end_date,add_distance = 0):
    #Function that lists all the traffic sensors within a given radius of the target coords

    ## Function loads SUFO data for the time frame given and lists all SCC_flow sensors in the data
    ## To maintain speed the timeframe is limited to 1 month, and a maximum radius of 1km
    ## Add_distance is disabled by default but allows for the distance of each sensor to also be calculated, this should be off for plotting.

    import requests
    from datetime import datetime
    import SUFO_Traffic

    #Convert to datetime objects
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    #Check end is after start
    if end_dt < start_dt:
        raise ValueError(f"End date cannot be before start date")
    
    #Add a warning for if the date is too long
    if (end_dt - start_dt).days > 35:
        raise ValueError("Large timeframe, ensure you are requesting no more than one month")
    
    #Custom URL
    url = "https://ufdev21.shef.ac.uk/sufobin/sufoDXT?Tfrom=" + start_date + "&Tto=" + end_date + "&midLon=" + long + "&midLat=" + lat + "&zRad=" + radius + "&byFamily=SCC_flow&freqInMin=1&qcopt=prunedata&udfnoval=-32768&udfbelow=-32769&udfabove=-32767&hrtFormat=iso8601&tabCont=rich&gdata=byPairId&src=data&op=getdata&fmt=jsonrows&output=zip&tok=generic&spatial=zone"
    
    #Now get the data and search
    try:
        response = requests.get(url)
        response.raise_for_status()  #error if issue
        request_data = response.json()
        
        sensor_locs = {}

        for x in range(request_data["nBundles"]):
            sensor_lat = request_data["bundles"][x]["location"]["latitude"]
            sensor_long = request_data["bundles"][x]["location"]["longitude"]
            sensor_name = request_data["bundles"][x]["identity"]["site.id"]

            if add_distance != 0:
                distance = SUFO_Traffic.line_distance(lat,long,sensor_lat,sensor_long)
                sensor_locs[sensor_name] = (sensor_lat, sensor_long), distance
            else:
                sensor_locs[sensor_name] = (sensor_lat, sensor_long)

        return sensor_locs

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except ValueError:  #ValueError will occur for JSON parse failure
        print("No data for specified parameters")
        return None
    
def line_distance(lat1,long1,lat2,long2):
    #Funciton that calculates the straight-line distance between two points using the haversine formula
    import math
     # Earth radius in kilometers
    R = 6371.0

    #Make sure inputs are numbers
    lat1_f = float(lat1)
    long1_f = float(long1)
    lat2_f = float(lat2)
    long2_f = float(long2)

    # Convert latitude and longitude from degrees to radians
    lat1_f = math.radians(lat1_f)
    long1_f = math.radians(long1_f)
    lat2_f = math.radians(lat2_f)
    long2_f = math.radians(long2_f)

    # Compute differences
    dlat = lat2_f - lat1_f
    dlon = long2_f - long1_f

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_f) * math.cos(lat2_f) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance

    
def calculate_bounding_box(lat:str, lon:str, radius:str):

    #Function that will calculate a bounding box from a coord and a radius
    import math

    #Convert from strings to floats
    lat_f = float(lat)
    lon_f = float(lon)
    radius_f = float(radius)
    # Earth's radius in meters
    R = 6371000.0

    #Convert radius to m and increase a little
    radius_f = radius_f *1200

    # Convert latitude and longitude from degrees to radians
    lat_rad = math.radians(lat_f)
    lon_rad = math.radians(lon_f)

    # Calculate the bounding box
    d_lat = radius_f / R
    d_lon = radius_f / (R * math.cos(lat_rad))

    south = lat_f - math.degrees(d_lat)
    north = lat_f + math.degrees(d_lat)
    west = lon_f - math.degrees(d_lon)
    east = lon_f + math.degrees(d_lon)

    return north, east, south, west

    
def plot_sites(lat,long,radius, date_start, date_end, path):
    #Function to plot the locations of sites for a given map. Download an OSM shape file from: https://www.openstreetmap.org/ and save it to the specified path.

    ## This function will take the shape file you provide, the coordinates of a site a radius from the site, and a date. 
    ## The function will then find sites that report at least one data point within the dates provided.

    import datetime
    from dateutil.relativedelta import relativedelta
    import SUFO_Traffic
    import osmnx as ox
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point

    #First we have a look at the dates, if less than 1 month, just do, else split up.

    #Convert to datetime objects
    start_dt = datetime.datetime.fromisoformat(date_start)
    end_dt = datetime.datetime.fromisoformat(date_end)

    #Check end is after start
    if end_dt < start_dt:
        raise ValueError(f"End date cannot be before start date")

    n_days = (end_dt - start_dt).days

    if n_days <= 35:
        #Do it
        sensors = locate_traffic_sites(lat, long, radius,date_start,date_end,add_distance=0)

        if sensors is None:
            return print("No Traffic Sensors in radius/timeframe")
    else:
        return print("Please choose a timeframe less than 1 month")
        # # Create a list to store the entire months
        # dates_list = []
        
        # # Start with the initial month
        # current_date = start_dt.replace(day=1)  # Start from the 1st day of the month

        # while current_date <= end_dt:
        #     # Add the current month to the list
        #     dates_list.append(current_date.strftime("%Y-%m-%dT%H:%M:%S"))

        #     # Move to the next month
        #     current_date += relativedelta(months=1)  #Add one month

    
    #Now we plot

    # Define the bounding box coordinates (in this example, a small area in Sheffield)
    north, east, south, west = SUFO_Traffic.calculate_bounding_box(lat, long, radius)

    # Download the street network data for the bounding box
    graph = ox.graph_from_bbox(north, south, east, west, network_type='all')

    filename = f"sheffield_{south}_{west}_{north}_{east}.osm"
    path = "G:/My Drive/03 Semester 3/Maps/"

    # Save the street network as a shapefile (or in other formats supported by osmnx)
    ox.save_graph_shapefile(graph, filepath= path+ filename)
    
    # Load the graph within the bounding box
    graph = ox.graph_from_bbox(north, south, east, west, network_type='all', simplify=True)
    # Convert the graph to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(graph)

    # Create a GeoDataFrame for the main marker
    main_marker = gpd.GeoDataFrame(geometry=[Point(long, lat)], crs="EPSG:4326")
    main_marker = main_marker.to_crs(edges.crs)

    #ADD OTHER POINTS

    # Create a GeoDataFrame for the additional points
    additional_markers = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lat, lon in sensors.values()], 
        crs="EPSG:4326"
    )
    additional_markers = additional_markers.to_crs(edges.crs)

    # Plot the map with the specified extent, only plotting the edges
    fig, ax = plt.subplots(figsize=(10, 10))
    edges.plot(ax=ax, linewidth=1, edgecolor='black', alpha = 0.6)

    # Plot the main marker
    main_marker.plot(ax=ax, color='blue', marker='o', markersize=100, label='AQ Sensor')

    # Plot the additional markers
    additional_markers.plot(ax=ax, color='red', marker='^', markersize=100, label='Traffic Sensors', alpha = 1)

    # Set the extent of the plot
    ax.set_xlim([west, east])
    ax.set_ylim([south, north])

    plt.title(f"Traffic Sensors within {radius} km of {round(float(lat),4), round(float(long),4)}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Add a legend
    plt.legend()

    return plt

def get_sensor(site_id, start_date, end_date):
    #Function that returns a DF with the data from SUFO for a traffic sensor over a given time period
    import requests
    from datetime import datetime
    import pandas as pd

    #Convert to datetime objects
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    #Check end is after start
    if end_dt < start_dt:
        raise ValueError(f"End date cannot be before start date")

    #Add a warning for if the date is too long
    if (end_dt - start_dt).days > 35:
        raise ValueError("Large timeframe, ensure you are requesting no more than one month")

    #Custom URL
    url = "https://ufdev21.shef.ac.uk/sufobin/sufoDXT?Tfrom=" + start_date + "&Tto=" + end_date + "&byFamily=SCC_flow&bySite=" + site_id + "&freqInMin=1&qcopt=prunedata&udfnoval=-32768&udfbelow=-32769&udfabove=-32767&hrtFormat=iso8601&tabCont=rich&gdata=byPairId&src=data&op=getdata&fmt=jsonrows&output=zip&tok=generic&spatial=none"

    #Now get the data and search
    try:
        response = requests.get(url)
        response.raise_for_status()  #error if issue
        request_data = response.json()

        #Find where the flow data is
        if request_data["nBundles"] > 1:
            raise ValueError("More than one bundle")

        json_data = request_data["bundles"][0]["dataByRow"]

        #convert into a temporary df
        json_df = pd.DataFrame(json_data)
        # Modify column names to remove anything before the first "."
        json_df.columns = json_df.columns.str.split('.').str[1]
        json_df = json_df[["time","flow"]]
        return json_df
    
    except requests.exceptions.HTTPError as http_err:
            if response.status_code == 500:
                #try a different interval, if this is blank then the error is no data (return none)
                url = "https://ufdev21.shef.ac.uk/sufobin/sufoDXT?Tfrom=" + start_date + "&Tto=" + end_date + "&byFamily=SCC_flow&bySite=" + site_id + "&freqInMin=1&qcopt=prunedata&udfnoval=-32768&udfbelow=-32769&udfabove=-32767&hrtFormat=iso8601&tabCont=rich&gdata=byPairId&src=data&op=getdata&fmt=jsonrows&output=zip&tok=generic&spatial=none"
                try:
                    response = requests.get(url)
                    response.raise_for_status()  #error if issue
                    request_data = response.json()
                except ValueError:  #ValueError will occur for JSON parse failure
                    print("No data for specified parameters")
                    return None
            else:
                print(f"HTTP error occurred: {http_err}")
                return None
    except ValueError:  #ValueError will occur for JSON parse failure
        print("No data for specified parameters")
        return None

def parse_sensor(site_id,date_start,date_end,path):
    #Function that will take a large date range for a site id and create a pickle for the specified range. If the pickle already exists it will not function

    #libraries
    import datetime
    import SUFO_Traffic
    import numpy as np
    from dateutil.relativedelta import relativedelta
    import pandas as pd

    #Check for pickle presence on path
    fname = site_id + "_" + date_start[:10].replace('-', '') + "_" + date_end[:10].replace('-', '') + "_" + "flow"

    try:
        #Try and read
        pd.read_pickle(path + fname)
        return print("Pickle already exists, please load instead")
    except Exception as e:
        #Else get the data to a pickle and read that

        #Convert to datetime objects
        start_dt = datetime.datetime.fromisoformat(date_start)
        end_dt = datetime.datetime.fromisoformat(date_end)

        #Check end is after start
        if end_dt < start_dt:
            raise ValueError(f"End date cannot be before start date")

        n_days = (end_dt - start_dt).days

        #If less than 35 query
        if n_days <= 35:
            df = SUFO_Traffic.get_sensor(site_id,date_start,date_end)

            #If empty DF then return nothing
            if df is None:
                return None
                print("Blank")

            ##Check the target columns exist
            if("time" not in df.columns):
                raise KeyError(f"Column 'time' not found in data")

            
            # Convert UNIX time to datetime
            df['ISO_time'] = pd.to_datetime(df['time'], unit='s')

            #Drop UNIX time
            df = df[["ISO_time","flow"]]

            #Resample from 1 min to 30mins
            df = df.set_index('ISO_time').resample('30min').mean().reset_index()

            #Now pickle
            df.to_pickle(path + fname)

        else:
            #split into smaller chunks
            # Create a list to store the entire months
            dates_list = []
            
            # Start with the initial month
            current_date = start_dt.replace(day=1)  # Start from the 1st day of the month

            while current_date <= end_dt:
                # Add the current month to the list
                dates_list.append(current_date.strftime("%Y-%m-%dT%H:%M:%S"))

                # Move to the next month
                current_date += relativedelta(months=1)  #Add one month
            
            #Create empty dict to store output
            all_data_dict = {}

            for x in range(len(dates_list) - 1):

                #Name for the dict element
                df_name = dates_list[x].split("T")[0]
                #Get the sensor data
                print(f"{dates_list[x]} to {dates_list[x+1]}")
                df = SUFO_Traffic.get_sensor(site_id,dates_list[x],dates_list[x+1])

                #Resample from 1 min to 30min

                ##If blank return -15,000 for all values (one a day)
                if df is None:
                    # Create a date range from start_date to end_date
                    date_range = pd.date_range(start=date_start, end=date_end, freq='D')

                    # Convert the date range to UNIX timestamps
                    unix_dates = date_range.astype(np.int64) // 10**9

                    # Create a DataFrame
                    df = pd.DataFrame({
                        "time": unix_dates,
                        'ISO_time': date_range,
                        "flow": -15000
                    })

                    all_data_dict[df_name] = pd.DataFrame()

                ##Check the target columns exist
                if ("time" not in df.columns):
                    raise KeyError(f"Column 'time' not found in data")
                
                # Convert UNIX time to datetime
                df['ISO_time'] = pd.to_datetime(df['time'], unit='s')

                #Drop UNIX time
                df = df[["ISO_time","flow"]]

                #Resample from 1 min to 30mins
                df = df.set_index('ISO_time').resample('30min').mean().reset_index()

                # Take just the two columns and store in a dictionary
                all_data_dict[df_name] = df[["ISO_time","flow"]]
                #print(f"Loaded {dates_list[x]} to {dates_list[x +1]}")

            #Now we'll output a DF with them concetenated
            frames = [all_data_dict[key] for key in list(all_data_dict.keys())]
            all_data_df = pd.concat(frames)

            #Now pickle
            all_data_df.to_pickle(path + fname)

def replace_missing(data_in):
    #Function that replaces missing values in a dataset with the mean value 
    ## Takes a DF as an input

    import pandas as pd

    #First remove massive negatives (sensor error codes)
    data_in = data_in[data_in["flow"] > -1000]
    data_in['ISO_time'] = pd.to_datetime(data_in['ISO_time']) # Ensure 'iso_time' is in datetime format

    #Replace zeroes and blanks
    mean_value = data_in["flow"][data_in["flow"] != 0].mean()
    data_out = data_in
    data_out["flow"].fillna(mean_value, inplace=True)
    data_out["flow"].replace(0, mean_value, inplace=True)

    return data_out

def calculate_traffic_diff(data_in):
    #Function to calculate the percentage change in daily flow

    #Takes a DF with the mean daily traffic flow as an input, replaces missing values with the mean then estimates the change in daily traffic flow (overall daily cars/min)
    import pandas as pd
    import SUFO_Traffic
    import datetime
    from scipy import stats

    data_out = SUFO_Traffic.replace_missing(data_in) #Replace missing witht the mean

    # Extract date part and create a new column 'date'
    data_out['date'] = data_out['ISO_time'].dt.date

    #Split before and after CAZ
    # Define the cutoff date
    cutoff_date = datetime.date(2023,2,27)

    #Calculate daily means
    avg_per_day = data_out.groupby('date')['flow'].mean().reset_index()

    # Filter the DataFrame to include only rows with dates before the cutoff date
    data_pre = avg_per_day[avg_per_day['date'] < cutoff_date]
    data_pre = data_pre[data_pre['date'] >= cutoff_date -  datetime.timedelta(days=365)]

    data_post = avg_per_day[avg_per_day['date'] >= cutoff_date]
    data_post = data_post[data_post['date'] < cutoff_date + datetime.timedelta(days=365)]

    #Calcualte Daily means
    # Convert 'date' column to datetime format if necessary
    data_pre['date'] = pd.to_datetime(data_pre['date'])
    data_pre['days_since'] = data_pre['date'].dt.dayofyear

    # Convert 'date' column to datetime format if necessary
    data_post['date'] = pd.to_datetime(data_post['date'])
    data_post['days_since'] = data_post['date'].dt.dayofyear

    #Remove the index
    data_post = data_post.reset_index()
    data_pre = data_pre.reset_index()

    #Column Subset
    data_pre_sub = data_pre[["flow","days_since"]]
    data_pre_sub.columns = ["flow_pre","days_since"]


    data_post_sub = data_post[["flow","days_since"]]
    data_post_sub.columns = ["flow_post","days_since"]

    #Now merge based on the days and estimate
    merged_df = pd.merge(data_pre_sub, data_post_sub, on='days_since', how='inner')
    merged_df['diff'] = merged_df["flow_pre"] - merged_df["flow_post"]
    merged_df['pct_diff'] = ((merged_df["flow_post"] - merged_df["flow_pre"]) / merged_df["flow_pre"]) * 100
    avg_pre = merged_df['flow_pre'].mean()
    avg_post = merged_df['flow_post'].mean()

    # Perform paired sample t-test
    t_statistic, p_value = stats.ttest_rel(merged_df["flow_post"], merged_df["flow_pre"])

    return [avg_pre, avg_post, ((avg_post - avg_pre) / avg_pre * 100),t_statistic,p_value]


