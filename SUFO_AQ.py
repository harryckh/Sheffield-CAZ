#SUFO_AQ

#Series of functions to aid in the retrieval of data from SUFO.

def get_sensor(site_id:str, start:str, end:str):
    # Function that will call the API to get data for the specified ID and time frame. Ensure that VPN is on, or connected to eduroam.

    ## Function connects to the API by creating a request from the input parameters. By default the data will load at the lowest possible frequency.
    ## The JSON is loaded and converted to a DF with all columns.
    import requests
    import pandas as pd
    from datetime import datetime

    #Check function input types (expecting strings)
    if not isinstance(site_id,str):
        raise ValueError(f"Expected 'site_id' to be string, but received {type(site_id).__name__}")
    if not isinstance(start,str):
        raise ValueError(f"Expected 'start' to be string, but received {type(start).__name__}")
    if not isinstance(end,str):
        raise ValueError(f"Expected 'end' to be string, but received {type(end).__name__}")

    #Convert to datetime objects
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    #Check end is after start
    if end_dt < start_dt:
        raise ValueError(f"End date cannot be before start date")

    #Add a warning for if the date is too long
    if (end_dt - start_dt).days > 35:
        raise ValueError("Large timeframe, ensure you are requesting no more than one month")
    
    # If the site is one of the DEFRA ones we retreive a little differently
    if (site_id == "UKA00575" or site_id == "UKA00181" or site_id ==  "UKA00622"):
        url = "https://ufdev21.shef.ac.uk/sufobin/sufoDXT?Tfrom=" + start + "&Tto=" + end + "&byFamily=defra&freqInMin=1&qcopt=prunedata&udfnoval=-32768&udfbelow=-32769&udfabove=-32767&hrtFormat=iso8601&tabCont=rich&gdata=byPairId&src=data&op=getdata&fmt=jsonrows&output=zip&tok=generic"

        #If there is no data, then the request responds with a HTML, so .json() will fail, use try:except to catch this
        try:
            response = requests.get(url)
            response.raise_for_status()  #error if issue
            request_data = response.json()
            
            #Create a dict that will dynamically access the correct site
            defra_dict = {request_data["bundles"][0]["identity"]["site.id"]:0,
                    request_data["bundles"][1]["identity"]["site.id"]:1,
                    request_data["bundles"][2]["identity"]["site.id"]:2}

            #Now parse the JSON and convert to DF
            #Take the data from the nested element
            #Check the number of bundles 

            json_data = request_data["bundles"][defra_dict[site_id]]["dataByRow"]

            #convert into a temporary df
            json_df = pd.DataFrame(json_data)
            # Modify column names to remove anything before the first "."
            json_df.columns = json_df.columns.str.split('.').str[1]
            
            return json_df

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return None
        except ValueError:  #ValueError will occur for JSON parse failure
            print("No data for specified parameters")
            return None
    else:
        #Otherwise execute normally
        #Create the URL
        url = "https://ufdev21.shef.ac.uk/sufobin/sufoDXT?Tfrom=" + start + "&Tto=" + end +"&bySite=" + site_id + "&freqInMin=30&qcopt=prunedata&udfnoval=-32768&udfbelow=-32769&udfabove=-32767&hrtFormat=iso8601&tabCont=rich&gdata=byPairId&src=data&op=getdata&fmt=jsonrows&output=zip&tok=generic&spatial=none"

        # We sometimes get an internal error caused by the interval being set to 30, the error happens when the sensor is usually per minute, there is no data and
        # the server tries to calculate what the 30min data should be, since there is no data to calculate with we get an internal server error. In this case, check
        # if data gets returned at a 1min interval, if that data is blank, output blank, else give the server error

        #If there is no data, then the request responds with a HTML, so .json() will fail, use try:except to catch this
        try:
            response = requests.get(url)
            response.raise_for_status()  #error if issue
            request_data = response.json()

            #Now parse the JSON and convert to DF
            #Take the data from the nested element
            #Check the number of bundles 
            if request_data["nBundles"] != 1:
                raise ValueError("More than one sensor in data")
            
            json_data = request_data["bundles"][0]["dataByRow"]

            #convert into a temporary df
            json_df = pd.DataFrame(json_data)
            # Modify column names to remove anything before the first "."
            json_df.columns = json_df.columns.str.split('.').str[1]

            return json_df

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 500:
                #try a different interval, if this is blank then the error is no data (return none)
                url = "https://ufdev21.shef.ac.uk/sufobin/sufoDXT?Tfrom=" + start + "&Tto=" + end +"&bySite=" + site_id + "&freqInMin=1&qcopt=prunedata&udfnoval=-32768&udfbelow=-32769&udfabove=-32767&hrtFormat=iso8601&tabCont=rich&gdata=byPairId&src=data&op=getdata&fmt=jsonrows&output=zip&tok=generic&spatial=none"
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
    

def parse_sensor(site_id:str,date_start:str, date_end:str,time_col:str,pollutant:str,path:str,file_type = "pkl", moving_avg:int = 0):
    #Function that accesses the data and saves a pickled pandas DF with the data. 

    ## This function takes two elements from the dates list (one start, one end) and loops through the list. Data is retrieved using
    ## Get sensor, then parsed to only include the desired column and timestamp. Data is then pickled to the specified folder
    ## where it can be read into different scripts.
    ## Moving average is an optional parameter, the defualt value is 0 and no column will be added, anything above will give you data
    ## By defualt the file type is a pickle, but you can also get a csv, set file_type to "csv"
    import datetime
    from dateutil.relativedelta import relativedelta
    import SUFO_AQ
    import pandas as pd
    import numpy as np

    #Check the input types 
    if not isinstance(site_id,str):
        raise ValueError(f"Expected 'site_id' to be string, but received {type(site_id).__name__}")
    if not isinstance(date_start,str):
        raise ValueError(f"Expected 'date_start' to be string, but received {type(date_start).__name__}")
    if not isinstance(date_end,str):
        raise ValueError(f"Expected 'date_end' to be string, but received {type(date_end).__name__}")
    if not isinstance(time_col,str):
        raise ValueError(f"Expected 'time_col' to be string, but received {type(time_col).__name__}")
    if not isinstance(pollutant,str):
        raise ValueError(f"Expected 'pollutant' to be string, but received {type(pollutant).__name__}")
    if not isinstance(path,str):
        raise ValueError(f"Expected 'path' to be string, but received {type(path).__name__}")
    if not isinstance(moving_avg,int):
        raise ValueError(f"Expected 'moving_avg' to be an integer, but received {type(moving_avg).__name__}")

    #Convert to datetime objects
    start_dt = datetime.datetime.fromisoformat(date_start)
    end_dt = datetime.datetime.fromisoformat(date_end)

    #Check end is after start
    if end_dt < start_dt:
        raise ValueError(f"End date cannot be before start date")

    n_days = (end_dt - start_dt).days

    #If less than 35, we don't need the dict and the function can just pickle straight away
    if n_days <= 35:
        df = SUFO_AQ.get_sensor(site_id,date_start,date_end)

        #If empty DF then return nothing
        if df is None:
            return None

        ##Check the target columns exist
        if (time_col not in df.columns):
            raise KeyError(f"Column {time_col} not found in data")

        if (pollutant not in df.columns):
            raise KeyError(f"Column {pollutant} not found in data")
        
        #Add ISO time, converting the UNIX column specified
        df['ISO_time'] = df[time_col].apply(lambda x: datetime.datetime.utcfromtimestamp(x).isoformat())

        #Convert target column to numeric
        df[pollutant] = df[pollutant].astype('float64')

        #If the option is selected, add a column with the moving average
        if moving_avg > 0:
            ma_col = pollutant + ".MA." + str(moving_avg)
            df[ma_col] = df[pollutant].rolling(window = moving_avg).mean()
            all_data_df= df[[time_col, "ISO_time",pollutant,ma_col]]
        else:
            all_data_df= df[[time_col, "ISO_time",pollutant]]
    else:
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

        print(f"Querying Data for {site_id}")

        for x in range(len(dates_list) - 1):

            #Name for the dict element
            df_name = dates_list[x].split("T")[0]
            #Get the sensor data
            df = SUFO_AQ.get_sensor(site_id,dates_list[x],dates_list[x+1])

            ##If blank return -15,000 for all values (one a day)
            if df is None:
                # Create a date range from start_date to end_date
                date_range = pd.date_range(start=date_start, end=date_end, freq='D')

                # Convert the date range to UNIX timestamps
                unix_dates = date_range.astype(np.int64) // 10**9

                # Create a DataFrame
                df = pd.DataFrame({
                    time_col: unix_dates,
                    'ISO_time': date_range,
                    pollutant: -15000
                })

                all_data_dict[df_name] = pd.DataFrame()

            ##Check the target columns exist
            if (time_col not in df.columns):
                raise KeyError(f"Column {time_col} not found in data")

            if (pollutant not in df.columns):
                raise KeyError(f"Column {pollutant} not found in data")
            
            #Add ISO time, converting the UNIX column specified
            df['ISO_time'] = df[time_col].apply(lambda x: datetime.datetime.utcfromtimestamp(x).isoformat())

            #Convert target column to numeric
            df[pollutant] = df[pollutant].astype('float64')

            # Take just the two columns and store in a dictionary
            
            #If the option is selected, add a column with the moving average
            if moving_avg > 0:
                ma_col = pollutant + ".MA." + str(moving_avg)
                df[ma_col] = df[pollutant].rolling(window = moving_avg).mean()
                all_data_dict[df_name] = df[[time_col, "ISO_time",pollutant,ma_col]]
            else:
                all_data_dict[df_name] = df[[time_col, "ISO_time",pollutant]]

            #print(f"Loaded {dates_list[x]} to {dates_list[x +1]}")

        #Now we'll output a DF with them concetenated
        frames = [all_data_dict[key] for key in list(all_data_dict.keys())]
        all_data_df = pd.concat(frames)

    #Finally save

    fname = site_id + "_" + date_start[:10].replace('-', '') + "_" + date_end[:10].replace('-', '') + "_" + pollutant
    if file_type == "pkl":
        all_data_df.to_pickle(path + fname)
    elif file_type == "csv":
        all_data_df.to_csv(path + fname)
    else:
        return TypeError("File type not supported")

def qual_calc(site_id:str,date_start:str,date_end:str,pollutant:str,path:str):
    # Function that will evaluate the data quality for a given sensor, pollutant and time period.

    
    ## Data quality is expressed as the number of days with no data points, this is more relevant to long-term trends.
    ## The function takes in the site, date range and pollutant. The path is also required to either retrieve a pickle
    ## or to retreive data from SUFO and store as a pickle. 
    #Packages for function
    import SUFO_AQ
    import pandas as pd
    import datetime

    #convert dates
    #Convert to datetime objects
    start_dt = datetime.datetime.fromisoformat(date_start)
    end_dt = datetime.datetime.fromisoformat(date_end)

    #Get the data for this one site over the time period
    ## Check if we have it pickled already

    fname = site_id + "_" + date_start[:10].replace('-', '') + "_" + date_end[:10].replace('-', '') + "_" + pollutant

    try:
        #Try and read
        qual_eval = pd.read_pickle(path + fname)
    except Exception as e:
        #Else get the data to a pickle and read that
        SUFO_AQ.parse_sensor(site_id,date_start, date_end,"time",pollutant,path)
        qual_eval = pd.read_pickle(path + fname)

    #Remove big negative values (these are QC flags)
    qual_eval = qual_eval[qual_eval[pollutant] > -1000]
    #Make sure in datetime
    qual_eval['ISO_time'] = pd.to_datetime(qual_eval['ISO_time'])

    #Group by day and count
    points_per_day = qual_eval.groupby(qual_eval['ISO_time'].dt.date).size()
    # Create a date range that covers the entire period of the data
    date_range = pd.date_range(start = datetime.datetime.strptime("2022-02-01", "%Y-%m-%d"), end=datetime.datetime.strptime("2024-02-01", "%Y-%m-%d"), freq='D')
    # Reindex to include missing dates, filling missing values with 0
    points_per_day = points_per_day.reindex(date_range, fill_value=0)

    n_missing = (points_per_day == 0).sum()
    earliest_date = qual_eval['ISO_time'].min()
    latest_date = qual_eval['ISO_time'].max()

    return [n_missing,earliest_date,latest_date]
    
def sites_qual_eval(sites_list:list, start_date:str,end_date:str,pollutant:str,path:str):
    # Function that calculautes the data quality for a list of sites

    ## Function inputs a list of sites (can be only one site but must be in a list). The function will iterate over the list and return a DF with the metrics
    ## Function operates by calling qual_calc() for each site over the time period, then concetentating the metrics into a dict, then a DF to be returned.
    ## All the sites in the list must be reporting the same pollutant (e.g. All NO2)

    import SUFO_AQ
    import pandas as pd
    #Create blank output
    out_dict = {}

    #Loop through list
    for site in sites_list:
        out_list = qual_calc(site,start_date,end_date,pollutant,path)

        out_dict[site] = {
            'n_missing': out_list[0],
            'earliest': out_list[1],
            'latest': out_list[2]
        }
    
    # Convert the dictionary to a DataFrame
    out_df = pd.DataFrame.from_dict(out_dict, orient='index').reset_index()

    # Rename the columns
    out_df = out_df.rename(columns={'index': 'site_id', 'n_missing': 'n_missing', 'earliest': 'earliest', 'latest': 'latest'})

    #Return the df
    return out_df

def calculate_aq_diff(site_id, pollutant):
    #Function that estimates the difference in air quality one year either side of the CAZ

    import pandas as pd

    data_in = pd.read_pickle("G:/My Drive/03 Semester 3/SUFO Data/Pickles/" + site_id + "_20220201_20240601_" + pollutant)
    data_in = data_in[data_in[pollutant] > -1000]
    data_in['ISO_time'] = pd.to_datetime(data_in['ISO_time']) # Ensure 'iso_time' is in datetime format

    #Mean Daily concentration
    daily_mean = data_in.set_index('ISO_time').resample('D').mean().reset_index()
    #Split before and after CAZ

    # Define the cutoff date
    cutoff_date = pd.Timestamp('2023-02-27')

    # Filter the DataFrame to include only rows with dates before the cutoff date
    data_pre = daily_mean[daily_mean['ISO_time'] < cutoff_date]
    data_pre = data_pre[data_pre['ISO_time'] >= cutoff_date - pd.DateOffset(years=1)]

    data_post = daily_mean[daily_mean['ISO_time'] >= cutoff_date]
    data_post = data_post[data_post['ISO_time'] < cutoff_date + pd.DateOffset(years=1)]


    # Days since year start
    data_pre.set_index('ISO_time', inplace=True)
    data_pre['days_since'] = data_pre.index.dayofyear

    data_post.set_index('ISO_time', inplace=True)
    data_post['days_since'] = data_post.index.dayofyear

    #Remove the index
    data_post = data_post.reset_index()
    data_pre = data_pre.reset_index()

    #Column Subset
    data_pre_sub = data_pre[[pollutant,"days_since"]]
    pre_name = pollutant + "_pre"
    data_pre_sub.columns = [pre_name,"days_since"]


    data_post_sub = data_post[[pollutant,"days_since"]]
    post_name = pollutant + "_post"
    data_post_sub.columns = [post_name,"days_since"]

    #Now merge based on the days and estimate
    merged_df = pd.merge(data_pre_sub, data_post_sub, on='days_since', how='inner')
    merged_df['diff'] = merged_df[pre_name] - merged_df[post_name]
    merged_df['pct_diff'] = ((merged_df[pre_name] - merged_df[post_name]) / merged_df[pre_name]) * 100
    return merged_df['pct_diff'].mean()

def locate_AQ_sites(start_date:str,end_date:str,lat:str,long:str,radius:str):
    #Function that allows you to locate AQ sites within a given radius of some coords and a given timeframe.

    #The function will call the SUFO API to locate sites within the specified radius. For now this is limited to one month
    import requests
    import datetime
    from dateutil.relativedelta import relativedelta
    import pandas as pd

    #First define the function that will retreive the data
    def find_site(start_date, end_date,lat,long,radius):
        aq_sites = []
        url = "https://ufdev21.shef.ac.uk/sufobin/sufoDXT?Tfrom=" + start_date + "&Tto=" + end_date + "&midLon=" + long + "&midLat=" + lat + "&zRad=" + radius + "&bySitesSet=PeakPark,defra_Sheffield,luftdaten,ufloSites&freqInMin=1&qcopt=prunedata&udfnoval=-32768&udfbelow=-32769&udfabove=-32767&hrtFormat=iso8601&tabCont=rich&gdata=byPairId&src=data&op=getdata&fmt=jsonrows&output=zip&tok=generic&spatial=none"
        response = requests.get(url)
        response.raise_for_status()  #error if issue
        request_data = response.json()

        for x in range(0, request_data["nBundles"]):
            #Get the site name and coordinates
            site_id = request_data["bundles"][x]["identity"]["site.id"]
            coords = request_data["bundles"][x]["location"]["latitude"],request_data["bundles"][x]["location"]["longitude"]

            #Now get the reported pollutants (and other data)
            out_list = [item.split(".")[1] for item in request_data["bundles"][x]["outKeys"]]
            aq_sites.append([site_id,coords,out_list])
        
        return aq_sites
        
    #Convert to datetime objects
    start_dt = datetime.datetime.fromisoformat(start_date)
    end_dt = datetime.datetime.fromisoformat(end_date)

    #Check end is after start
    if end_dt < start_dt:
        raise ValueError(f"End date cannot be before start date")

    n_days = (end_dt - start_dt).days

    if n_days <= 35:
        return find_site(start_date,end_date,lat,long,radius)
    else:
        raise ValueError("Please reduce timeframe to fewer than 35 days")


