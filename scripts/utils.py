from functools import partial
import pandas as pd                        # Pandas to load and handle the data

def set_niv_label(ALS_df, time_window_days=90):
    # Create the label column
    niv_label = ALS_df['niv']

    def set_niv_label_in_row(df, ALS_df, time_window_days=90):
        # Get a list of all the timestamps in the current patient's time series
        subject_ts_list = ALS_df[ALS_df.subject_id == df.subject_id].ts
        try:
            # Try to find the timestamp of a sample that is equal or bigger than
            # the current one + the desired time window
            closest_ts = subject_ts_list[subject_ts_list >= df.ts+time_window_days].iloc[0]
        except IndexError:
            # Just use the data from the subject's last sample if there are no
            # samples in the desired time window for this subject
            closest_ts = subject_ts_list.iloc[-1]
        # Check if the patient has been on NIV anytime during the defined time window
        if closest_ts > df.ts+time_window_days:
            time_window_data = ALS_df[(ALS_df.subject_id == df.subject_id)
                                      & (ALS_df.ts < closest_ts)
                                      & (ALS_df.ts > df.ts)]
        else:
            time_window_data = ALS_df[(ALS_df.subject_id == df.subject_id)
                                      & (ALS_df.ts <= closest_ts)
                                      & (ALS_df.ts > df.ts)]
        if time_window_data.empty:
            # Just use the last NIV indication when it's the last sample in the subject's
            # time series or there are no other samples in the specified time window
            time_window_data = ALS_df[(ALS_df.subject_id == df.subject_id)
                                      & (ALS_df.ts == df.ts)]
        return time_window_data.niv.max() == 1

    # Make sure that the label setting apply method as access to the ALS dataframe
    # and uses the right time window
    set_niv_label_in_row_prt = partial(set_niv_label_in_row, 
                                       ALS_df=ALS_df,
                                       time_window_days=time_window_days)
    # Apply the row by row label setting method
    niv_label = ALS_df.apply(set_niv_label_in_row_prt, axis=1)
    return niv_label
