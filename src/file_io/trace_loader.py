from enum import Enum
import pandas as pd


class TraceColumns(Enum):
    """Columns names for traces
    """
    UID_COLUMN = 'uid'
    START_TIME_COLUMN = 'enter_time'
    END_TIME_COLUMN = 'exit_time'
    LAT_COLUMN = 'lat'
    LNG_COLUMN = 'lng'
    DURATION = 'duration'


class TraceLoader(object):
    """ Load traces data from a variety of sources
    """

    @staticmethod
    def load_traces_from_df(data_frame,
                            uid_column='uid',
                            start_time_column='enter_time',
                            end_time_column='exit_time',
                            lat_column='lat',
                            lng_column='lng',
                            tz_data='UTC',
                            tz_local='US/Pacific',
                            ambiguous='raise'):
        """Format traces dataframe.
        Args:
            data_frame (pd.DataFrame): Unformatted dataframe.
            uid_column (str): Input uid column name.
            start_time_column (str): Input start time column name.
            end_time_column (str): Input end time column name.
            lat_column (str): Input latitude column name.
            lng_column (str): Input longitude column name.
            tz_data (str): Input time zone.
            tz_local (str): Output time zone.
            ambiguous (str or bool): See pandas.DatetimeIndex documentation
        Returns:
            (pd.DataFrame) Dataframe of traces.
        """
        if len(data_frame) < 1:
            return data_frame

        data_frame[[TraceColumns.UID_COLUMN.value,
                    TraceColumns.LAT_COLUMN.value,
                    TraceColumns.LNG_COLUMN.value]] = data_frame[
            [uid_column, lat_column, lng_column]]

        # Pandas will complain if the input time zone is not UTC
        data_frame[TraceColumns.START_TIME_COLUMN.value] = pd.DatetimeIndex(
            data_frame[start_time_column], tz=tz_data, ambiguous=ambiguous).tz_convert(tz_local)
        data_frame[TraceColumns.END_TIME_COLUMN.value] = pd.DatetimeIndex(
            data_frame[end_time_column], tz=tz_data, ambiguous=ambiguous).tz_convert(tz_local)

        return data_frame[[TraceColumns.UID_COLUMN.value,
                           TraceColumns.START_TIME_COLUMN.value,
                           TraceColumns.END_TIME_COLUMN.value,
                           TraceColumns.LAT_COLUMN.value,
                           TraceColumns.LNG_COLUMN.value]]

    @staticmethod
    def load_traces_from_csv(csv_file, **kwargs):
        """Load traces from CSV.
        Args:
            csv_file (str): CSV file name.
            **kwargs: will be passed into load_traces_from_df()
        Returns:
            (pd.DataFrame) Dataframe of traces.
        """
        data_frame = pd.read_csv(csv_file)
        return TraceLoader.load_traces_from_df(data_frame, **kwargs)