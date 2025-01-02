
from movementpredictor.config import ModelConfig
from movementpredictor.data.trackedobjectposition import TrackedObjectPosition
from datetime import datetime
import psycopg2
import logging

log = logging.getLogger(__name__)

def getTrackedBaseData(path) -> list[TrackedObjectPosition]:
    try:
        with open(path, 'rb') as file:  # binary data
            ...
    except Exception as e:
        log.error(e)

    #dm = DataManager()
    #dm.connectToDB()
    #trackedObjects = dm.getTracks()
    #dm.close()
    #return trackedObjects


class DataManager:

    log = logging.getLogger(__name__)
    isConnected = False
    conn = None


    def __init__(self) -> None:
        self.config = ModelConfig.get_instance()
        self.log.info("Fetching data from database {}, table {}", 
                      self.config.database_url, self.config.database_table)
        

    def connectToDB(self) -> None:
        try:
            conn_params = {
                "user": self.config.user,
                "password": self.config.password,
                "sslmode": "require"
            }
            self.conn = psycopg2.connect(self.config.database_url, **conn_params)
            #self.conn = psycopg2.connect("postgresql://" + conn_params.get("user") + ":"
            #                             + conn_params.get("passwort") + "@sae:100.70.113.34:30001/")
            #self.config.database_url = "postgresql://sae:sae@100.70.113.34:30001/sae?sslmode=require"
            #self.conn = psycopg2.connect(self.config.database_url)
            self.isConnected = True

        except psycopg2.Error as e:
            self.log.error("Error connecting to PostgreSQL database:", e)
        

    def get_timestamp(self, input_str) -> datetime:
        date_str, time_str = input_str.split("-")
        day, month, year = map(int, date_str.split("."))
        hour, minute = map(int, time_str.split(":"))
        return datetime(year, month, day, hour, minute)


    def getTracks(self) -> list[TrackedObjectPosition]:
        result = []

        timestamp_start = self.get_timestamp(self.config.start_time)
        timestamp_end = self.get_timestamp(self.config.end_time)

        query = f"""
            SELECT capture_ts, object_id, class_id, point((min_x + max_x)/2, (min_y + max_y)/2) AS center_point
            FROM public.{self.config.database_table}
            WHERE class_id = 2
            AND camera_id = '{self.config.camera_id}'
            AND capture_ts >= '{timestamp_start}'
            AND capture_ts <= '{timestamp_end}'
            ORDER BY capture_ts
        """

        try:
            cursor = self.conn.cursor()
            cursor.execute(query)

            for row in cursor:
                track = TrackedObjectPosition()
                track.set_capture_ts(row[0])
                track.set_uuid(row[1])
                track.set_class_id(row[2])
                parts = row[3].strip('()').split(',')
                center = tuple(float(part) for part in parts)
                center_x, center_y = float(center[0]), float(center[1])
                track.set_center((center_x, center_y))
                result.append(track)

            cursor.close()
            
            return result

        except psycopg2.Error as e:
            self.log.error("Can't execute query:", e)

        return result

    def close(self) -> None:
        try:
            self.conn.close()
        except psycopg2.Error as e:
            self.log.error("Can't close db connection:", e)
        