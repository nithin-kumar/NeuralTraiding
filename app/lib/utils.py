import calendar
from datetime import timedelta
import datetime
from datetime import datetime as dtime
class Utils(object):
	@staticmethod
	def utc_to_local(utc_dt):
	    # get integer timestamp to avoid precision lost
	    timestamp = calendar.timegm(utc_dt.timetuple())
	    local_dt = dtime.fromtimestamp(timestamp)
	    assert utc_dt.resolution >= timedelta(microseconds=1)
	    return local_dt.replace(microsecond=utc_dt.microsecond)
		