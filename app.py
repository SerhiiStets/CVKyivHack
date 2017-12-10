
import cv2
import numpy as np

class GpsPoint:
  timestamp = 0.0
  latitude = 0.0
  longitude = 0.0
  velocity = 0.0
  altitude = 0.0
  orientation = 0.0

  def __init__(self, line = ""):
    if line == "":
      return

    items = list(map(float, line.split(" ")))

    self.timestamp = items[0]/1e6
    self.latitude = items[1]
    self.longitude = items[2]
    self.velocity = items[6]
    self.altitude = items[7]
    self.orientation = items[8]

  def __str__(self):
    return "Ts: {}\nLat: {}\nLong: {}\nVel: {}\nAlt: {}\nOr: {}".format(
      self.timestamp, self.latitude, self.longitude, self.velocity, self.altitude, self.orientation)

  def interp(self, t, other):
    pt = GpsPoint()
    a = (1-t)
    pt.timestamp = a*self.timestamp + t*other.timestamp
    pt.latitude = a*self.latitude + t*other.latitude
    pt.longitude = a*self.longitude + t*other.longitude
    pt.velocity = a*self.velocity + t*other.velocity
    pt.altitude = a*self.altitude + t*other.altitude
    pt.orientation = a*self.orientation + t*other.orientation # wrong !!!

    return pt

class ParkPlaceSercher:
  items = [] 
  car_lenght = 20
  dot = []

  def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    m = 6371000* c
    return m

  def center(lan1, lat1, lan2, lat2):
    pt = GpsPoint()
    pt.latitude = (lan1 + lan2)/2
    pt.longitude = (lat1 + lat2)/2
    return pt    

  def __init__(self):
    pass

  def add(self, n):
    items.append(n)


  def parking_spots(self):
    for i in range(0, len(items)-1):
      if haversine(items[i], items[i+1])>20:
        dot.append(center(items[i], items[i+1]))




    distance_lenght = haversine()
    if distance_lenghts >= car_lenght:





class ClosestMap:
  items = {}

  def __init__(self, fname):
    with open(fname) as txt:
      print("dropping {}".format(txt.readline()))
      
      for line in txt:
        if line.strip() != "":
          item = GpsPoint(line)
          self.items[item.timestamp] = item
    print("Got {} data points".format(len(self.items)))

  def __getitem__(self, timestamp):
    prev_key, prev_el = next(iter(self.items.items()))

    index = 0
    for k, v in self.items.items():
      if k > timestamp:
        dt = k - prev_key
        if dt != 0:
          return prev_el.interp((timestamp-prev_key)/dt, v)
        else:
          return prev_el
      prev_el = v
      index += 1



def processImage(frame):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  result = gray

  edges = cv2.Canny(gray, 100, 200)

  return edges

def main():
  path = "data1/2/{}"

  gps_map = ClosestMap(path.format("gps.txt"))

  capture = cv2.VideoCapture(path.format("camera1.avi"))

  with open(path.format("camera1.tfd")) as txt:
    # print("dropping {}".format(txt.readline()))
    # print("dropping {}".format(txt.readline()))

    while capture.isOpened():
      ret, frame = capture.read()
      if ret:
        width = 1280
        height = 720

        timestamp = int(txt.readline())/1e6

        processed_frame = processImage(frame)
        location = gps_map[timestamp]

        cv2.putText(processed_frame, "Timestamp: {}".format(timestamp), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255), 2)

        for i, line in enumerate("{}".format(location).split("\n")):
          cv2.putText(processed_frame, line, (10, 60+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 1)

        cv2.imshow("frame", processed_frame)
        key = cv2.waitKey(32) & 0xff
        if key == ord('q') or key == 27:
          break

      else:
        break

  capture.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
