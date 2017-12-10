import cv2
import numpy as np

from object_detection import CarDetector


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

  def add(self, n):
    items.append(n)


  def parking_spots(self):
    for i in range(0, len(items)-1):
      if haversine(items[i].latitude, items[i].longitude, items[i+1].latitude, items[i+1].longitude)>20:
        dot.append(center(items[i].latitude, items[i].longitude, items[i+1].latitude, items[i+1].longitude))



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

class Frame:
  timestamp = 0
  image = []
  location = []
  boxes = []
  cars = []

class FrameSequence:
  counter = 0

  def __init__(self, path):
    self.capture = cv2.VideoCapture(path.format("camera1.avi"))
    if not self.capture.isOpened():
      raise "Some nasty shit going here, niggas've stolen files"
    self.gps_map = ClosestMap(path.format("gps.txt"))

    with open(path.format("camera1.tfd")) as txt:
      self.timestamps = list(map(lambda x: int(x)/1e6, txt))


  def hasFrames(self):
    return (self.counter != len(self.timestamps))

  def getNextFrame(self):
    ret, image = self.capture.read()
    if not ret:
      raise "Wrong"

    new_frame = Frame()
    new_frame.image = image
    new_frame.timestamp = self.timestamps[self.counter]
    self.counter += 1
    new_frame.location = self.gps_map[new_frame.timestamp]

    return new_frame

  def __del__(self):
    self.capture.release()
    cv2.destroyAllWindows()


def main():
  path = "data1/8/{}"

  seq = FrameSequence(path)
  detector = CarDetector()

  width = 1280
  height = 720

  while seq.hasFrames():
    frame = seq.getNextFrame()

    # processed_image = processImage(frame.image)
    frame.cars = []
    frame.boxes = []
    (frame.boxes, processed_image) = detector.getBoxes(frame.image, width, height)
    for box in frame.boxes:
      y1, x1, y2, x2 = box
      dist = (1-(x2-x1)) ** 4
      frame.cars.append((box, dist))

    cv2.putText(processed_image, "Timestamp: {}".format(frame.timestamp), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255), 2)
    for i, line in enumerate("{}".format(frame.location).split("\n")):
      cv2.putText(processed_image, line, (10, 60+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 1)

    for box, dist in frame.cars:
      y1, x1, y2, x2 = box
      cx = int(width*(x2+x1)/2)
      cy = int(height*(y2+y1)/2)
      cv2.putText(processed_image, "{}".format(dist), (cx, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)

    cv2.imshow("frame", processed_image)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q') or key == 27:
      break

if __name__ == '__main__':
  main()