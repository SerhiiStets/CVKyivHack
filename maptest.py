import gmplot
import time
from selenium import webdriver
import time, threading
from googlemaps import GoogleMaps

class MapCreator:
  items_track_x = []
  items_track_y = []
  items_markers_x = []
  items_markers_y = []
  x = 0
  y = 0
  path = "/tmp/mymap.html"
  done = False

  def __init__(self):   
    
    with open(self.path, "w") as f:
      f.write("map")

    self.driver = webdriver.Firefox()
    self.driver.get("file://{}".format(self.path))
    self.driver.set_window_size(480, 480)
    self.driver.set_window_position(0, 768-480)
    threading.Timer(1, self.plot).start()


  def add_track_dot(self, x, y):
    self.items_track_x.append(x)
    self.items_track_y.append(y)
    self.x = x
    self.y = y

  def add_marker_dot(self, x, y):
    self.items_markers_x.append(x)
    self.items_markers_y.append(y)


  def plot(self):
    gmap = gmplot.GoogleMapPlotter(self.x, self.y, 18)
    gmap.plot(self.items_track_x, self.items_track_y, 'cornflowerblue', edge_width=10)
    gmap.scatter(self.items_markers_x, self.items_markers_y, '#3B0B39', size=5, marker=False)
    gmap.draw(self.path)
    self.driver.refresh()
    #print(self.items_track_x)
    if not self.done:
      threading.Timer(1, self.plot).start()



class Tracer:
  def __init__(self):
    gmaps = GoogleMaps(api_key)

  def find_destination(self, x1, y1, x2, y2):
    destination = gmaps.latlng_to_address(x2, y2)
    adress = gmaps.latlng_to_address(x1, y1)
    directions = gmaps.directions(address, destination) 
    print (directions['Directions']['Distance']['meters'])
"""
def main():
  x = [50.4235915,50.4240628333]
  y = [30.4738741667, 30.4754195]
  map_creator = MapCreator(x, y)

  #map_creator.createTrack(x, y)
if __name__ == "__main__":
  main()
"""

  # gmap.scatter(latitudes, longitudes, '#3B0B39', size=40, marker=False)
  # gmap.scatter(more_lats, more_lngs, '#3B0B39', size=40, marker=False)
# gmap.heatmap(heat_lats, heat_lngs)

# import time
# import urllib
# import urllib2

# x=raw_input("Enter the URL")
# refreshrate=raw_input("Enter the number of seconds")
# refreshrate=int(refreshrate)
# driver = webdriver.Firefox()
# driver.get("http://"+x)
# while True:
#     time.sleep(refreshrate)
#     driver.refresh()
