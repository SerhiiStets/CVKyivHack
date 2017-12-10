import gmplot
import time
from selenium import webdriver
import time, threading


class MapCreator:
  items_track_x = []
  items_track_y = []
  items_markers_x = []
  items_markers_y = []

  free_places_x = []
  free_places_y = []

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

  def add_free_dot(self, x, y):
    self.free_places_x.append(x)
    self.free_places_y.append(y)


  def plot(self):
    gmap = gmplot.GoogleMapPlotter(self.x, self.y, 18)
    gmap.plot(self.items_track_x, self.items_track_y, 'cornflowerblue', edge_width=5)
    # gmap.s12catter(self.items_markers_x, self.items_markers_y, '#3B0B39', size=5, marker=False)
    gmap.scatter(self.free_places_x, self.free_places_y, 'k', marker=True)
    gmap.heatmap(self.items_markers_x, self.items_markers_y)
    gmap.draw(self.path)
    self.driver.refresh()
    #print(self.items_track_x)
    if not self.done:
      threading.Timer(1, self.plot).start()
    else:
      pass
      # time.sleep(3)
      # self.driver.close()

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
