import gmplot
import time
from selenium import webdriver

gmap = gmplot.GoogleMapPlotter(50.4235915,30.4738741667, 16)

path = "/tmp/mymap.html"
driver = webdriver.Firefox()
driver.get("file://{}".format(path))

latitudes = [50.4235915,50.4240628333]
longitudes = [30.4738741667, 30.4754195]

for i in range(10):
  latitudes.append(latitudes[-1]+0.0001)
  longitudes.append(longitudes[-1]+0.0001)
  gmap.plot(latitudes, longitudes, 'cornflowerblue', edge_width=10)
  # gmap.scatter(latitudes, longitudes, 'k', marker=True)
  gmap.draw(path)
  time.sleep(1)
  driver.refresh()

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