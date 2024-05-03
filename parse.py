import numpy as np
from PIL import Image

a = open("log", "r").read().strip().split("\n")

arr = np.zeros((1245, 825))

for i in a:
	if "Total intersections" not in i: continue
	x = int(i.split("(")[1].split(",")[0])
	y = int(i.split(", ")[1].split(")")[0])
	n = int(i.split(": ")[1].strip())
	arr[x, y] = n

print(np.max(arr))
print(np.min(arr))

arr = arr.astype(float)

arr = (arr - np.min(arr)) / np.max(arr)
arr = arr.T * 255.0
arr = arr.astype(np.uint8)

im = Image.fromarray(arr)
im.save("map.png")

arr = np.zeros((1245, 825))

for i in a:
	if not i.startswith("Pix"): continue
	x = int(i.split("(")[1].split(",")[0])
	y = int(i.split(", ")[1].split(")")[0])
	n = int(i.split(": ")[1].strip())
	arr[x, y] = n

print(np.max(arr))
print(np.min(arr))

arr = arr.astype(float)

arr = (arr - np.min(arr)) / np.max(arr)
arr = arr.T * 255.0
arr = arr.astype(np.uint8)

im = Image.fromarray(arr)
im.save("map2.png")
