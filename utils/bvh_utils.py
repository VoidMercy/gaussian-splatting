import time
import sys
import torch
import numpy as np

# TODO: we can implement this in CUDA later and make it faster. But, let's make it work first!

class BBox:
	def __init__(self):
		self.min = np.array([sys.float_info.max, sys.float_info.max, sys.float_info.max], dtype=float)
		self.max = np.array([sys.float_info.min, sys.float_info.min, sys.float_info.min], dtype=float)

	def enclose_point(self, p):
		self.min = np.minimum(self.min, p)
		self.max = np.maximum(self.max, p)

	def enclose_bb(self, other):
		self.min = np.minimum(self.min, other.min)
		self.max = np.maximum(self.max, other.max)

	def empty(self):
		return self.min[0] > self.max[0] or self.min[1] > self.max[1] or self.min[2] > self.max[2]

	def surface_area(self):
		if self.empty(): return 0.0
		extent = self.max - self.min
		return 2.0 * (extent[0] * extent[2] + extent[0] * extent[1] + extent[1] * extent[2])

	def __str__(self):
		return str((self.min, self.max))

	def __repr__(self):
		return str(self)

class BVHNode:
	def __init__(self, bbox, start, size):
		self.bbox = bbox
		self.start = start
		self.size = size
		self.left = None
		self.right = None

	def __str__(self):
		return str((self.bbox, self.start, self.size, self.left, self.right))

	def __repr__(self):
		return str(self)

class BVH:
	def __init__(self, aabb):
		self.aabb = aabb
		self.bvh = []
		self.primitives = np.arange(self.aabb.shape[0])

		self.construct(aabb)

	def new_node(self, bbox, start, size):
		self.bvh.append(BVHNode(bbox, start, size))
		return len(self.bvh) - 1

	def construct(self, aabb):
		# Now construct BVH on CPU (could speed up in the future by doing it in CUDA and parallelizing)

		# BVH data structure is a tree with each node stored in a 1D array
		# Each node has properties:
		# - bbox_min
		# - bbox_max
		# - left (index of left node)
		# - right (index of right node)
		# - start (index in primitive array with the primitive information)
		# - size (how many primitives under this node)

		# Primitive array data structure has the information
		# - index of gaussian

		root_bbox = BBox()
		root_bbox.enclose_point(np.min(aabb[:, :, 0], axis=0))
		root_bbox.enclose_point(np.max(aabb[:, :, 0], axis=0))
		self.new_node(root_bbox, 0, self.primitives.shape[0])

		print(root_bbox)
		print(self.bvh[0])
