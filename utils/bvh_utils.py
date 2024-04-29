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

	def range(self):
		return self.bbox.max - self.bbox.min

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
		root_bbox.enclose_point(np.max(aabb[:, :, 1], axis=0))
		root_node_idx = self.new_node(root_bbox, 0, self.primitives.shape[0])

		BUCKET_SIZE = 8
		LEAF_NUM_PRIMITIVES = 1

		stack = [root_node_idx]
		while len(stack) > 0:
			current_node_idx = stack.pop()
			current_node = self.bvh[current_node_idx]
			start_idx = current_node.start
			end_idx = current_node.start + current_node.size

			print(f"Current node with size {current_node.size}")

			bbox_range = current_node.range()

			least_cost = sys.float_info.max
			best_axis = None
			best_bucket = None
			best_num_a = None
			best_num_b = None
			best_bbox_a = None
			best_bbox_b = None
			best_indices = None

			# For each axis, try BUCKET_SIZE buckets, and choose the axis + bucket with least cost
			primitive_bbs = self.aabb[self.primitives]
			primitive_midpoints = (primitive_bbs[:, :, 0] + primitive_bbs[:, :, 1]) / 2
			for axis in range(3):
				if bbox_range[axis] == 0:
					continue

				# For each bucket, compute the split of primitives to left and right and compute cost
				for bucket in np.arange(current_node.bbox.min[axis] + bbox_range[axis] / BUCKET_SIZE, current_node.bbox.max[axis], bbox_range[axis] / BUCKET_SIZE):
					less_than_bucket = np.where(primitive_midpoints[start_idx:end_idx, axis] < bucket)[0] + start_idx
					more_than_bucket = np.where(primitive_midpoints[start_idx:end_idx, axis] >= bucket)[0] + start_idx
					num_a = less_than_bucket.shape[0]
					num_b = more_than_bucket.shape[0]

					if num_a >= 1 and num_b >= 1:
						bbox_a_min = np.min(primitive_bbs[less_than_bucket][:, :, 0], axis=0)
						bbox_a_max = np.max(primitive_bbs[less_than_bucket][:, :, 1], axis=0)
						bbox_b_min = np.min(primitive_bbs[more_than_bucket][:, :, 0], axis=0)
						bbox_b_max = np.max(primitive_bbs[more_than_bucket][:, :, 1], axis=0)

						bbox_a = BBox()
						bbox_b = BBox()
						bbox_a.enclose_point(bbox_a_min)
						bbox_a.enclose_point(bbox_a_max)
						bbox_b.enclose_point(bbox_b_min)
						bbox_b.enclose_point(bbox_b_max)

						# Compute cost
						sn = np.linalg.norm(current_node.bbox.max - current_node.bbox.min)**2
						sa = np.linalg.norm(bbox_a.max - bbox_a.min)**2
						sb = np.linalg.norm(bbox_b.max - bbox_b.min)**2
						cost = 1.0 + sa / sn * num_a + sb / sn * num_b
						if cost < least_cost:
							least_cost = cost
							best_axis = axis
							best_bucket = bucket
							best_num_a = num_a
							best_num_b = num_b
							best_bbox_a = bbox_a
							best_bbox_b = bbox_b
							best_indices = np.concatenate((less_than_bucket, more_than_bucket), axis=0)

			if least_cost == sys.float_info.max:
				# Didn't find any good splits, just don't split
				continue

			# Rearrange primitives array for this split
			self.primitives[start_idx:end_idx] = best_indices

			node_addr_l = self.new_node(best_bbox_a, current_node.start, best_num_a)
			node_addr_r = self.new_node(best_bbox_b, current_node.start + best_num_a, best_num_b)

			if best_num_b > LEAF_NUM_PRIMITIVES: stack.append(node_addr_r)
			if best_num_a > LEAF_NUM_PRIMITIVES: stack.append(node_addr_l)

		print("Done constructing BVH!")
		print(f"Number of BVH nodes {len(self.bvh)}")