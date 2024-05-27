import matplotlib.pyplot as plt
import numpy as np
import math

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

WIDTH = 1245
HEIGHT = 825
ASPECT_RATIO = WIDTH / HEIGHT

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, image_width : int, image_height : int, method : int):
	for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
		out = render(view, gaussians, pipeline, background, image_width=image_width, image_height=image_height, method=method)
		# Release memory
		if method == 0 or method == 2:
			return out["benchmark"][0].item()
		else:
			# return (out["benchmark"][0] - out["benchmark"][3]).item()
			return (out["benchmark"][1] + out["benchmark"][2]).item()
		# rendering = out["render"]
		# gt = view.original_image[0:3, :, :]
		# torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
		# torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
	with torch.no_grad():
		gaussians = GaussianModel(dataset.sh_degree)
		scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

		bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
		background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

		args = (dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
		return args

if __name__ == "__main__":
	print("Example Usage: python benchmark.py --model_path /mnt/d/Stanford/CS348K/models/stump --source_path /mnt/d/Stanford/CS348K/360_v2/stump")
	method_names = ["Rasterizer", "Hardware-accelerated ray tracing", "CUDA ray tracing"]
	# Set up command line argument parser
	parser = ArgumentParser(description="Testing script parameters")
	model = ModelParams(parser, sentinel=True)
	pipeline = PipelineParams(parser)
	parser.add_argument("--iteration", default=-1, type=int)
	parser.add_argument("--quiet", action="store_true")
	parser.add_argument("--benchmark", default=1, type=int)
	args = get_combined_args(parser)

	# Initialize system state (RNG)
	safe_state(args.quiet)

	render_args = render_sets(model.extract(args), args.iteration, pipeline.extract(args))

	# Number of gaussians vs runtime for all three methods.
	if args.benchmark == 1:
		gaussians = render_args[4]
		P = gaussians._orig_xyz.shape[0]

		N_BINS = 25
		BIN_SIZE = int(P / N_BINS)
		x = []
		for i in range(1, P, BIN_SIZE):
			x.append(i)
		x.append(P)

		times = [[], [], []]
		for n_gaussians in x[::-1]:
			gaussians.set_n(n_gaussians)
			for method in [1, 2, 0]:
				arr = []
				for i in range(3):
					arr.append(render_set(*render_args, 596, 410, method))
				t = np.median(arr)
				times[method].append(t)
				print(method, n_gaussians, t)

		print(x)
		print(times)
	elif args.benchmark == 2:
		# Resolution (number of pixels) vs runtime for all three methods. Keep aspect ratio and scale
		x = []
		N_BINS = 25
		BIN_SIZE = int(WIDTH * HEIGHT / N_BINS)
		resolutions = []
		for i in range(1, WIDTH * HEIGHT, BIN_SIZE):
			# w / h = ASPECT_RATIO
			# w * h = i
			# ASPECT_RATIO * h * h = i
			#h = sqrt(i / ASPECT_RATIO)
			h = math.sqrt(i / ASPECT_RATIO)
			w = ASPECT_RATIO * h
			resolutions.append((round(w), round(h)))
			x.append(w * h)
		resolutions.append((WIDTH, HEIGHT))
		x.append(WIDTH * HEIGHT)
		print("Resolutions", resolutions)
		times = [[], [], []]
		for method in [1, 0, 2]:
			for r in resolutions:
				t = render_set(*render_args, r[0], r[1], method)
				times[method].append(t)
				print(method, r, t)

		print(x)
		print(times)
	else:
		# Just singular runtime
		times = [0, 0, 0]
		for method in [1, 0, 2]:
			t = render_set(*render_args, 0, 0, method)
			times[method] = t
		print("Times", times)
	# plt.title("Runtime Performances over Number of Gaussians")
	# plt.xlabel("Number of Gaussians")
	# plt.ylabel("Runtime (s)")
	# for method in [0, 1, 2]:
	# 	plt.plot(x, times[method][::-1], label=method_names[method])
	# plt.legend()
	# plt.show()

	# plt.title("Runtime Performances over Different Resolutions")
	# plt.xlabel("Number of pixels")
	# plt.ylabel("Runtime (s)")
	# for method in [0, 1, 2]:
	# 	plt.plot(x, times[method], label=method_names[method])
	# plt.legend()
	# plt.show()
