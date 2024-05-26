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
	parser.add_argument("--method", default=1, type=int)
	args = get_combined_args(parser)

	# Initialize system state (RNG)
	safe_state(args.quiet)

	render_args = render_sets(model.extract(args), args.iteration, pipeline.extract(args))

	# Number of gaussians vs runtime for all three methods.
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
				arr.append(render_set(*render_args, 600, 400, method))
			t = np.median(arr)
			times[method].append(t)
			print(method, n_gaussians, t)

	print(times)
	plt.title("Runtime Performances over Number of Gaussians")
	plt.xlabel("Number of Gaussians")
	plt.ylabel("Runtime (s)")
	for method in [0, 1, 2]:
		plt.plot(x, times[method][::-1], label=method_names[method])
	plt.legend()
	plt.show()

	exit()

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
	# times = [[], [], []]
	# for method in [1, 0, 2]:
	# 	y = []


	# 	for r in resolutions:
	# 		t = render_set(*render_args, r[0], r[1], method)
	# 		times[method].append(t)
	# 		y.append(t)
	# 		print(method, r, t)

	times = [[0.10785049945116043, 0.1049313023686409, 0.10213159769773483, 0.1030023992061615, 0.10079299658536911, 0.10204560309648514, 0.10140909999608994, 0.10853700339794159, 0.10233909636735916, 0.10312890261411667, 0.10939919948577881, 0.10955680161714554, 0.1045243963599205, 0.10501989722251892, 0.10531820356845856, 0.10595300048589706, 0.106706902384758, 0.10635799914598465, 0.10707800090312958, 0.10657750070095062, 0.10776489973068237, 0.1082758978009224, 0.1099231019616127, 0.10941629856824875, 0.10946729779243469, 0.10908240079879761], [0.026702599599957466, 0.028108401224017143, 0.03968549892306328, 0.050963498651981354, 0.06300970166921616, 0.08603189885616302, 0.09589649736881256, 0.10904889553785324, 0.11695409566164017, 0.1320198029279709, 0.13898520171642303, 0.14716899394989014, 0.1585713028907776, 0.1691761016845703, 0.1761419028043747, 0.1576344072818756, 0.16675539314746857, 0.17511430382728577, 0.18155008554458618, 0.19038230180740356, 0.19685040414333344, 0.20464679598808289, 0.2221003919839859, 0.23252540826797485, 0.24552109837532043, 0.2565454840660095], [0.11788590252399445, 0.2254820019006729, 0.33644360303878784, 0.4408116042613983, 0.5427958965301514, 0.6409103870391846, 0.7380824089050293, 0.832411527633667, 0.9235827922821045, 1.0157628059387207, 1.1058189868927002, 1.186479926109314, 1.2854548692703247, 1.3716466426849365, 1.4537259340286255, 1.5459749698638916, 1.6295570135116577, 1.714815616607666, 1.8027230501174927, 1.8881235122680664, 1.9737039804458618, 2.0623831748962402, 2.1429829597473145, 2.220365285873413, 2.306436777114868, 2.386662483215332]]
	print(times)

	plt.title("Runtime Performances over Different Resolutions")
	plt.xlabel("Number of pixels")
	plt.ylabel("Runtime (s)")
	for method in [0, 1, 2]:
		plt.plot(x, times[method], label=method_names[method])
	plt.legend()
	plt.show()
