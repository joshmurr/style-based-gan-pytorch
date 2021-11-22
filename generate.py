import argparse
import math

import torch
from torchvision import utils, transforms, io
import numpy as np
from PIL import Image

from model import StyledGenerator


@torch.no_grad()
def get_mean_style(generator, device):
	mean_style = None

	for i in range(10):
		style = generator.mean_style(torch.randn(256, 64).to(device))

		if mean_style is None:
			mean_style = style

		else:
			mean_style += style

	mean_style /= 10
	return mean_style

@torch.no_grad()
def sample(generator, step, mean_style, n_sample, device):
	image = generator(
		torch.randn(n_sample, 64).to(device),
		step=step,
		alpha=1,
		mean_style=mean_style,
		style_weight=0.7,
	)

	return image

@torch.no_grad()
def linear_interpolate(code1, code2, alpha):
	return code1 * alpha + code2 * (1 - alpha)


@torch.no_grad()
def get_interp_frames(generator, step, mean_style, num_interps):
	fps = 25
	step_size = 1.0/num_interps
	amounts = np.arange(0, 1, step_size)

	codes = torch.randn(2, 64).to(device)

	all_zs = torch.stack([linear_interpolate(codes[0], codes[1], alpha) for alpha in amounts])

	images = generator(
		all_zs,
		step=step,
		alpha=1,
		mean_style=mean_style,
		style_weight=0.7,
	)

	return images, codes

def normalize_frames(frames):
	normed = torch.zeros(*frames.size())
	for i, img in enumerate(frames):
		_min = img.min()
		_max = img.max()
		img -= _min
		img *= 255 / (_max - _min)
		normed[i] = img
	return normed

def make_gif(frames, size=(256,256), name='latent_space_traversal.gif'):
	normed_frames = normalize(frames)

	for img in normed_framed:
		img = img.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

		all_imgs.append(Image.fromarray(img).resize(size))

	all_imgs[0].save(name, save_all=True, append_images=all_imgs[1:], duration=1000/fps, loop=0)

	return images

def make_video(frames, size=(256,256), name='latent_space_traversal.mp4'):
	fps = 24
	normed_frames = normalize_frames(frames)
	io.write_video(name, normed_frames.permute(0, 2, 3, 1).cpu(), fps)


@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device):
	source_code = torch.randn(n_source, 64).to(device)
	target_code = torch.randn(n_target, 64).to(device)
	shape = 4 * 2 ** step
	alpha = 1

	images = [torch.ones(1, 3, shape, shape).to(device) * -1]

	source_image = generator(
		source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
	)
	target_image = generator(
		target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
	)

	images.append(source_image)

	for i in range(n_target):
		image = generator(
			[target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
			step=step,
			alpha=alpha,
			mean_style=mean_style,
			style_weight=0.7,
			mixing_range=(0, 1),
		)
		images.append(target_image[i].unsqueeze(0))
		images.append(image)

	images = torch.cat(images, 0)

	return images


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--size', type=int, default=256, help='size of the image')
	parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
	parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
	parser.add_argument('--state_dict', type=str, default='g_running', help='state dict')
	parser.add_argument('--output', type=str, default='images', help='image, gif or video')
	parser.add_argument('--frames', type=int, default=32, help='num frames for GIF or video')
	parser.add_argument('path', type=str, help='path to checkpoint file')

	args = parser.parse_args()

	device = 'cuda'

	generator = StyledGenerator().to(device)
	if 'train_step' in args.path:
		print('Loading from train_step')
		generator.load_state_dict(torch.load(args.path)[args.state_dict])
	else:
		generator.load_state_dict(torch.load(args.path))
	generator.eval()

	mean_style = get_mean_style(generator, device)

	step = int(math.log(args.size, 2)) - 2

	if args.output == 'gif':
		frames, _ = get_interp_frames(generator, step, mean_style, args.frames)
		img = make_gif(frames)
	elif args.output == 'video':
		frames, _ = get_interp_frames(generator, step, mean_style, args.frames)
		img = make_video(frames)
	elif args.output == 'image':
		img = sample(generator, step, mean_style, args.n_row * args.n_col, device)
		utils.save_image(img, 'sample.png', nrow=args.n_col, normalize=True, range=(-1, 1))
	else:
		print('Please choose image, gif or video for --output')

	# for j in range(20):
	# img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device)
	# utils.save_image(
	# img, f'sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
	# )
