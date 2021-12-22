import argparse
import os
import jax
import wandb
import training


def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--work_dir', type=str, default='logging', help='Directory for logging and checkpoints.')
    parser.add_argument('--data_dir', type=str, default='datasets/Sketches', help='Directory of the dataset.')
    parser.add_argument('--project', type=str, default='few-shot-adapt', help='Name of this project.')
    parser.add_argument('--name', type=str, default='default', help='Name of this experiment.')
    parser.add_argument('--group', type=str, default='default', help='Group name of this experiment (for Weights&Biases).')
    parser.add_argument('--source_ckpt_path', type=str, default='ffhq_256x256.pickle', help='Path to the checkpoint of the source model.')
    # Training
    parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--num_prefetch', type=int, default=2, help='Number of prefetched examples for the data pipeline.')
    parser.add_argument('--resolution', type=int, default=256, help='Image resolution. Must be a multiple of 2.')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training.')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--subspace_std', type=float, default=0.1, help='Std for sampling z_latents around the anchor points.')
    parser.add_argument('--subspace_freq', type=int, default=4, help='Frequency for sampling z_latents from anchor regions.')
    # Generator
    parser.add_argument('--fmap_base', type=int, default=16384, help='Overall multiplier for the number of feature maps.')
    # Discriminator
    parser.add_argument('--mbstd_group_size', type=int, help='Group size for the minibatch standard deviation layer, None = entire minibatch.')
    # Exponentially Moving Average of Generator Weights
    parser.add_argument('--ema_kimg', type=float, default=20.0, help='Controls the ema of the generator weights (larger value -> larger beta).')
    # Losses
    parser.add_argument('--pl_decay', type=float, default=0.01, help='Exponentially decay for mean of path length (Path length regul).')
    parser.add_argument('--pl_weight', type=float, default=2, help='Weight for path length regularization.')
    parser.add_argument('--kl_weight', type=float, default=1000.0, help='Weight for distance consistency loss.')
    # Regularization
    parser.add_argument('--mixing_prob', type=float, default=0.9, help='Probability for style mixing.')
    parser.add_argument('--G_reg_interval', type=int, default=4, help='How often to perform regularization for G.')
    parser.add_argument('--D_reg_interval', type=int, default=16, help='How often to perform regularization for D.')
    parser.add_argument('--r1_gamma', type=float, default=10.0, help='Weight for R1 regularization.')
    # Model
    parser.add_argument('--c_dim', type=int, default=0, help='Conditioning label (C) dimensionality, 0 = no label.')
    # Logging
    parser.add_argument('--wandb', action='store_true', help='Log to Weights&bBiases.')
    parser.add_argument('--log_every', type=int, default=50, help='Log every log_every steps.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save every save_every steps. Will be ignored if FID evaluation is enabled.')
    # FID
    parser.add_argument('--eval_fid_every', type=int, default=1000, help='Compute FID score every eval_fid_every steps.')
    parser.add_argument('--num_fid_images', type=int, default=10000, help='Number of images to use for FID computation.')
    parser.add_argument('--disable_fid', action='store_true', help='Disable FID evaluation.')

    args = parser.parse_args()
    
    if jax.process_index() == 0:
        args.ckpt_dir = os.path.join(args.work_dir, args.group, args.name, 'checkpoints')
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

        if args.wandb:
            wandb.init(project=args.project,
                       group=args.group,
                       config=args,
                       name=args.name,
                       dir=os.path.join(args.work_dir, args.group, args.name))

    training.train_and_evaluate(args)
    

if __name__ == '__main__':
    main()

