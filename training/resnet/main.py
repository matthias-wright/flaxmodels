import argparse
import os
import jax
import wandb
import training


def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--work_dir', type=str, default='/export/scratch/mwright/projects/misc/imagenette', help='Directory for logging and checkpoints.')
    parser.add_argument('--data_dir', type=str, default='/export/data/mwright/tensorflow_datasets', help='Directory for storing data.')
    parser.add_argument('--name', type=str, default='test', help='Name of this experiment.')
    parser.add_argument('--group', type=str, default='default', help='Group name of this experiment.')
    # Training
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='Architecture.')
    parser.add_argument('--resume', action='store_true', help='Resume training from best checkpoint.')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--warmup_epochs', type=int, default=9, help='Number of warmup epochs with lower learning rate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes.')
    parser.add_argument('--img_size', type=int, default=224, help='Image size.')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training.')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed.')
    # Logging
    parser.add_argument('--wandb', action='store_true', help='Log to Weights&bBiases.')
    parser.add_argument('--log_every', type=int, default=100, help='Log every log_every steps.')
    args = parser.parse_args()
    
    if jax.process_index() == 0:
        args.ckpt_dir = os.path.join(args.work_dir, args.group, args.name, 'checkpoints')
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

        if args.wandb:
            wandb.init(entity='matthias-wright',
                       project='imagenette',
                       group=args.group,
                       config=args,
                       name=args.name,
                       dir=os.path.join(args.work_dir, args.group, args.name))

    training.train_and_evaluate(args)


if __name__ == '__main__':
    main()
