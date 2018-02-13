from __future__ import absolute_import, division, print_function
import argparse
import tensorflow as tf
import time
import os
import sys
from aae import GAN

def configure():
    flags = tf.app.flags
    flags.DEFINE_integer("batch_size", 10, "batch size")
    flags.DEFINE_integer("updates_per_epoch", 500, "number of updates per epoch")
    flags.DEFINE_integer("max_epoch", 1000, "max epoch for total training")
 #   flags.DEFINE_integer("max_con_epoch", 300, "max epoch for conditional part training")
    flags.DEFINE_integer("max_generated_imgs", 200, "max generated imgs for each input")
    flags.DEFINE_integer("max_test_epoch", 107, "max  test epoch")
    flags.DEFINE_integer("summary_step", 100, "save summary per #summary_step iters")
    flags.DEFINE_integer("save_step", 1000, "save model per #save_step iters")
    flags.DEFINE_integer("n_class", 10, "number of classes")
    flags.DEFINE_float("learning_rate", 2e-4, "learning rate")
    flags.DEFINE_float("gamma_gen", 1e-4, "gamma ratio for generator loss")
#   flags.DEFINE_float("gamma_dec", 1e-4, "gamma ratio for decoder")
    flags.DEFINE_float("gan_noise", 0.01, "injection noise for the GAN")
    flags.DEFINE_bool("noise_bool", False, "add noise on all GAN layers")
    flags.DEFINE_string("working_directory", "/tempspace/hyuan/cell_aae",
                        "the file directory")
    flags.DEFINE_integer("hidden_size", 16, "size of the hidden VAE unit")
 #   flags.DEFINE_integer("channel", 64, "size of initial channel in decoder")
    flags.DEFINE_integer("checkpoint", 500000, "number of epochs to be reloaded")
 #   flags.DEFINE_string("model_name", 'low_rank', "vanilla or low_rank")
    flags.DEFINE_integer("height", 256, "height of image")
    flags.DEFINE_integer("width", 256, "width of image")
    flags.DEFINE_string("modeldir", './modeldir_unet_guide', "the model directory")
    flags.DEFINE_string("logdir", './logdir_unet_guide', "the log directory")
    flags.DEFINE_string("sampledir", './sampledir_unet_guide', "the sample directory")
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS
 
def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--action',
        dest='action',
        type=str,
        default='train',
        help='actions: train, or test')
    args = parser.parse_args()
    if args.action not in ['train', 'test']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test")
    else:
        model= GAN(tf.Session(),configure())
        getattr(model,args.action)()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tf.app.run()





