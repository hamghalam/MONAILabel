import copy
import logging
import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from lib import MyInfer, MyStrategy, MyTrain
from monai.apps import load_from_mmar

from monailabel.interfaces import MONAILabelApp
from monailabel.utils.activelearning import Random
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference

from monai.data import DataLoader, PersistentDataset
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    CropForegroundd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

logger = logging.getLogger(__name__)

import argparse
parser = argparse.ArgumentParser(description='Active Learning Setting')

parser.add_argument('--checkpoint', default=None)
parser.add_argument('--logdir', default=None)
parser.add_argument('--save_checkpoint', action='store_true')
parser.add_argument('--optim_name', default='adamw', type=str)

parser.add_argument('--reg_weight', default=1e-5, type=float)
parser.add_argument('--task', default='brats18')

parser.add_argument('--dropout_prob', default=0, type=float)
parser.add_argument('--model_name', default=None)

# Directory & Json
parser.add_argument('--base_dir', default='/home/vishwesh/experiments/active_learning_random_test', type=str)
parser.add_argument('--data_root', default='/home/vishwesh/experiments/monai_label_spleen/data', type=str)
parser.add_argument('--json_path', default='/home/vishwesh/experiments/monai_label_spleen/data/dataset_10_init_v2.json', type=str)

# Active learning parameters
parser.add_argument('--active_iters', default=5, type=int)
parser.add_argument('--dropout_ratio', default=0.2, type=float)
parser.add_argument('--mc_number', default=10, type=int)
parser.add_argument('--queries', default=5, type=int)
parser.add_argument('--random_strategy', default=0, type=int)

# DL Hyper-parameters
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--val_batch_size', default=1, type=int)
parser.add_argument('--lr', default=1e-4, type=float)

class MyApp(MONAILabelApp):
    def __init__(self, app_dir, studies):
        self.model_dir = os.path.join(app_dir, "model")
        self.final_model = os.path.join(self.model_dir, "model.pt")

        self.mmar = "clara_pt_spleen_ct_segmentation_1"

        super().__init__(app_dir, studies, os.path.join(self.model_dir, "train_stats.json"))


    '''
    def init_infers(self):
        infers = {
            "segmentation_spleen": MyInfer(self.final_model, load_from_mmar(self.mmar, self.model_dir)),
        }

        # Simple way to Add deepgrow 2D+3D models for infer tasks
        infers.update(self.deepgrow_infer_tasks(self.model_dir))
        return infers
    '''

    def init_strategies(self):
        return {
            "random": Random(),
            "first": MyStrategy(),
        }

    def train(self, request):
        logger.info(f"Training request: {request}")

        #output_dir = os.path.join(self.model_dir, request.get("name", "model_infer_test"))
        output_dir = request.get("output_dir")
        network = request.get("network")
        train_d = request.get("train_data")
        val_d = request.get("val_data")
        epochs = request.get("epochs")
        lr = request.get("lr")
        train_batch_size = request.get("train_batch_size")
        val_batch_size = request.get("val_batch_size")

        self.final_model = os.path.join(output_dir, 'model.pt')
        self.train_stats_path = os.path.join(output_dir, 'train_stats.json')

        if "load_path" in request:
            ckpt_path = request.get("load_path")
            task = MyTrain(
                output_dir=output_dir,
                train_datalist=train_d,
                val_datalist=val_d,
                network=network,
                load_path=ckpt_path,
                publish_path=self.final_model,
                stats_path=self.train_stats_path,
                device=request.get("device", "cuda"),
                lr=lr,
                #val_split=request.get("val_split", 0.2),
                max_epochs=epochs,
                amp=request.get("amp", False),
                train_batch_size=train_batch_size,
                val_batch_size=val_batch_size,
            )
        else:
            task = MyTrain(
                output_dir=output_dir,
                train_datalist=train_d,
                val_datalist=val_d,
                network=network,
                load_path=None,
                publish_path=self.final_model,
                stats_path=self.train_stats_path,
                device=request.get("device", "cuda"),
                lr=lr,
                #val_split=request.get("val_split", 0.2),
                max_epochs=epochs,
                amp=request.get("amp", False),
                train_batch_size=train_batch_size,
                val_batch_size=val_batch_size,
            )

        # App Owner can decide which checkpoint to load (from existing output folder or from base checkpoint)
        #load_path = os.path.join(output_dir, "model.pt")
        '''
        if not os.path.exists(load_path) and request.get("pretrained", True):
            load_path = None
            network = load_from_mmar(self.mmar, self.model_dir)
        else:
            network = load_from_mmar(self.mmar, self.model_dir, pretrained=False)


        network = UNet(
                        dimensions=3,
                        in_channels=1,
                        out_channels=2,
                        channels=(16, 32, 64, 128, 256),
                        strides=(2, 2, 2, 2),
                        num_res_units=2,
                        norm=Norm.BATCH,
                        dropout=0.2
                    )
        
        # Datalist for train/validation
        #train_d, val_d = self.partition_datalist(self.datastore().datalist(), request.get("val_split", 0.2))

        # Load Json file
        data_root = self.studies

        json_file_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen/data/dataset_0.json')
        with open(json_file_path) as json_file:
            json_data = json.load(json_file)
        json_file.close()

        train_d = json_data['training']
        val_d = json_data['validation']

        # Add data_root to json
        for idx, each_sample in enumerate(train_d):
            train_d[idx]['image'] = os.path.join(data_root, train_d[idx]['image'])
            train_d[idx]['label'] = os.path.join(data_root, train_d[idx]['label'])

        for idx, each_sample in enumerate(val_d):
            val_d[idx]['image'] = os.path.join(data_root, val_d[idx]['image'])
            val_d[idx]['label'] = os.path.join(data_root, val_d[idx]['label'])
    
        print('Debug here')
        '''
        #self.batch_infer()

        return task()


def unl_pre_transforms():
    return [
        LoadImaged(keys=("image")),
        AddChanneld(keys=("image")),
        Spacingd(
            keys=("image"),
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=("image"), source_key="image"),
        ToTensord(keys=("image")),
    ]

def unl_data_loader(_unl_datalist):
    return (
        DataLoader(
            dataset=PersistentDataset(_unl_datalist, unl_pre_transforms(), cache_dir=None),
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )
#        if _val_datalist and len(_val_datalist) > 0
#        else None
    )


def entropy_3d_volume(vol_input):
    # The input is assumed with repetitions, channels and then volumetric data
    vol_input = vol_input.astype(dtype='float32')
    dims = vol_input.shape
    reps = dims[0]
    entropy = np.zeros(dims[2:], dtype='float32')

    # Threshold values less than or equal to zero
    threshold = 0.00005
    vol_input[vol_input <= 0] = threshold

    # Looping across channels as each channel is a class
    if len(dims) == 5:
        for channel in range(dims[1]):
            t_vol = np.squeeze(vol_input[:, channel, :, :, :])
            t_sum = np.sum(t_vol, axis=0)
            t_avg = np.divide(t_sum, reps)
            t_log = np.log(t_avg)
            t_entropy = -np.multiply(t_avg, t_log)
            entropy = entropy + t_entropy
    else:
        t_vol = np.squeeze(vol_input)
        t_sum = np.sum(t_vol, axis=0)
        t_avg = np.divide(t_sum, reps)
        t_log = np.log(t_avg)
        t_entropy = -np.multiply(t_avg, t_log)
        entropy = entropy + t_entropy

    return entropy

def main():

    # Argument parser Code    # monai.config.print_config()
    #
    args = parser.parse_args()

    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
        print('-----------------')

    # TODO Notes run 4 iterations generate model directory names in the format "model_1", "model_2" ...
    # TODO In the json list get an unlabeled pool of data, start with 2 or 3 volumes, keep 9 for validation
    # TODO Run uncertainty on post procesed activated probability maps
    # TODO Run inference and compute uncertainty for all unlabeled data, get file names attached with it

    # Base directory where all model collections will go
    base_model_dir = os.path.normpath(args.base_dir)
    if os.path.exists(args.base_dir) == False:
        os.mkdir(args.base_dir)

    base_model_dir = os.path.join(base_model_dir, 'all_models')
    if os.path.exists(base_model_dir) == False:
        os.mkdir(base_model_dir)

    json_base_dir = os.path.join(base_model_dir, 'all_jsons')
    if os.path.exists(json_base_dir) == False:
        os.mkdir(json_base_dir)

    fig_base_dir = os.path.join(base_model_dir, 'qa_figs')
    if os.path.exists(fig_base_dir) == False:
        os.mkdir(fig_base_dir)

    # Root data path
    data_root = os.path.normpath(args.data_root)

    # Active Json Paths
    new_json_path = ''

    # Previous Ckpt Path
    prev_best_ckpt = ''

    # Model Definition
    network = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=args.dropout_ratio
    )

    # Active Learning iterations
    for active_iter in range(args.active_iters):

        print('Currently on Active Iteration: {}'.format(active_iter))

        # Create current active model path
        model_name = 'model_' + str(active_iter)
        active_model_dir = os.path.join(base_model_dir, model_name)
        if os.path.exists == False:
            os.mkdir(active_model_dir)

        # Load JSON data
        if active_iter==0:
            print('Opening json file {} for Active iteration {}'.format(args.json_path, active_iter))
            with open(args.json_path, 'rb') as f:
                json_data = json.load(f)
            f.close()

        elif active_iter>0:
            print('Opening json file {} for Active iteration {}'.format(new_json_path, active_iter))
            with open(new_json_path, 'rb') as f:
                json_data = json.load(f)
            f.close()

        train_d = json_data['training']
        copy_train_d = copy.deepcopy(train_d)
        val_d = json_data['validation']
        copy_val_d = copy.deepcopy(val_d)
        unl_d = json_data['unlabeled']
        copy_unl_d = copy.deepcopy(unl_d)

        # TODO DONT FORGET TO REMOVE THE BELOW SIX LINES FROM THE MAIN APP CODE FUNCTION
        # Add data_root to json
        for idx, each_sample in enumerate(train_d):
            train_d[idx]['image'] = os.path.join(data_root, train_d[idx]['image'])
            train_d[idx]['label'] = os.path.join(data_root, train_d[idx]['label'])

        for idx, each_sample in enumerate(val_d):
            val_d[idx]['image'] = os.path.join(data_root, val_d[idx]['image'])
            val_d[idx]['label'] = os.path.join(data_root, val_d[idx]['label'])

        for idx, each_sample in enumerate(unl_d):
            unl_d[idx]['image'] = os.path.join(data_root, unl_d[idx]['image'])
            unl_d[idx]['label'] = os.path.join(data_root, unl_d[idx]['label'])

        # Define Request for App with hyper-parameters such as Lr, Batchsize, epochs
        request = {}
        request["val_batch_size"] = args.val_batch_size
        request["train_batch_size"] = args.batch_size
        request["epochs"] = args.epochs
        request["lr"] = args.lr
        request["val_data"] = val_d
        request["train_data"] = train_d
        request["network"] = network
        request["output_dir"] = active_model_dir
        if active_iter > 0:
            print('Sending Request to Load Check point {} for Active Iteration: {}'.format(prev_best_ckpt, active_iter))
            request["load_path"] = prev_best_ckpt

        # Create App Object
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        app_dir_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen')
        studies_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen/data')
        al_app = MyApp(app_dir=app_dir_path, studies=studies_path)

        print('Creating Training Request for Active Iteration: {}'.format(active_iter))
        al_app.train(request=request)
        print('Training Completed for Active Iteration: {}'.format(active_iter))

        if args.random_strategy == 0:

            print('Prepping to run inference on unlabeled pool of data')
            print('Loading the final set of trained weights for running inference')
            prev_best_ckpt = os.path.join(active_model_dir, 'model.pt')

            device = torch.device("cuda:0")
            ckpt = torch.load(prev_best_ckpt)
            network.load_state_dict(ckpt)
            network.to(device=device)

            # PLEASE NOTE THAT THE MODEL IS BEING PUT INTO train mode explicitly for MC simulations
            network.train()
            print('Weights Loaded and the Network has been put in TRAIN mode, not eval')

            unl_loader = unl_data_loader(unl_d)

            scores = {}

            print('Running inference for uncertainty ...')
            with torch.no_grad():
                counter = 1
                for unl_data in unl_loader:
                    unl_inputs = unl_data["image"].to(device)

                    roi_size = (160, 160, 160)
                    sw_batch_size = 4

                    accum_unl_outputs = []

                    for mc in range(args.mc_number):
                        unl_outputs = sliding_window_inference(
                            unl_inputs, roi_size, sw_batch_size, network)

                        # Activate the output with Softmax
                        unl_act_outputs = torch.softmax(unl_outputs, dim=1)

                        # Accumulate
                        accum_unl_outputs.append(unl_act_outputs)

                    # Stack it up
                    accum_tensor = torch.stack(accum_unl_outputs)

                    # Squeeze
                    accum_tensor = torch.squeeze(accum_tensor)

                    # Send to CPU
                    accum_numpy = accum_tensor.to('cpu').numpy()
                    # accum_numpy = accum_numpy[:, 1, :, :, :]
                    # Generate Entropy Map and Plot all slices
                    # TODO Only send the Spleen Prediction Class for Uncertainty Estimation,
                    #  modify when using other multi-class datasets
                    accum_numpy = accum_numpy[:, 1, :, :, :]

                    entropy = entropy_3d_volume(accum_numpy)
                    scores[unl_data['image_meta_dict']['filename_or_obj'][0]] = np.sum(entropy)
                    print('Entropy for image: {} is: {}'.format(unl_data['image_meta_dict']['filename_or_obj'][0], np.sum(entropy)))
                    #print('Debug here')

                    # Plot with matplotlib and save all slices
                    plt.figure(1)
                    plt.imshow(np.squeeze(entropy[:, :, 50]))
                    plt.colorbar()
                    plt.title('Dropout Uncertainty')
                    fig_path = os.path.join(fig_base_dir, 'active_{}_file_{}.png'.format(active_iter, counter))
                    plt.savefig(fig_path)
                    plt.clf()
                    plt.close(1)
                    counter = counter + 1


            print('Inference for Uncertainty Complete, working on ranking the unlabeled data')

            # Detach values and keys for sorting
            scores_vals = []
            scores_keys = []

            for key, value in scores.items():
                scores_vals_t = value
                score_keys_t = key.split('/')[-2:]
                print('Score Key is {} and value is {}'.format(score_keys_t, scores_vals_t))

                score_key_path = os.path.join(score_keys_t[0], score_keys_t[1])

                scores_vals.append(scores_vals_t)
                scores_keys.append(score_key_path)

            sorted_indices = np.argsort(scores_vals)

            # Retrieve most unstable samples list
            most_unstable = sorted_indices[-args.queries:]
            scores_keys = np.array(scores_keys)
            most_unstable_names = scores_keys[most_unstable]
            most_unstable_names = most_unstable_names.tolist()

            # Form the new JSON
            # Get indices from unlabeled data pool using most unstable names
            grab_indices = []
            for each_unstable_name in most_unstable_names:
                for idx_unl, each_sample in enumerate(copy_unl_d):
                    print(each_sample)
                    if each_unstable_name == each_sample["image"]:
                        grab_indices.append(idx_unl)

            copy_unl_d = np.array(copy_unl_d)
            samples = copy_unl_d[grab_indices]
            for each in samples:
                copy_train_d.append(each)

            copy_unl_d = np.delete(copy_unl_d, grab_indices)
            copy_unl_d = copy_unl_d.tolist()

            print("Updated Json File Sample Count")
            print("Number of Training samples for next iter: {}".format(len(copy_train_d)))
            print("Number of Unlabeled samples for next iter: {}".format(len(copy_unl_d)))

        elif args.random_strategy == 1:
            print('Random Strategy is being used for selection')
            grab_indices = random.sample(range(1, len(copy_unl_d)), args.queries)

            copy_unl_d = np.array(copy_unl_d)
            samples = copy_unl_d[grab_indices]
            for each in samples:
                copy_train_d.append(each)

            copy_unl_d = np.delete(copy_unl_d, grab_indices)
            copy_unl_d = copy_unl_d.tolist()

            print("Updated Json File Sample Count")
            print("Number of Training samples for next iter: {}".format(len(copy_train_d)))
            print("Number of Unlabeled samples for next iter: {}".format(len(copy_unl_d)))


        # Write new json file
        new_json_dict = {}
        new_json_dict['training'] = copy_train_d
        new_json_dict['unlabeled'] = copy_unl_d
        new_json_dict['validation'] = copy_val_d

        new_json_file_path = os.path.join(json_base_dir, 'json_iter_{}.json'.format(active_iter))
        with open(new_json_file_path, 'w') as j_file:
            json.dump(new_json_dict, j_file)
        j_file.close()

        # Update New Json path
        new_json_path = new_json_file_path
        print('Active Iteration {} Completed'.format(active_iter))

    return None


########################
'''
    device = torch.device("cuda:0")

    model_weights_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen/model/model_dropout_02_bs4_sameres/model.pt')
    ckpt = torch.load(model_weights_path)
    network.load_state_dict(ckpt)
    network.to(device=device)
    print('Weights Loaded Succesfully')

    #network.eval()
    network.train()
    # TODO Batch Inference
    # Send the datastore
    #al_app.batch_infer(datastore=)
    #infers = MyInfer(self.final_model, load_from_mmar(self.mmar, self.model_dir))

    data_root = studies_path
    json_file_path = os.path.normpath('/home/vishwesh/experiments/monai_label_spleen/data/dataset_0.json')
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)
    json_file.close()

    train_d = json_data['training']
    val_d = json_data['validation'][0:2]

    # Add data_root to json
    for idx, each_sample in enumerate(train_d):
        train_d[idx]['image'] = os.path.join(data_root, train_d[idx]['image'])
        train_d[idx]['label'] = os.path.join(data_root, train_d[idx]['label'])

    for idx, each_sample in enumerate(val_d):
        val_d[idx]['image'] = os.path.join(data_root, val_d[idx]['image'])
        val_d[idx]['label'] = os.path.join(data_root, val_d[idx]['label'])

    val_loader = val_data_loader(val_d)

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            roi_size = (160, 160, 160)
            sw_batch_size = 4

            accum_val_inputs = []

            for mc in range(10):
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, network)

                # Activate the output with Softmax
                val_act_outputs = torch.softmax(val_outputs, dim=1)

                # Accumulate
                accum_val_inputs.append(val_act_outputs)


            # Stack it up
            accum_tensor = torch.stack(accum_val_inputs)

            # Squeeze
            accum_tensor = torch.squeeze(accum_tensor)

            # Send to CPU
            accum_numpy = accum_tensor.to('cpu').numpy()
            #accum_numpy = accum_numpy[:, 1, :, :, :]
            # Generate Entropy Map and Plot all slices
            entropy = entropy_3d_volume(accum_numpy)

            # Plot with matplotlib and save all slices
            plt.imshow(np.squeeze(entropy[:, :, 50]))
            plt.colorbar()
            plt.title('Dropout Uncertainty')
            plt.show()
            print('Debug here')
'''

if __name__=="__main__":
    main()