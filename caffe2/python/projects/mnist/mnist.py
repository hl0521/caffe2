# -*- coding: UTF-8 -*-
## @package mnint
# Module caffe2.python.models.mnist.mnist
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import sys
import shutil

from caffe2.python import core, cnn, net_drawer, workspace, visualize, brew

def global_init():
    # If you would like to see some really detailed initializations,
    # you can change --caffe2_log_level=0 to --caffe2_log_level=-1
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    caffe2_root = "~/work/caffe2"
    print("global init end{}!".format(sys._getframe().f_lineno))

def download_dataset(url, path):
    import request, zipfile, StringIO
    print "Downloading... ", url, "to", path
    r = request.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall(path)

def generate_db(image, label, name):
    name = os.path.join(data_folder, name)
    print 'DB: ', name
    if not os.path.exists(name):
        syscall = "~/work/caffe2/build/caffe2/binaries/binaries/make_mnist_db --channel_first --db leveldb --image_file " + image + " --label_file " + label + " --output_file " + name
        print "Creating database with: ", syscall
        os.system(syscall)
    else:
        print "Database exists already. Delete the folder if you have issues/corrupted DB, then rerun this."
        if os.path.exists(os.path.join(name, "LOCK")):
            print "Deleting the pre-existing lock file"
            os.remove(os.path.join(name, "LOCK"))


def add_input(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label

def add_lenet_model(model, data):
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax

def add_accuracy(model, softmax, label):
    accuracy = model.Accuracy([softmax, label], "accuracy")
    return accuracy

def add_training_operators(model, softmax, label):
    # something very important happens here
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    add_accuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = model.Iter("iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - ModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)
    # let's checkpoint every 20 iterations, which should probably be fine.
    # you may need to delete tutorial_files/tutorial-mnist to re-run the tutorial
    model.Checkpoint([ITER] + model.params, [],
                   db="mnist_lenet_checkpoint_%05d.leveldb",
                   db_type="leveldb", every=20)

def add_bookkeeping_operators(model):
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.
    print("Bookkeeping function created")

def main():
    ############################################################################
    # working directory and download datasets
    # current_folder = "~/caffe2_notebooks"
    current_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks')
    # data_folder = "~/caffe2_notebooks/tutorial_data/mnist"
    data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
    # root_folder = "~/caffe2_notebooks/tutorial_files/tutorial_mnist"
    root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
    # image_file_train = "~/caffe2_notebooks/tutorial_data/mnist/train-images-idx3-ubyte"
    image_file_train = os.path.join(data_folder, "train-images-idx3-ubyte")
    # label_file_train = "~/caffe2_notebooks/tutorial_data/mnist/train-labels-idx1-ubyte"
    label_file_train = os.path.join(data_folder, "train-labels-idx1-ubyte")
    # image_file_test = "~/caffe2_notebooks/tutorial_data/mnist/t10k-images-idx3-ubyte"
    image_file_test = os.path.join(data_folder, "t10k-images-idx3-ubyte")
    # label_file_test = "~/caffe2_notebooks/tutorial_data/mnist/t10k-labels-idx1-ubyte"
    label_file_test = os.path.join(data_folder, "t10k-labels-idx1-ubyte")

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(label_file_train):
        download_dataset("https://s3.amazonaws.com/caffe2/datasets/mnist/mnist.zip", data_folder)

    if os.path.exists(root_folder):
        print("Looks like you ran this before, so we need to cleanup those old files...")
        shutil.rmtree(root_folder)

    # (Re)generate the leveldb database (known to get corrupted...)
    generate_db(image_file_train, label_file_train, "mnist-train-nchw-leveldb")
    generate_db(image_file_test, label_file_test, "mnist-test-nchw-leveldb")

    print("training data folder: " + data_folder)
    print("workspace root folder: " + root_folder)

    ############################################################################
    # create train and test net
    arg_scope = {"order": "NCHW"}
    train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)
    data, label = add_input(train_model, batch_size=64,
        db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'),
        db_type='leveldb')
    softmax = add_lenet_model(train_model, data)
    add_training_operators(train_model, softmax, label)
    add_bookkeeping_operators(train_model)

    # Testing model. We will set the batch size to 100, so that the testing
    # pass is 100 iterations (10,000 images in total).
    # For the testing model, we need the data input part, the main LeNetModel
    # part, and an accuracy part. Note that init_params is set False because
    # we will be using the parameters obtained from the train model.
    test_model = model_helper.ModelHelper(
        name="mnist_test", arg_scope=arg_scope, init_params=False)
    data, label = add_input(test_model, batch_size=100,
        db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'),
        db_type='leveldb')
    softmax = add_lenet_model(test_model, data)
    add_accuracy(test_model, softmax, label)

    # Deployment model. We simply need the main LeNetModel part.
    deploy_model = model_helper.ModelHelper(
        name="mnist_deploy", arg_scope=arg_scope, init_params=False)
    add_lenet_model(deploy_model, "data")
    # You may wonder what happens with the param_init_net part of the deploy_model.
    # No, we will not use them, since during deployment time we will not randomly
    # initialize the parameters, but load the parameters from the db.

    print(str(train_model.param_init_net.Proto()))

    with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:
        fid.write(str(train_model.net.Proto()))
    with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
        fid.write(str(train_model.param_init_net.Proto()))
    with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:
        fid.write(str(test_model.net.Proto()))
    with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:
        fid.write(str(test_model.param_init_net.Proto()))
    with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
        fid.write(str(deploy_model.net.Proto()))
    print("Protocol buffers files have been created in your root folder: "+root_folder)

    ############################################################################
    # The parameter initialization network only needs to be run once.
    workspace.RunNetOnce(train_model.param_init_net)
    # creating the network
    workspace.CreateNet(train_model.net)
    # set the number of iterations and track the accuracy & loss
    total_iters = 200
    accuracy = np.zeros(total_iters)
    loss = np.zeros(total_iters)
    # Now, we will manually run the network for 200 iterations.
    for i in range(total_iters):
        workspace.RunNet(train_model.net.Proto().name)
        accuracy[i] = workspace.FetchBlob('accuracy')
        loss[i] = workspace.FetchBlob('loss')


if __name__ == '__main__'
    main()
