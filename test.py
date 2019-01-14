import os, sys
sys.path.insert(0,"/Users/momo/wkspace/caffe_space/detection/caffe/python")
import numpy as np
import scipy.misc
from PIL import Image
import scipy.io
import cv2
import time
import caffe

EPSILON = 1e-8
data_root = '/Users/momo/wkspace/videoQ/data/'
with open('/Users/momo/wkspace/videoQ/data/test.lst') as f:
#data_root = '/Users/momo/Downloads/MSRA-B/'
#with open('/Users/momo/Downloads/test.lst') as f:
    test_lst = f.readlines()

test_lst = [data_root+x.strip() for x in test_lst]

caffe.set_mode_cpu()
caffe.SGDSolver.display = 0
# load net
net = caffe.Net('deploy.prototxt', 'dss_model_released.caffemodel', caffe.TEST)
#idx = 1
for idx in xrange(len(test_lst)):
    # load image
    img = Image.open(test_lst[idx])
    print test_lst[idx]
#    save_name = test_lst[idx].split('/')[-1].split('.')[0]
    print img.size
#    img.show()         #show ori pic
    img.save('../data/'+str(idx)+'.jpg')
    img = np.array(img, dtype=np.uint8)
    im = np.array(img, dtype=np.float32)
    im = im[:,:,::-1]
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = im.transpose((2,0,1))

    # load gt
    #gt = Image.open(test_lst[idx][:-4] + '.png')
    #gt.show()
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im
    # run net and take argmax for prediction
    start = time.clock()
    net.forward()
    finish = time.clock()
    print "time:", (finish-start)*1000.0, "ms"
    out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
    out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
    out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
    out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
    out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
    out6 = net.blobs['sigmoid-dsn6'].data[0][0,:,:]
    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    res = (out3 + out4 + out5 + fuse) / 4
    res = (res - np.min(res) + EPSILON) / (np.max(res) - np.min(res) + EPSILON)
    out_lst = [out1, out2, out3, out4, out5]
    name_lst = ['SO1', 'SO2', 'SO3', 'SO4', 'SO5']
    #plot_single_scale(out_lst, name_lst, 10)
    #out_lst = [out6, fuse, res, img, gt]
    #name_lst = ['SO6', 'Fuse', 'Result', 'Source', 'GT']
    #plot_single_scale(out_lst, name_lst, 10)
    #print out_lst[0].shape
    #im = Image.fromarray(np.uint8(out1*1000))
    #im.show()
    #im = Image.fromarray(np.uint8(out2*1000))
    #im.show()
    #im = Image.fromarray(np.uint8(out3*1000))
    #im.show()
    #im = Image.fromarray(np.uint8(out4*1000))
    #im.show()
    #im = Image.fromarray(np.uint8(out5*1000))
    #im.show()
    im = Image.fromarray(np.uint8(res*1000))
#    im.show()           #show ret pic
    im.save('../out/'+str(idx)+'_res.jpg')
