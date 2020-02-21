import os
import cv2
import torch
import argparse
import numpy as np
from simplenet_train import Tailor as Net
from simplenet_train import DilateSR, ResUNet
from torchvision.utils import save_image

os.environ['CUDA_VISIBLE_DEVICES']='0'
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_test', type=str, default='model/checkpoint.pth.tar', 
                        help='the path of the restored model checkpoint')
    parser.add_argument('--test_dir', type=str, default='samples/', 
                        help='the directory of testing samples')   
    parser.add_argument('--output_dir', type=str, default='results/', 
                        help='the directory to store demo outputs')
    parser.add_argument('--is_crf', type=int, default=0, 
                        help='whether use CRF as post-processing or not')
    parser.add_argument('--img_channel', type=int, default=3, 
                        help='the number of input image channels')
    parser.add_argument('--num_channel', type=int, default=100, 
                        help='the number of output channels')
    
    args = parser.parse_args()
    demo(args)

def demo(args):
    if not os.path.exists(args.output_dir):
        print("===== create output directory: {} =====".format(args.output_dir))
        os.makedirs(args.output_dir)
        
    num_gpus = torch.cuda.device_count()
    use_cuda = True if num_gpus >= 1 else False
    
    # ===========
    # model definition
    # ===========
    encoder = DilateSR(args.img_channel, args.num_channel)
    decoder = ResUNet()
    model = Net(encoder, decoder)
    
    
    # ==========
    # load weights
    # ==========
    if args.weights_test:
        weight_dict = torch.load(args.weights_test, map_location='cpu')
        model.load_state_dict(weight_dict['state_dict'])
        print("Loading weights from {}".format(args.weights_test))
    else:
        print("Please provide the location of weight files using --weights argument")
        return

    # ==========
    # predict
    # ==========
    # print(use_cuda)
    if use_cuda:
        torch.cuda.empty_cache()
        model.cuda()
    with torch.no_grad():
        label_colours = np.random.randint(255,size=(100,3))
        test_file = [os.path.join(args.test_dir, f) for f in sorted(os.listdir(args.test_dir))]
        for f in test_file:
            # torch.cuda.empty_cache()         
            # load image
            im = cv2.imread(f)
            data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )

            if use_cuda:
                data = data.cuda()	
            # data = torch.autograd.Variable(data)
            
            # forwarding
            output, reconst = model( data )
            output = output[ 0 ]
            raw = output
            output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.num_channel ) # channel map

            ignore, target = torch.max( output, 1 ) # p-map, (h*w, 1), grayscale level = p
            num_label = len(torch.unique(target))
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target)) # nLabels is the value of p
           
            # CRF post-processing
            if args.is_crf:
                from crf import dense_crf
                p_map = torch.nn.functional.softmax(raw, dim=0)
                p_map = p_map.cpu().numpy()
                img_data = im
                crf_label = dense_crf(img_data, p_map)
                crf_label = crf_label.flatten()
                im_target = crf_label
           
            im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
            f = os.path.splitext(f)[0]+'.bmp'
            cv2.imwrite(os.path.join(args.output_dir, os.path.basename(f)), im_target_rgb )
            print("{} done.".format(f))
            
if __name__ == '__main__':
    main()