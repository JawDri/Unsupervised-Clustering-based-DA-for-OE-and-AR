import argparse


def opts():
    parser = argparse.ArgumentParser(description='H-SRDC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # datasets
    parser.add_argument('--data_path_source', type=str, default='./data/datasets/Office31/', help='root of source training set')
    parser.add_argument('--data_path_target', type=str, default='./data/datasets/Office31/', help='root of target training set')
    parser.add_argument('--data_path_target_t', type=str, default='./data/datasets/Office31/', help='root of target test set')
    parser.add_argument('--src', type=str, default='amazon', help='source training set')
    parser.add_argument('--tar', type=str, default='webcam_half', help='target training set')
    parser.add_argument('--tar_t', type=str, default='webcam_half2', help='target test set')
    parser.add_argument('--num_classes', type=int, default=2, help='class number')#######
    # general optimization options
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--no_da', action='store_true', help='whether to not use data augmentation')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lrplan', type=str, default='dao', help='learning rate decay plan of dao or exp or step for attention module')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (l2 penalty)')
    parser.add_argument('--nesterov', action='store_true', help='whether to use nesterov SGD')
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120], help='decrease learning rate at these epochs for step decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='lr is multiplied by gamma on schedule')
    parser.add_argument('--eps', type=float, default=1e-8, help='a small value to prevent underflow')
    # specific optimization options
    parser.add_argument('--cluster_iter', type=int, default=5, help='number of iterations of k-means')
    parser.add_argument('--cluster_method', type=str, default='kmeans', help='cluster method to choose [kmeans|spherical_kmeans]')
    parser.add_argument('--beta', type=float, default=1.0, help='weight of auxiliary target distribution or assigned cluster labels')
    parser.add_argument('--initial_cluster', type=int, default=0, help='target or source class centroids for initialization of k-means')
    parser.add_argument('--src_cen_first', action='store_true', help='whether to use source centroids as target cluster centers at the first epoch')
    parser.add_argument('--src_cls', action='store_true', help='whether to classify source instances when clustering target instances')
    parser.add_argument('--src_soft_label', action='store_true', help='whether to use convex combination of true label vector and predicted label vector as training guide')
    parser.add_argument('--learn_embed', action='store_true', help='whether to apply embedding clustering')
    parser.add_argument('--embed_softmax', action='store_true', help='whether to use softmax to normalize soft cluster assignments for embedding clustering')
    parser.add_argument('--scale', type=float, default=1.0, help='maximum value of lambda (default: 1.0)')
    parser.add_argument('--div', type=str, default='kl', help='measure of divergence between one target instance and its perturbed counterpart')
    parser.add_argument('--gray_tar_agree', action='store_false', help='whether to enforce consistency between RGB and gray images on the target domain')
    parser.add_argument('--aug_tar_agree', action='store_false', help='whether to enforce consistency between RGB and augmented images on the target domain')
    parser.add_argument('--sigma', type=float, default=0.1, help='standard deviation of Gaussian for data augmentation operation of blurring')
    ## source sample selection
    parser.add_argument('--src_soft_select', action='store_true', help='whether to softly select source instances')
    parser.add_argument('--record_weight_rank', action='store_true', help='whether to record weight rank of source samples')
    parser.add_argument('--src_mix_weight', action='store_true', help='whether to mix 1 and soft weight')    
    # checkpoints
    parser.add_argument('--log', type=str, default='./checkpoints/office31', help='log folder')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume (default: None)')
    # architecture
    parser.add_argument('--arch', type=str, default='resnet50', help='model name')
    parser.add_argument('--num_neurons', type=int, default=128, help='number of neurons of fc1 of new model')
    parser.add_argument('--pretrained', action='store_false', help='whether using pretrained model')
    parser.add_argument('--only_decoder', action='store_true', help='whether to only use decoder for set transformer')
    parser.add_argument('--num_head', type=int, default=4, help='head number of multi-head attention module')
    # i/o
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 4)')
    parser.add_argument('--start_epoch', type=int, default=0, help='manual epoch number (useful on restarts)')
    parser.add_argument('--stop_epoch', type=int, default=200, help='stop epoch (default: 200)')
    parser.add_argument('--print_freq', type=int, default=1, help='print frequency (default: 10)')
    
    args = parser.parse_args()
    args.pretrained = True
    args.src_cls = True
    args.src_cen_first = True
    args.learn_embed = True
    args.embed_softmax = True
    if args.tar.find('amazon') != -1 and args.src.find('webcam') != -1:
        args.beta = 0.5
        args.initial_cluster = 1
        
    args.log = args.log + '_' + args.src + '2' + args.tar + '_bs' + str(args.batch_size) + '_' + args.arch + '_lr' + str(args.lr) + '_' + args.cluster_method

    return args
