import statistics
import numpy as np
import h5py

nns = ['train/KM3', 'train/SHL 17', 'train/SHL 28']
mm = ['CV', 'DV', 'CDV', 'SCV']
mml = {'CV':1, 'DV':2, 'CDV':3, 'SCV':4}
sz_thres = 5
sz = np.array([5,31,31])
szh = sz//2
out_im = np.zeros([0] + list(sz), np.uint8)
out_mask = np.zeros([0] + list(sz), np.uint8)
tmp = np.zeros(sz, np.uint8)
out_l = []
for i in range(len(nns)):
    im = read_h5(f'{D0}vol{i}_im.h5')
    ves = read_h5(f'{D0}vol{i}_vesicle_ins.h5')
    ves_l = read_h5(f'{D0}vol{i}_vesicle.h5')
    ves[ves_l==4] = 0
    bbs = compute_bbox_all(ves)
    for bb in bbs:
        bsz = (bb[2::2]-bb[1::2])+1
        # remove too small ones
        if bb[1:].min() > sz_thres:
            tmp = ves_l[ves==bb[0]]
            # ideally, tmp>0, but some VAST modification
            ll = statistics.mode(tmp[tmp>0])
            if ll==0:
                import pdb; pdb.set_trace()
            out_l.append(ll)
            cc = (bb[1::2]+bb[2::2])//2
            crop = im[max(0,cc[0]-szh[0]):cc[0]+szh[0], max(0,cc[1]-szh[1]):cc[1]+szh[1], max(0,cc[2]-szh[2]):cc[2]+szh[2]]
            diff = (sz - crop.shape) // 2
            # much blank
            diff2 = sz - crop.shape -diff
            tmp = np.pad(crop, [(diff[0],diff2[0]), (diff[1],diff2[1]), (diff[2],diff2[2])], 'edge')
            # pad: xy-edge, z-reflect
            out_im = np.concatenate([out_im, tmp[None]], axis=0)
            tmp[:] = 0

            crop = ves[max(0,cc[0]-szh[0]):cc[0]+szh[0], max(0,cc[1]-szh[1]):cc[1]+szh[1], max(0,cc[2]-szh[2]):cc[2]+szh[2]]==bb[0]
            tmp = np.pad(crop, [(diff[0],diff2[0]), (diff[1],diff2[1]), (diff[2],diff2[2])], 'edge')
            out_mask = np.concatenate([out_mask, tmp[None]], axis=0)
            tmp[:] = 0
write_h5(f'{D0}/bigV_cls_im.h5', out_im)
write_h5(f'{D0}/bigV_cls_mask.h5', out_mask)
write_h5(f'{D0}/bigV_cls_label.h5', np.array(out_l).astype(np.uint8))