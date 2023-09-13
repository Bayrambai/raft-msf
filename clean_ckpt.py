import torch
from collections import OrderedDict

best_ckpt = torch.load('./ckpt/checkpoint_best.ckpt', map_location='cuda')
state_dict = best_ckpt['state_dict']

new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if "flow_head" in k:
        continue
    else:
        new_state_dict[k] = v


best_ckpt['state_dict'] = new_state_dict
torch.save(best_ckpt, 'clean_raft_msf.ckpt')
