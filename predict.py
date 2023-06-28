import os,sys,fnmatch,argparse,csv
import pandas as pd
import numpy as np

from typing import Dict, List
import torch
import torch.nn as nn
from torch import Tensor, nn

import time

class EnsembledModel(nn.Module):
    def __init__(self, models: List, x=['coord', 'numbers', 'charge'], out=['energy'], detach=False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.x = x
        self.out = out
        self.detach = detach

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        res : List[Dict[str, Tensor]] = []
        for model in self.models:
            _in = dict()
            for k in data:
                if k in self.x:
                    _in[k] = data[k]
            _out = model(_in)
            _r = dict()
            for k in _out:
                if k in self.out:
                    _r[k] = _out[k]
                    if self.detach:
                        _r[k] = _r[k].detach()
            res.append(_r)

        for k in res[0]:
            v = []
            for x in res:
                v.append(x[k])
            vv = torch.stack(v, dim=0)
            data[k] = vv.mean(dim=0)
            data[k + '_std'] = vv.std(dim=0)
        return data

def parse_xyz(fname=None):
    coord = []
    numbers = []
    elem_to_num = {'H':1, 'C':6, 'N':7, 'O':8}
    with open(fname) as f:
        line = f.readline()
        row = line.split()
        n_atoms = int(row[0])
        while line:
            row = line.split()
            if len(row) > 1:
                numbers.append(elem_to_num[row[0]])
                coord.append([float(row[-3]),float(row[-2]), float(row[-1])])
                if len(numbers) == n_atoms:
                    f.close()
                    break
            line = f.readline()
    return coord, numbers

def main(argv):

    parser = argparse.ArgumentParser(description='This script will provide B3LYP-D3/TZVP (G4) level energy predictions based' + \
                                     'on the energies and geometries computed at GFN2-xTB level of theory')

    parser.add_argument('-g', dest='geometry', default='input_geo',
                        help = 'The program expects a folder of xyz files which contain GFN2-xTB optimized geometries')

    parser.add_argument('-e', dest='energy', default='xTB_energy.csv',
                        help = 'The program expects a csv file of GFN2-xTB level energy (match with input geometry)')

    parser.add_argument('-l', dest='level', default='DFT',
                        help = 'Energies at two level of theory can be selected, DFT (B3LYP-D3/TZVP) and Gaussian-4')

    parser.add_argument('-o', dest='output', default='Delta2_pred.csv',
                        help = 'Output file storing the ')

    # parse configuration
    args=parser.parse_args()

    # initialize output file
    with open(args.output, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['name','xTB_SPE','pred_SPE','std'])

    # time counting
    start_time = time.time()

    # load in input geometries
    xyzs = os.listdir(args.geometry)

    # prepare basic xTB energy dictionary
    df = pd.read_csv(args.energy)
    ene_dict = dict()
    fail_list = []
    for xyz in xyzs:
        _id = xyz.split('.')[0]
        try:
            ene_dict[_id] = float(df[df.name==_id].xTB_ene.item())
        except:
            fail_list.append(_id)

    # check missing xTB energies
    if len(fail_list) > 0:
        print("Check missing values for {}, quit...".format(fail_list))
        exit()

    # prepare input geometry dictionary
    ds = dict()
    for xyz in xyzs:

        _id = xyz.split('.')[0]
        try:
            coords, numbers = parse_xyz(args.geometry+'/'+xyz)
        except:
            print("Have trouble loading {}, skip...".format(xyz))
            pass

        d = {'_id':_id,'coord':coords,'numbers':numbers}
        dd = dict() 
        dd['_id'] = np.array([d['_id']])[()].astype('S')
        dd['coord'] = torch.tensor(d['coord']).view(1,-1,3).numpy()[()].astype('float32')
        dd['numbers'] = torch.tensor(d['numbers']).view(1, -1).numpy()[()].astype('uint8')
        dd['charge'] = np.array([0.0])[()].astype('int8')

        if dd['numbers'].shape[1] in ds:
            ds[dd['numbers'].shape[1]]['_id'] = np.hstack([ds[dd['numbers'].shape[1]]['_id'], dd['_id']])
            ds[dd['numbers'].shape[1]]['coord'] = np.vstack([ds[dd['numbers'].shape[1]]['coord'], dd['coord']])
            ds[dd['numbers'].shape[1]]['numbers'] = np.vstack([ds[dd['numbers'].shape[1]]['numbers'], dd['numbers']])
            ds[dd['numbers'].shape[1]]['charge'] = np.hstack([ds[dd['numbers'].shape[1]]['charge'], dd['charge']])
        else:
            ds[dd['numbers'].shape[1]] = dd

    for d in ds:
        ds[d]['_id'] = ds[d]['_id'].flatten()
        ds[d]['charge'] = ds[d]['charge'].flatten()

    # time counting
    model_loading_time = time.time()
    print("Time uses for loading input geometries: {:<10.4f}s".format(model_loading_time-start_time))

    # search for device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in models
    model_list = []
    if args.level == 'DFT': 
        sae = {1: -0.09565701919376791, 6: -35.97628109983475, 7: -51.84891086964016, 8: -71.18551920716808}
        prefix = 'DFT_models'
    elif args.level == 'G4': 
        prefix = 'G4_models'
        sae = {1: -0.09364398454577838, 6: -35.95667813236401, 7: -51.81919960387285, 8: -71.14043670055156}
    else: 
        print("Only two models are avaiable, DFT and G4, please check the input, quit...")
        quit()

    # search for jpt files
    jpts = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(prefix) for f in filenames if (fnmatch.fnmatch(f,"*.jpt") )])

    # construct the ensemble model
    for i in jpts:
        model = torch.jit.load(i, map_location='cpu') # change into device?
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        model_list.append(model)

    ens_model = EnsembledModel(model_list, x=['coord', 'numbers', 'charge'], out=['energy'], detach=False)
    ens_model = torch.jit.script(ens_model).to(device)

    # time counting
    computing_time = time.time()
    print("Time uses for construct ensemble model: {:<10.4f}s".format(computing_time-model_loading_time))

    # run Delta (AIMNET2) on input geometries
    for k,v in ds.items():
    
        # prepare input format
        coord = torch.tensor(v['coord'],requires_grad=False,dtype=torch.float32,device=device)
        numbers = torch.tensor(v['numbers'],requires_grad=False,dtype=torch.int64,device=device)
        charge = torch.zeros(len(v['charge']),device=device) # SHAPE IS BATCH
        _in  = dict(coord=coord, numbers=numbers,charge=charge)
        
        # run ensemble prediction
        _out = ens_model(_in)
        
        # search for xTB energies
        xtb_enes = torch.tensor(np.array([ene_dict[name.decode('utf8')] for name in v['_id']]),dtype=torch.float64,device=device)

        # compute correction term
        corr = torch.tensor(np.array([sum([sae[j] for j in number]) for number in v['numbers']]),dtype=torch.float64,device=device)

        # obtain mean and std of ensemble model
        E_mean = corr + xtb_enes + _out['energy']
        E_std  = _out['energy_std']

        # write down to output file
        with open(args.output, 'a') as csvfile: 
            csvwriter = csv.writer(csvfile)
            for i in range(len(v['charge'])):
                csvwriter.writerow([v['_id'][i].decode('utf8'),xtb_enes[i].detach().cpu().numpy(),
                                    E_mean[i].detach().cpu().numpy(),E_std[i].detach().cpu().numpy()])


    # time counting
    finish_time = time.time()
    print("Time uses for running aimnet2 model: {:<10.4f}s".format(finish_time-computing_time))
    print("Total Time uses: {:<10.4f}s".format(finish_time-start_time))

if __name__ == "__main__":
    main(sys.argv[1:])
