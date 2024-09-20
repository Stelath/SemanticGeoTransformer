from typing import Dict

import torch
import ipdb
from tqdm import tqdm

from geotransformer.engine.base_tester import BaseTester
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.timer import Timer
from geotransformer.utils.common import get_log_string
from geotransformer.utils.torch import release_cuda, to_cuda

import os
import numpy as np


class SingleTester(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)
        

        for iteration, data_dict in pbar:
            # on start
            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
       
            self.before_test_step(self.iteration, data_dict)
            # test step
          
            torch.cuda.synchronize()
            timer.add_prepare_time()
         
            output_dict = self.test_step(self.iteration, data_dict)
           
            torch.cuda.synchronize()
            timer.add_process_time()
            # eval step
          
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            # after step    
         
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)

            if self.args.save_registered_point_clouds:
                # Write the .npy of the two point clouds along with the outputted transformation matrix
                # to the output directory
                seq_id = data_dict['seq_id']
                ref_frame = data_dict['ref_frame']
                src_frame = data_dict['src_frame']

                output_dir = os.path.join(self.output_dir, f'{seq_id}')
                
                os.makedirs(output_dir, exist_ok=True)
                print(f"Saving to {output_dir}")

                # Remove any file extensions from ref_frame and src_frame
                if type(ref_frame) == type("string"):
                    ref_frame = ref_frame.split('.')[0]
                    src_frame = src_frame.split('.')[0]

                # Save the ground truth
                np.save(os.path.join(output_dir, f'{ref_frame}_{src_frame}_transform_gt.npy'), data_dict['transform'].cpu().numpy())
                np.save(os.path.join(output_dir, f'{ref_frame}_{src_frame}_transform_pr.npy'), output_dict['estimated_transform'].cpu().numpy())
                np.save(os.path.join(output_dir, f'{ref_frame}_{src_frame}_ref.npy'), output_dict['ref_points'].cpu().numpy())
                np.save(os.path.join(output_dir, f'{ref_frame}_{src_frame}_src.npy'), output_dict['src_points'].cpu().numpy())
                

            # logging
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += f', {timer.tostring()}'
            pbar.set_description(message)
           
            torch.cuda.empty_cache() 
           
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)
