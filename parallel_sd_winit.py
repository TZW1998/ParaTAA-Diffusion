import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
import argparse
from tqdm.auto import tqdm
import numpy as np
import time
from PIL import Image
import shutil, os
from accelerate import Accelerator
import pickle
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
import pickle
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)

class ParallelDDIM:

    def __init__(self, timesteps = 100,
                       scheduler = None,
                       window_size = 5, 
                       eta = 0.0, 
                       cfg_scale = 4.0, 
                       order = 3, 
                       memory_size = 0,
                       lmd = 1e-3,
                       tol1 = 1e-2,
                       tol2 = 5e-2,
                       device = "cuda",
                       use_fp16 = True):
        self.eta = eta 
        self.timesteps = timesteps
        self.num_steps = timesteps
        self.scheduler = scheduler
        self.device = device
        self.cfg_scale = cfg_scale
        self.tol1 = tol1
        self.tol2 = tol2
        self.lmd = lmd
        self.memory_size = memory_size
        self.use_fp16 = use_fp16

        self.window_size = window_size
        self.order = order
        assert order <= window_size

        # compute some coefficients in advance
        scheduler_timesteps = self.scheduler.timesteps.tolist()
        scheduler_prev_timesteps = scheduler_timesteps[1:]
        scheduler_prev_timesteps.append(0)
        self.scheduler_timesteps = scheduler_timesteps[::-1]
        scheduler_prev_timesteps = scheduler_prev_timesteps[::-1]
        alphas_cumprod = [1 - self.scheduler.alphas_cumprod[t] for t in self.scheduler_timesteps]
        alphas_cumprod_prev = [1 - self.scheduler.alphas_cumprod[t] for t in scheduler_prev_timesteps]

        now_coeff = torch.tensor(alphas_cumprod, dtype = torch.float64)
        next_coeff = torch.tensor(alphas_cumprod_prev, dtype = torch.float64)
        now_coeff = torch.clamp(now_coeff, min = 0)
        next_coeff = torch.clamp(next_coeff, min = 0)
        m_now_coeff = torch.clamp(1 - now_coeff, min = 0)
        m_next_coeff = torch.clamp(1 - next_coeff, min = 0)
        self.noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
        self.nl = self.noise_thr * self.eta
        self.nl[0] = 0.
        m_nl_next_coeff = torch.clamp(next_coeff - self.nl**2, min = 0)
        self.coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
        self.coeff_d = torch.sqrt(m_nl_next_coeff) - torch.sqrt(now_coeff) * self.coeff_x

        self.noise_thr = self.noise_thr.to(self.device)
        self.nl = self.nl.to(self.device)
        self.coeff_x = self.coeff_x.to(self.device)
        self.coeff_d = self.coeff_d.to(self.device)

        # compute the coeff matrix in advance
        self.coeff_x_mat = torch.zeros(self.num_steps, self.num_steps, device = self.device, dtype = torch.float64)
        self.coeff_d_mat = torch.zeros(self.num_steps, self.num_steps, device = self.device, dtype = torch.float64)
        self.nl_mat = torch.zeros(self.num_steps, self.num_steps, device = self.device, dtype = torch.float64)
        for t in range(self.num_steps):
            self.coeff_x_mat[t, t:] = torch.cumprod(self.coeff_x[t:], dim = 0)
            self.coeff_d_mat[t, t] = self.coeff_d[t]
            self.coeff_d_mat[t, (t+1):] = self.coeff_x_mat[t, t:-1] * self.coeff_d[t+1:]
            self.nl_mat[t, t] = self.nl[t]
            self.nl_mat[t, (t+1):] = self.coeff_x_mat[t, t:-1] * self.nl[t+1:]

        self.scheduler_timesteps_list = self.scheduler_timesteps.copy()
        self.scheduler_timesteps = torch.tensor(self.scheduler_timesteps, device = self.device)


    def is_finished(self):
        return self._is_finished

    def get_last_sample(self):
        return self._samples[0].clone()

    
    def initialize_samples(self):
        initialized_len = (self.finished_index - self.uninitialized_index)
        if (self.uninitialized_index >= 0) and (initialized_len < self.window_size):
            end_index = max([-1, self.finished_index - self.window_size])
            init_indexes = torch.arange(end_index + 1, self.uninitialized_index + 1)
            self._samples[init_indexes] = self._samples[self.uninitialized_index + 1].repeat(len(init_indexes), 1, 1, 1)
                 
            self.uninitialized_index = end_index

    def prepare_model_kwargs(self, rank, world_size):
        self.initialize_samples()
        
        t_end = self.finished_index + 1
        t_start = max([t_end - self.window_size, 1])
        NFE = t_end - t_start
        
        x = self._samples[t_start:t_end]
        x_cfg = torch.cat([x, x], dim = 0)
        ts = torch.arange(t_start, t_end) - 1
        ts = self.scheduler_timesteps[ts].to(self.device)
        ts_cfg = torch.cat([ts, ts], dim = 0)
            
        y = self.prompt_embeds[:NFE]
        y_null = self.null_prompt_embeds[:NFE]
        
        y_cfg = torch.cat([y_null, y], dim = 0)
     
        model_kwargs = {
            "sample": x_cfg,
            "timestep": ts_cfg,
            "encoder_hidden_states": y_cfg,
        }
        
        local_NFE = NFE // world_size

        if rank < world_size - 1:
            local_x = x[rank * local_NFE:(rank + 1) * local_NFE]
            local_ts = ts[rank * local_NFE:(rank + 1) * local_NFE]
            local_y = y[rank * local_NFE:(rank + 1) * local_NFE]
            local_y_null = y_null[rank * local_NFE:(rank + 1) * local_NFE]
        else:
            local_x = x[rank * local_NFE:]
            local_ts = ts[rank * local_NFE:]
            local_y = y[rank * local_NFE:]
            local_y_null = y_null[rank * local_NFE:]

        local_model_kwargs = {
                "sample": torch.cat([local_x, local_x], dim = 0),
                "timestep": torch.cat([local_ts, local_ts], dim = 0),
                "encoder_hidden_states": torch.cat([local_y, local_y_null], dim = 0),
            }

        if self.use_fp16:
            local_model_kwargs["sample"] = local_model_kwargs["sample"].to(dtype = torch.float16)
            local_model_kwargs["encoder_hidden_states"] = local_model_kwargs["encoder_hidden_states"].to(dtype = torch.float16)
        else:
            local_model_kwargs["sample"] = local_model_kwargs["sample"].to(dtype = torch.float32)

        local_model_kwargs["sample"] = self.scheduler.scale_model_input(local_model_kwargs["sample"], ts)

        return local_model_kwargs, model_kwargs


    def step(self, direction, model_kwargs):
        direction = direction.to(dtype = torch.float64)

        timestep, _ = model_kwargs["timestep"].chunk(2, dim=0)
        t = torch.tensor([self.scheduler_timesteps_list.index(ts) for ts in timestep], device = self.device)
        
        # main step  
        new_samples = torch.einsum("i,ijkl->ijkl", self.coeff_x[t], self._samples[t+1]) + \
                                torch.einsum("i,ijkl->ijkl", self.coeff_d[t], direction) + \
                                torch.einsum("i,ijkl->ijkl", self.nl[t], self.noise_vectors[t])
        residual = new_samples - self._samples[t]

        # determine which indexes are finished
        residual_norm = torch.sqrt(torch.mean(residual ** 2, dim =(1,2,3)))
        unfinished_index = torch.where((residual_norm > self.tol1) & (residual_norm > (self.tol2 * self.noise_thr[t])))[0]

        #print(unfinished_index)
        self.taa_step(t, direction, residual, unfinished_index)
        
        self.uninitialized_index = min([self.uninitialized_index, t[0].item()])

        if self.finished_index == 0:
            self._is_finished = True


    def compute_ho_residual(self, t, direction):
        update_times = min([self.order, len(t)])
        #time_start = time.time()
        order_index = torch.clamp(t + update_times, max = t[-1] + 1)

        # slice the matrix from t[0] to t[-1]
        coeff_d_sub_mat = self.coeff_d_mat[t[0]:(t[-1]+1), t[0]:(t[-1]+1)]
        coeff_nl_sub_mat = self.nl_mat[t[0]:(t[-1]+1), (t[0]):(t[-1]+1)]
        # let the index i,j to be zero if j > i + update_times
        coeff_d_sub_mat = torch.tril(coeff_d_sub_mat, diagonal = update_times - 1)
        coeff_nl_sub_mat = torch.tril(coeff_nl_sub_mat, diagonal = update_times - 1)

        new_samples = torch.einsum("i,ijkl->ijkl", self.coeff_x_mat[t, order_index - 1], self._samples[order_index]) + \
                           torch.einsum("ip,pjkl->ijkl", coeff_d_sub_mat, direction) + \
                           torch.einsum("ip,pjkl->ijkl", coeff_nl_sub_mat, self.noise_vectors[t])

        ho_residual = new_samples - self._samples[t]

        return ho_residual

    
    def taa_step(self, t, direction, residual, unfinished_index):
        if len(unfinished_index) == 0:
            self.finished_index = t[0]
        else:
            #print("unfinished_index", unfinished_index, unfinished_index.max(), t)
            self.finished_index = t[unfinished_index.max()].item() 

        ho_residual = self.compute_ho_residual(t, direction)

        Gf = torch.zeros_like(ho_residual)
        if self.residual_memory is None:
            self.residual_memory = torch.zeros(1, self.num_steps + 1, 4, 64, 64, device = self.device, dtype = torch.float64)
            self.samples_memory = torch.zeros(1, self.num_steps + 1, 4, 64, 64, device = self.device, dtype = torch.float64)
            self.residual_memory[0, t] = ho_residual
            self.samples_memory[0, t] = self._samples[t]
            self.memory_indexes[t] = torch.clamp(self.memory_indexes[t] + 1, max = self.memory_size)

        else:
            padded_residual = torch.zeros(1, self.num_steps + 1, 4, 64, 64, device = self.device, dtype = torch.float64)
            padded_samples = torch.zeros(1, self.num_steps + 1, 4, 64, 64, device = self.device, dtype = torch.float64)
            padded_residual[0, t] = ho_residual
            padded_samples[0, t] = self._samples[t]
            self.residual_memory = torch.cat([self.residual_memory, padded_residual], dim = 0)
            self.samples_memory = torch.cat([self.samples_memory, padded_samples], dim = 0)

            self.residual_memory = self.residual_memory[-self.memory_size:]
            self.samples_memory = self.samples_memory[-self.memory_size:]
            self.memory_indexes[t] = torch.clamp(self.memory_indexes[t] + 1, max = self.memory_size)

            residual_diff = self.residual_memory[1:] - self.residual_memory[:-1]
            sample_diff = self.samples_memory[1:] - self.samples_memory[:-1]

            residual_diff_t = residual_diff[:,t,:,:,:]
            sample_diff_t = sample_diff[:,t,:,:,:]

            #print("residual_diff", residual_diff.shape, "sample_diff", sample_diff.shape)
            
            update_indexes = unfinished_index.max() if len(unfinished_index) > 0 else 0
            update_indexes = torch.arange(update_indexes, device = self.device)
            update_indexes = update_indexes[self.memory_indexes[t[update_indexes]] > 1]

            time1 = time.time()
            if len(update_indexes) > 0:
                use_memory = self.memory_indexes[t[update_indexes]] - 1
                #print("use_memory", use_memory)
                use_memory_max = use_memory.max()
                sample_diff_mat = sample_diff_t[:use_memory_max,update_indexes[0]:,:,:,:]
                res_diff_mat = residual_diff_t[:use_memory_max,update_indexes[0]:,:,:,:]
                flip_res_diff_mat = torch.flip(res_diff_mat, dims = [1])
                B = torch.einsum("ijklm,pjklm->ipj", flip_res_diff_mat, flip_res_diff_mat)
                B = torch.flip(torch.cumsum(B, dim = 2), dims = [2])[:,:,:len(update_indexes)]
                B = B.permute(2, 0, 1) # [len(update_indexes), m_k, m_k]
                

                flip_ho_residual = torch.flip(ho_residual[update_indexes[0]:,:,:,:], dims = [0])
                d = torch.einsum("ijklm,jklm->ji", flip_res_diff_mat, flip_ho_residual)
                d = torch.flip(torch.cumsum(d, dim = 0), dims = [0])[:len(update_indexes)] # [len(update_indexes), m_k]
                # let d[i,j] = 0 if j > use_memory[i], cannot use tril because there is a bug
                indices = torch.arange(d.shape[1], device = d.device).unsqueeze(0).expand(d.shape)
                mask_d = (indices + 1) > use_memory.unsqueeze(1)
                #import ipdb; ipdb.set_trace()
                d[mask_d] = 0
                # let B[i,j,k] = 0 if j > use_memory[i] or k > use_memory[i]
                mask_B = mask_d.unsqueeze(2) | mask_d.unsqueeze(1)
                B[mask_B] = 0
                B = B + self.lmd * torch.eye(use_memory_max, device=B.device, dtype = torch.float64).unsqueeze(0) # [len(update_indexes), m_k, m_k]

                if use_memory_max == 1:
                    solve_d = d / B.squeeze(-1)
                else:
                    solve_d = torch.linalg.solve(B, d) # [len(update_indexes), m_k]
                
                A = sample_diff_mat[:,:len(update_indexes),:,:,:] + res_diff_mat[:,:len(update_indexes),:,:,:] # [m_k, len(update_indexes), 4, latent_size, latent_size]
                #print("debug", A.shape, solve_d.shape, update_indexes)
                Gf_flat = torch.einsum("ijklm,ji->jklm", A, solve_d)
                Gf[update_indexes] = Gf_flat
     
        
        self._samples[t] = self._samples[t] + ho_residual - Gf


    def initialize(self, noise = None):
        self._is_finished = False
        self.finished_index = self.num_steps
        self.uninitialized_index = self.num_steps - 1

        if noise is not None:
            self.noise_vectors = noise.to(self.device, dtype = torch.float64)
        else:
            self.noise_vectors = torch.randn(self.num_steps + 1, 4, 64, 64, device = self.device, dtype = torch.float64)
        
        self._samples = torch.zeros(self.num_steps + 1, 4, 64, 64, device = self.device, dtype = torch.float64)
        self._samples[-1] = self.noise_vectors[-1]
            
    
        self.samples_memory = None
        self.residual_memory = None
        self.memory_indexes = torch.zeros(self.num_steps + 1, dtype = torch.long, device = self.device)

    def set_prompt_emebds(self, prompt_embeds):
        self.prompt_embeds = prompt_embeds[0].repeat(self.window_size, 1, 1)
        self.null_prompt_embeds = prompt_embeds[1].repeat(self.window_size, 1, 1)


def parallel_sampling(pipeline, sampler, accelerator, prompt, noise_vec_list, max_steps, variation_steps): 

    pipeline = accelerator.prepare(pipeline)

    use_dtype = torch.float16 if sampler.use_fp16 else torch.float32
    
    images_list  = []
    
    noise = noise_vec_list[0] # we fix the noise vectors for the two prompts, but it is ok to use different noise vectors
    
    for i, p in tqdm(enumerate(prompt)):
        if i == 0:
            # Generate for the prompt 1 from scratch
            sampler.initialize(noise = noise)
        else:
            # Initialize from the trajectory of prompt 1
            samples_traj_copy = sampler._samples.clone()
            sampler.initialize(noise = noise)
            sampler._samples = samples_traj_copy
            sampler.uninitialized_index = -1   
            sampler.finished_index = variation_steps
        
        prompt_embeds = pipeline._encode_prompt(
                        p,
                        "cuda",
                        1,
                        True,
                    )
      
        sampler.set_prompt_emebds(prompt_embeds)

        images = [sampler.get_last_sample()]

        while (not sampler.is_finished()) and (max_steps is None or len(images) < max_steps):
    
            local_model_kwargs, model_kwargs = sampler.prepare_model_kwargs(accelerator.process_index, 
                                                        accelerator.num_processes)
            
            inference_time_start = time.time()
            
            if len(local_model_kwargs["timestep"]) > 0:
                with torch.no_grad():
                    model_output = pipeline.unet(**local_model_kwargs)
                model_output_uncond, model_output_text = model_output[0].chunk(2)
                direction = model_output_uncond + sampler.cfg_scale * (model_output_text - model_output_uncond)

                local_t, _ = local_model_kwargs["timestep"].chunk(2, dim=0)
                local_t = local_t.to(dtype = torch.long)
                local_t = local_t + 1
            else:
                direction = torch.zeros(0, 4, 64, 64, device = accelerator.device, dtype = use_dtype)
                local_t = torch.zeros(0, dtype = torch.long, device = accelerator.device)
     
            accelerator.wait_for_everyone()
            
            direction, local_t = accelerator.pad_across_processes((direction, local_t))
            direction, local_t = accelerator.gather((direction, local_t))
            direction = direction[local_t >= 0.5] # remove the padded direction

            sampler.step(direction, model_kwargs)
            images.append(sampler.get_last_sample())
            
        images_list.append(images)
        
    return images_list



def decode_latent(decoder, latent):
    img = decoder.decode(latent.unsqueeze(0) / 0.18215).sample
    img = torch.clamp(127.5 * img.cpu() + 128.0, 0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()

    return img[0]

def main(args):
    # Load model:
    accelerator = Accelerator()

    device = accelerator.device
    torch.set_grad_enabled(False)
    

    pipeline = StableDiffusionPipeline.from_pretrained("/mnt/workspace/workgroup/tangzhiwei.tzw/sdv1-5-full-diffuser")
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(args.timesteps)

    if args.fp16:
        pipeline = pipeline.to(dtype = torch.float16)

    vae = pipeline.vae.to(dtype = torch.float32)


    ddim_sampler = ParallelDDIM(timesteps = args.timesteps, 
                            scheduler = pipeline.scheduler,
                            cfg_scale= args.cfg_scale, 
                            window_size= args.window_size, 
                            order = args.order, 
                            eta= args.eta,
                            memory_size = args.memory_size,
                            tol1 = args.tol1, 
                            tol2 = args.tol2,
                            device = accelerator.device,
                            use_fp16 = args.fp16)

    # fix seed
    torch.manual_seed(args.seed)    
    # randomly sample the noise vectors for generation
    noise_vectors_list = torch.randn(1, 1, args.timesteps + 1, 4, 64, 64, device = device, dtype = torch.float64)

    # a hack to sync noise_vectors_list from GPU 0 to all GPUs
    noise_vectors_list = accelerator.gather(noise_vectors_list)[0]
    
    if accelerator.is_main_process:
        output_dir = os.path.join(args.output_path, "SD-winit-" + time.strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(output_dir, exist_ok = True)
        # save args
        with open(os.path.join(output_dir, "args.txt"), "w") as f:
            f.write(str(args))
        
    
    images_list = parallel_sampling(pipeline, 
                ddim_sampler,
                accelerator, 
                [args.prompt1, args.prompt2],
                noise_vectors_list,
                args.max_steps,
                args.variation_steps)

    # save images generation process of all samples
    if accelerator.is_main_process:
        for sample in range(2):
            image_dir = os.path.join(output_dir, "sample_{}".format(sample))
            os.makedirs(image_dir, exist_ok = True)
            for step, s in enumerate(images_list[sample]):
                img = decode_latent(vae, s.to(dtype = torch.float32))
                img = Image.fromarray(img)
                img.save(os.path.join(image_dir, "step_{}.png".format(step)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # config for the ddim sampling with classifier-free guidance
    parser.add_argument("--timesteps", type=int, default=50, help="number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="classifier-free guidance")
    parser.add_argument("--eta", type=float, default=0.0, help="eta for DDIM, eta=0 is ODE sampler, eta>0 is SDE sampler")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--prompt1", type=str, default="a cute dog", help="""The prompt use for generating a 
                                                                           initial trajectory for initializing 
                                                                           the generation with prompt2""")
    parser.add_argument("--prompt2", type=str, default="a cute cat", help="""The prompt use for generation with initialization""")

    # config for the ParaTAA
    parser.add_argument("--window_size", type=int, default=100, help="equivalent to effect batch size")
    parser.add_argument("--order", type=int, default=20,help="order of used nonlinear equations")
    parser.add_argument("--memory_size", type=int, default=2,help="""Parameter for Triangular Anderson Acceleration,
                                                    determines the number of previous iteration to be used, recommended to be set between 2-5. 
                                                    If set to 1, no Triangular Anderson Acceleration will be used and
                                                    it reduces to the naive Fixed-Point iteration.""")
    parser.add_argument("--tol1", type=float, default=1e-3, help="tolerance for deciding if the iteration has converged")
    parser.add_argument("--tol2", type=float, default=1e-3, help="tolerance for deciding if the iteration has converged")
    parser.add_argument("--max_steps", type=int, default=None, help="maximum number of iterations")
    parser.add_argument("--variation_steps", type=int, default=10, help="""The numbers of timestep to be updated
                                            for prompt 2 when initializing from prompt 1.
                                            E.g., if timesteps=50 and variation_steps=10, then only the last 10 timesteps will be updated
                                            when doing the generation for prompt 2.
                                            """)

    # config for the model
    parser.add_argument("--model_path", type=str, default="/mnt/workspace/workgroup/tangzhiwei.tzw/sdv1-5-full-diffuser", help="path to the SD v1.5 model")
    parser.add_argument("--output_path", type=str, default="/mnt/workspace/workgroup/tangzhiwei.tzw/spec_samp/ParaTAA-Diffusion/output", help="path to save the generated samples")
    parser.add_argument("--fp16", action = "store_true")
    args = parser.parse_args()
    main(args)
