import gradio as gr
import subprocess
import yaml
import os

training_options = [
    "invoke-finetune-lora-sd",
    "invoke-finetune-lora-sdxl",
    "invoke-dreambooth-lora-sd",
    "invoke-dreambooth-lora-sdxl"]

def run_script(training_type, base_output_dir, learning_rate, optimizer_type, weight_decay, use_bias_correction, safeguard_warmup, dataset_name, image_resolution, model, seed, gradient_accumulation_steps, mixed_precision, xformers, gradient_checkpointing, max_train_steps, save_every_n_epochs, save_every_n_steps, max_checkpoints, validation_prompts, validate_every_n_epochs, train_batch_size, num_validation_images_per_prompt):
    if training_type is not None:
        try:
            # Prepare the config data
            config_data = {
                'output': {'base_output_dir': base_output_dir},
                'optimizer': {
                    'learning_rate': learning_rate,
                    'optimizer': {
                        'optimizer_type': optimizer_type,
                        'weight_decay': weight_decay,
                        'use_bias_correction': use_bias_correction,
                        'safeguard_warmup': safeguard_warmup
                    }
                },
                'dataset': {
                    'dataset_name': dataset_name,
                    'image_transforms': {'resolution': image_resolution}
                },
                'model': model,
                'seed': seed,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'mixed_precision': mixed_precision,
                'xformers': xformers,
                'gradient_checkpointing': gradient_checkpointing,
                'max_train_steps': max_train_steps,
                'save_every_n_epochs': save_every_n_epochs,
                'save_every_n_steps': save_every_n_steps,
                'max_checkpoints': max_checkpoints,
                'validation_prompts': validation_prompts.split('|'),
                'validate_every_n_epochs': validate_every_n_epochs,
                'train_batch_size': train_batch_size,
                'num_validation_images_per_prompt': num_validation_images_per_prompt
            }

            # Create a YAML file with the config
            config_filename = 'config.yaml'
            with open(config_filename, 'w') as file:
                yaml.dump(config_data, file, sort_keys=False)

            command = training_type + ' --cfg-file ' + config_filename
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            return stdout.decode() + stderr.decode()
        
        except Exception as e:
            return str(e)
    
    else:
        return "Invalid script selected."

iface = gr.Interface(
    fn=run_script,
    inputs=[
        gr.Dropdown(training_options, label="Training Type"),
        gr.Textbox(label="Base Output Directory"),
        gr.Number(label="Learning Rate"),
        gr.Textbox(label="Optimizer Type"),
        gr.Number(label="Weight Decay"),
        gr.Checkbox(label="Use Bias Correction"),
        gr.Checkbox(label="Safeguard Warmup"),
        gr.Textbox(label="Dataset Name"),
        gr.Number(label="Image Resolution"),
        gr.Textbox(label="Model"),
        gr.Number(label="Seed"),
        gr.Number(label="Gradient Accumulation Steps"),
        gr.Radio(choices=["fp16", "fp32"], label="Mixed Precision"),
        gr.Checkbox(label="Xformers"),
        gr.Checkbox(label="Gradient Checkpointing"),
        gr.Number(label="Max Train Steps"),
        gr.Number(label="Save Every N Epochs"),
        gr.Number(label="Save Every N Steps"),
        gr.Number(label="Max Checkpoints"),
        gr.Textbox(label="Validation Prompts (separate with '|')"),
        gr.Number(label="Validate Every N Epochs"),
        gr.Number(label="Train Batch Size"),
        gr.Number(label="Number of Validation Images per Prompt")
    ],
    outputs=gr.Textbox(label="Script Output"),
    title="Train LoRAs with Invoke-Training",
    description="Fill in the configuration options and press Start to run.",
    allow_flagging="never"
)

iface.launch()
