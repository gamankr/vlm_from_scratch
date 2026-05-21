from collections.abc import Iterable
from typing import Any
import numpy as np
from PIL import Image
import torch


IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5] # one for each channels R, G, B
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_length, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_length}{bos_token}{prefix_prompt}\n"

def resize(
        image: np.ndarray,
        size: tuple[int, int],
        resample: Image.Resampling = None,
        reducing_gap: int | None = None
        ) -> np.ndarray:
    
    height, width = size
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    return resized_image

def rescale(image: np.ndarray,
            scale: float,
            dtype: np.dtype = np.float32
            ) -> np.ndarray:
    
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def normalize(image: np.ndarray,
              mean: float | Iterable[float],
              std: float| Iterable[float]
              ) -> np.ndarray:
    
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image =  (image - mean) / std
    return image


def process_images(
        images: list(Image.Image),
        size: dict(str, int) = None,
        resample: Image.Resampling = None,
        rescale_factor: float = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
    ) -> list[np.ndarray]:
      
    height, width = size[0], size[1]
    images = [resize(image=image, size=(height, width), resample=resample) for image in images]

    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in range [0,1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first position. Model expects the images of shape [channels, height, width]
    # [height, width, channels] ---> [channels, height, width]
    images = [image.transpose(2,0,1) for image in images]

    return images



class PaliGemmaProcessor:
    '''
    Given an image and prompt, this class will:
        - preprocess the images for the SiglipVisionModel encoder
        - create a prompt that combines text with the image token 
        - tokenize the prompt


    This processor adds following capabilities to tokenizer:
        - Add a token to tokenizer to represent images
        - Add tokens to tokensizer for object detection and object segmentation
        - Instantiates the tokenizer
    '''
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]   # These tokens are used for object detection (bounding boxes)
            # The number after 'loc' is the pixel number and a bounding box is represented by 4 'loc' values, each for the 4 corners
            
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]   # These tokens are used for object segmentation

        tokenizer.add_tokens(EXTRA_TOKENS) # add_tokens expects a list
                                           # https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PythonBackend.add_tokens

        # Add a token to represent the image.
        # Acts as a placeholder which will be replaced by the image embedding from the SigLIP encoder
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add) # add_special_tokens expects a dictionary
                                                    # https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PythonBackend.add_special_tokens

        # we will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self,
                 text: list[str],
                 images: list[Image.Image],
                 padding: str = "longest",
                 truncation: bool = True,
        ) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts"

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # Convert the list of numpy array into a single numpy array of shape [batch_size, channels, height, width]
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert numpy array to Torch tensor
        pixel_values = torch.from_numpy(pixel_values)

        # Prepend 'self.image_seq_length' number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_length=self.image_seq_length,
                image_token=self.IMAGE_TOKEN
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as Torch tensor
        # input_ids: positions of the tokens in the vocabulary
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
        
        
        

