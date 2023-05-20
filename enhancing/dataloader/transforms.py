from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from enhancing.dataloader.randaugment import RandomAugment
import numpy as np
from copy import deepcopy
from PIL import Image
import re

class BlipImageBaseProcessor():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

class BlipImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                # self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                # self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)


class TextProcessor():
    def __init__(self, tokenizer, max_target_length=1024):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __call__(self, caption, prompt=""):
        caption = self.pre_caption(caption)
        # print(prompt,caption)
        pre_prompt, post_prompt = prompt.split("<Image>")
        pre_prompt_ids = self.tokenizer.encode(pre_prompt, add_special_tokens=False)
        post_prompt_ids = self.tokenizer.encode(post_prompt, add_special_tokens=False)
        pre_image = len(pre_prompt_ids)
        prompt_ids = pre_prompt_ids + post_prompt_ids # self.tokenizer.encode(prompt, add_special_tokens=False)
        caption_ids = self.tokenizer.encode(caption, add_special_tokens=False)
        if len(prompt_ids) + len(caption_ids) > self.max_target_length-3:
            caption_ids = caption_ids[: self.max_target_length-len(prompt_ids)-3]
        input_ids = self.tokenizer.build_inputs_with_special_tokens(deepcopy(prompt_ids),caption_ids)

        context_length = input_ids.index(self.tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [-100] * context_length + input_ids[mask_position + 1:]

        pad_len = self.max_target_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len

        return input_ids,labels,pre_image

    def pre_caption(self, caption):
        # caption = re.sub(
        #     r"([.!\"()*#:;~])",
        #     " ",
        #     caption,
        # )

        # caption = re.sub(
        #     r"([^\w\s,.?，。？、()（）]|_)+",
        #     " ",
        #     caption,
        # )

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
        return caption

class UnmaskTextProcessor(TextProcessor):
    def __call__(self, caption):
        caption = self.pre_caption(caption)
        caption_tokens = self.tokenizer.tokenize(caption)
        caption_tokens += ["<sop>","[gMASK]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)
        if len(input_ids) > self.max_target_length:
            input_ids = input_ids[-self.max_target_length:]

        pad_len = self.max_target_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        return input_ids

if __name__ == "__main__":
    img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img, mode='RGB')
    transform_train = BlipImageTrainProcessor(224)
    transform_eval = BlipImageEvalProcessor(288)
    result = transform_train(pil_img)
    print('ok')
