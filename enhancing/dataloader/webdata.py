import webdataset as wds
from webdataset import ResampledShards, DataPipeline, tarfile_to_samples
from webdataset.utils import pytorch_worker_seed
from PIL import Image
import io
import sys
import torch
from torch.distributed import init_process_group
from functools import partial

def worker_seed_sat(group=None, seed=0):
    a=pytorch_worker_seed(group=group) + seed * 23
    print('-------------------------------------------------',a)
    return a


class ConfiguredResampledShards(ResampledShards):
    def __init__(self, urls, seed, nshards=sys.maxsize, deterministic=True):
        worker_seed_sat_this = partial(worker_seed_sat, group=None, seed=seed)
        super().__init__(urls, nshards, worker_seed_sat_this, deterministic)


from enhancing.dataloader.transforms import BlipImageTrainProcessor, BlipImageEvalProcessor
def process_fn_ImageWebDataset(src):
    image_processor = BlipImageEvalProcessor(256)
    image_processor = BlipImageTrainProcessor(256)
    for r in src:
        if ('png' not in r and 'jpg' not in r):
            continue

        img_bytes = r['png'] if 'png' in r else r['jpg']
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(e)
            continue
        iw, ih = img.size
        if iw < 14 or ih < 14:
            continue
        img = image_processor(img)

        ret = {'image': img}
        yield ret


class SimpleDistributedWebDataset(DataPipeline):
    def __init__(self, path, *, process_fn=process_fn_ImageWebDataset, seed=42, shuffle_buffer=50000):
        import json
        with open('/zhangpai21/workspace/wwh/VisualGLM/wds_path/cn_path.json') as f:
            cn_path1 = json.load(f)
        with open('/zhangpai21/workspace/wwh/VisualGLM/wds_path/cn_path2.json') as f:
            cn_path2 = json.load(f)
        with open('/zhangpai21/workspace/wwh/VisualGLM/wds_path/laion.json') as f:
            en_path1 = json.load(f)
        with open('/zhangpai21/workspace/wwh/VisualGLM/wds_path/fil-cc12.json') as f:
            en_path2 = json.load(f)
        with open('/zhangpai21/workspace/wwh/VisualGLM/wds_path/vqav2_cn.json') as f:
            vqa_cn = json.load(f)  # sum:436561
        with open('/zhangpai21/workspace/wwh/VisualGLM/wds_path/vqa_en.json') as f:
            vqa_en = json.load(f)  # sum:2436427
        with open('/zhangpai21/workspace/wwh/VisualGLM/wds_path/unusual_vqa.json') as f:
            vqa_en2 = json.load(f)  # sum:45000
        ocr_cn = ['/zhangpai21/webdataset/chinese_ocr/icpr.tar',
                  '/zhangpai21/webdataset/chinese_ocr/rctw.tar']  # sum:17752
        urls = cn_path1 * 14 + cn_path2 * 14 + en_path1 + en_path2 * 10 + vqa_cn * 40 + vqa_en * 30 + vqa_en2 * 60 + ocr_cn * 150

        super().__init__(
            ConfiguredResampledShards(urls, seed),  # Lots of shards are recommended, or not evenly
            tarfile_to_samples(),
            wds.shuffle(shuffle_buffer),
            # set shuffle_buffer = 1 to disable it, TODO model-parallel with different due to shuffle
            process_fn
        )
