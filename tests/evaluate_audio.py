import os
import argparse
import json
from medlvlm.datasets.datasets.audio_instruction import AudioInstruction
from torch.utils.data import DataLoader
from tqdm import tqdm
from medlvlm.common.registry import registry
from medlvlm.common.config import Config
from medlvlm.conversation.conversation import Conversation, SeparatorStyle

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

def list_of_str(arg):
    return list(map(str, arg.split(',')))

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to evaluate configuration file.")
    parser.add_argument("--eval-dataset", type=list_of_str, default='audio_val', help="dataset to evaluate")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args

def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], '{}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts

def init_model(cfg):
    print('Initialization Model')
    # cfg.model_cfg.ckpt = args.ckpt
    # cfg.model_cfg.lora_r = args.lora_r
    # cfg.model_cfg.lora_alpha = args.lora_alpha
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')

#     import pudb; pudb.set_trace()
    key = list(cfg.datasets_cfg.keys())[0]
    vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
    text_processor_cfg = cfg.datasets_cfg.get(key).text_processor.train
    audio_processor_cfg = cfg.datasets_cfg.get(key).audio_processor.train

    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)
    audio_processor = registry.get_processor_class(audio_processor_cfg.name).from_config(audio_processor_cfg)

    print('Initialization Finished')
    return model, vis_processor, text_processor, audio_processor

def evaluate(args):
    cfg = Config(args)
    model, vis_processor, text_processor, audio_processor = init_model(cfg)
    model.eval()

    conv_temp = CONV_VISION.copy()

    for dataset in args.eval_dataset:
        eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
        img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"]
        prompt_test = cfg.evaluation_datasets_cfg[dataset]["prompt_test"]
        batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
        max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]
        temperature = cfg.evaluation_datasets_cfg[dataset]["temperature"]
        top_p = cfg.evaluation_datasets_cfg[dataset]["top_p"]
        do_sample = cfg.evaluation_datasets_cfg[dataset]["do_sample"]

        audio_path = cfg.evaluation_datasets_cfg[dataset]["audio_path"]

        data = AudioInstruction(
            vis_processor=vis_processor,
            text_processor=text_processor,
            audio_processor=audio_processor,
            audio_dir=audio_path,
            ann_path=eval_file_path,
            vis_root=img_path,
            prompt_test=prompt_test
        )

        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        results = []
        for batch in tqdm(eval_dataloader):
            images = batch["image"].half()
            audios = batch["audio"]
            instruction_input = batch["instruction_input"]
            ground_truth = batch["answer"]
            image_ids = batch["image_id"]
            texts = prepare_texts(instruction_input, conv_temp)
            predicts = model.generate(images=images,
                                      audios=audios,
                                      texts=texts,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature,
                                      top_p=top_p,
                                      do_sample=do_sample)
            results.extend([{"image_id": image_id, "ground_truth": gt, "predict": predict} for image_id, gt, predict in zip(image_ids, ground_truth, predicts)])
            break
    with open(os.path.join(cfg.run_cfg.save_path, "outputs_test.json"),"w") as jsonfile:
        json.dump(results, jsonfile, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_args()
    print("Evaluating....................")
    evaluate(args)
    print("Done!")