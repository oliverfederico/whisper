from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from datasets import Dataset, Audio, load_dataset, DatasetDict, IterableDatasetDict
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# If you have a file path, create a dataset from it:
# data = {"file": ["./hello.wav"], "sentence": ["Hello, my name is Izaak."]}
# dataset = Dataset.from_dict(data)
dataset = IterableDatasetDict()
dataset["train"] = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)#, use_auth_token=True)
dataset["test"] = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)#, use_auth_token=True)

# for ds_name, iterable_ds in dataset.items():
#     ds = Dataset.from_generator(lambda: (yield from iterable_ds), features=iterable_ds.features)
#     dataset[ds_name] = ds

# print(dataset["file"][0])
# Cast the 'file' column as an Audio feature.
# dataset = dataset.cast_column("file", Audio(sampling_rate=16000))

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-tiny"#, language="", task="transcribe"
)

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny"#, language="Hindi", task="transcribe"
)

# print(dataset["file"][0])
# dataset["train"] = dataset["train"].batch(batch_size=16, drop_last_batch=True);

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


dataset = dataset.map(prepare_dataset, remove_columns=["file", "audio", "text", "speaker_id", "chapter_id", "id"])#, batched=True)

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

model.generation_config.forced_decoder_ids = None


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-sample",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    # fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    # report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    dataloader_pin_memory=True,
    dataloader_drop_last=True,
    dataloader_num_workers=6,
    # dataloader_prefetch_factor=2
    # push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

train_result = trainer.train()

trainer.save_model()
trainer.log_metrics("train", train_result.metrics)

from transformers import pipeline
import gradio as gr

pipe = pipeline("automatic-speech-recognition", model="./whisper-tiny-sample")  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

# iface = gr.Interface(
#     fn=transcribe,
#     inputs=gr.Audio(source="microphone", type="filepath"),
#     outputs="text",
#     title="Whisper Base Hindi",
#     description="Realtime demo for Hindi speech recognition using a fine-tuned Whisper base model.",
# )
gr.Interface.from_pipeline(pipe).launch()
# iface.launch()

def main():
    print("Hello from whisper!")


if __name__ == "__main__":
    main()
