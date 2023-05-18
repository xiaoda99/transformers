
from dataclasses import dataclass, field
from typing import Optional
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The name of the dataset to use (via the datasets library)."
        })
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The configuration name of the dataset to use (via the datasets library)."
        })
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    max_input_length: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help":
            "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )

    extra_tokens_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The file containing extra tokens to add to the tokenizer."
        },
    )

    group_texts: bool = field(
        default=False,
        metadata={"help": "Whether to group texts together when tokenizing"},
    )
    k_shot: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                # assert extension in [
                #     "csv", "json", "txt"
                # ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                # assert extension in [
                #     "csv", "json", "txt"
                # ], "`validation_file` should be a csv, a json or a txt file."
