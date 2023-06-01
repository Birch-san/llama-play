from dataclasses import dataclass, field
from typing import Optional, TypedDict, NamedTuple, List, Dict
import torch
from torch import LongTensor
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  GenerationConfig,
  HfArgumentParser,
  set_seed,
  StoppingCriteria,
  StoppingCriteriaList,
  LlamaForCausalLM,
  LlamaTokenizerFast
)
from peft import PeftModel, PeftModelForCausalLM
from src.callback_text_iterator_streamer import CallbackTextIteratorStreamer
import logging
from enum import Enum
import sys

logger = logging.getLogger(__name__)

class TokenizerOutput(TypedDict):
  input_ids: LongTensor
  attention_mask: LongTensor

class Participant(Enum):
  User = 'user'
  Assistant = 'assistant'
  System = 'system'

class Message(NamedTuple):
  participant: Participant
  message: str

@dataclass
class StopOnTokens(StoppingCriteria):
  stop_token_ids: List[int]
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    for stop_id in self.stop_token_ids:
      if input_ids[0][-1] == stop_id:
        return True
    return False

class SufficientResponse(BaseException): ...

@dataclass
class ModelArguments:
  model_name_or_path: Optional[str] = field(
    default="huggyllama/llama-7b"
  )
  lora_name_or_path: Optional[str] = field(
    default="tloen/alpaca-lora-7b"
  )
  trust_remote_code: Optional[bool] = field(
    default=False,
    metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
  )
  double_quant: bool = field(
    default=True,
    metadata={"help": "Compress the quantization statistics through double quantization."}
  )
  quant_type: str = field(
    default="nf4",
    metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
  )
  bits: int = field(
    default=4,
    metadata={"help": "How many bits to use."}
  )
  bf16: Optional[bool] = field(
    default=False,
    metadata={"help": "Compute type of the model. If quantizing: this is also the compute type used for quantized computations. Prefer to turn this on if you are quantizing and your GPU supports it. You probably also want it even if you're not quantizing."}
  )

@dataclass
class MiscArguments:
  seed: Optional[int] = field(
    default=64,
    metadata={"help": "Random seed, for deterministic generation."}
  )
  compile: bool = field(
    default=False,
    metadata={"help": "Invoke torch.compile() on the model, with mode='max-autotune'. Requires PyTorch 2, CUDA, and either Python 3.10 or Python 3.11 with a recent torch nightly. Will make the first inference from the model take a bit longer, but subsequent inferences will be faster."}
  )
  system_prompt: str = field(
    default="Write a response that answers the request according to the conversation history below.",
    metadata={"help": "The context which precedes the chat history. Can be used to influence the chatbot's responses."}
  )
  overrun_countermeasures: bool = field(
    default=True,
    metadata={"help": "Detect when bot is about to start talking to itself; end the generation before that happens. The bot is *supposed* to emit an end-of-sentence token to indicate that it's finished its reply, but neglects to do so in longer conversations, continuing to sequence-complete both sides of the conversation. Hence this countermeasure tries to detect and prevent that."}
  )

@dataclass
class GenerationArguments:
  # For more hyperparameters check:
  # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
  # Length arguments
  max_new_tokens: Optional[int] = field(
    default=256,
    metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                      "if predict_with_generate is set."}
  )
  min_new_tokens : Optional[int] = field(
    default=None,
    metadata={"help": "Minimum number of new tokens to generate."}
  )

  # Generation strategy
  do_sample: Optional[bool] = field(default=False)
  num_beams: Optional[int] = field(default=1)
  num_beam_groups: Optional[int] = field(default=1)
  penalty_alpha: Optional[float] = field(default=None)
  use_cache: Optional[bool] = field(default=True)

  # Hyperparameters for logit manipulation
  temperature: Optional[float] = field(default=1.0)
  top_k: Optional[int] = field(default=50)
  top_p: Optional[float] = field(default=1.0)
  typical_p: Optional[float] = field(default=1.0)
  diversity_penalty: Optional[float] = field(default=0.0)
  repetition_penalty: Optional[float] = field(default=1.0)
  length_penalty: Optional[float] = field(default=1.0)
  no_repeat_ngram_size: Optional[int] = field(default=0)

def get_model(args: ModelArguments) -> LlamaForCausalLM:
  config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=args.trust_remote_code,
  )
  cuda_avail = torch.cuda.is_available()
  compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
  load_in_4bit = args.bits == 4 and cuda_avail
  load_in_8bit = args.bits == 8 and cuda_avail

  quantization_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=args.double_quant,
    bnb_4bit_quant_type=args.quant_type,
  ) if cuda_avail else None

  if not cuda_avail:
    logger.warning("You don't have CUDA, so we have turned off quantization. If you happen to be on a Mac: you probably have enough unified memory to run in fp16 anyway…")

  if compute_dtype == torch.float16 and cuda_avail and torch.cuda.is_bf16_supported():
    print("Your GPU supports bfloat16; you may want to try it with --bf16 (note: I'm not sure how important this is for inference, but it's certainly preferred when training with 4-bit quantization.)")
  
  model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    device_map='auto',
    quantization_config=quantization_config,
    torch_dtype=compute_dtype,
    trust_remote_code=args.trust_remote_code,
  ).eval()
  model.config.torch_dtype=compute_dtype

  return model

def main():
  hfparser = HfArgumentParser((ModelArguments, GenerationArguments, MiscArguments))
  model_args, generation_args, misc_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
  if extra_args:
    raise f"Received unsupported command-line args: {extra_args}"
  generation_config = GenerationConfig(**vars(generation_args))

  model: LlamaForCausalLM = get_model(model_args)

  model: PeftModelForCausalLM = PeftModel.from_pretrained(
    model,
    model_args.lora_name_or_path,
    # torch_dtype=torch.bfloat16 if model_args.bf16 else torch.float16,
  ).eval()

  set_seed(misc_args.seed)
  if misc_args.compile:
    torch.compile(model, mode='max-autotune')

  tokenizer: LlamaTokenizerFast = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

  # stop_token_ids: List[int] = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
  stop_token_ids: List[int] = [tokenizer.eos_token_id]
  stop = StopOnTokens(stop_token_ids)
  stopping_criteria=StoppingCriteriaList([stop])

  history: List[Message] = [Message(Participant.System, misc_args.system_prompt)] if misc_args.system_prompt else []

  reset_ansi='\x1b[0m'
  blue_ansi='\x1b[31;34m'
  green_ansi='\x1b[31;32m'
  purple_ansi='\x1b[31;35m'
  prompt=f'{purple_ansi}$ '

  participant_names: Dict[Participant, str] = {
    Participant.User: 'Instruction',
    Participant.Assistant: 'Response',
  }

  def format_message(envelope: Message) -> str:
    participant, message = envelope
    if participant is Participant.System:
      return message
    return f'### {participant_names[participant]}:\n{message}'

  first = True
  while True:
    try:
      user_input = input(f'{blue_ansi}Type a message to begin the conversation…{reset_ansi}\n{prompt}' if first else prompt)
    except KeyboardInterrupt:
      sys.exit(0)
    print(reset_ansi, end='')

    first = False
    history += [Message(Participant.User, user_input)]
  
    chat_to_complete: str = '\n\n'.join([
      format_message(message) for message in [
        *history,
        Message(Participant.Assistant, ''),
      ]
    ])

    tokenized_prompts: TokenizerOutput = tokenizer([chat_to_complete], return_tensors='pt', truncation=True)
    
    print(green_ansi, end='', flush=True)

    response = ''
    if misc_args.overrun_countermeasures:
      # the model may continue adding to the conversation (replying to itself) instead of emitting an EOS token.
      # we try to intercept this. If it looks like it's starting a new message in the voice of either of the chat participants: don't print that, and stop generation.
      acc_overrun = ''

      def on_text(message: str, stream_end = False):
        nonlocal response, acc_overrun

        overrun_and_message = f'{acc_overrun}{message}'

        newline_ix = overrun_and_message.find('\n')
        if newline_ix > -1:
          pre_newline = overrun_and_message[:newline_ix]
          newline_onward = overrun_and_message[newline_ix:]

          if newline_onward.startswith('\n\n###'):
            raise SufficientResponse()
          if newline_onward.rstrip('\n\n###') == '':
            # could potentially grow into a \n\n###. Don't print it to the console just yet. we need to accumulate to see whether the bot's about to talk to itself.
            acc_overrun = newline_onward
            response += pre_newline
            print(pre_newline, end='', flush=True)
            return
          # newline_onward cannot grow into an Instruction/Response header, so this must be something else. flush everything we accumulated.

        response += overrun_and_message
        print(overrun_and_message, end='', flush=True)
        acc_overrun = ''
    else:
      def on_text(message: str, stream_end = False):
        nonlocal response
        response += message
        print(message, end='', flush=True)

    streamer = CallbackTextIteratorStreamer(tokenizer, callback=on_text, skip_prompt=True, skip_special_tokens=True)

    try:
      prediction: LongTensor = model.generate(
        input_ids=tokenized_prompts.input_ids.to(model.device),
        attention_mask=tokenized_prompts.attention_mask.to(model.device),
        generation_config=generation_config,
        do_sample=generation_config.temperature > 0.,
        stopping_criteria=stopping_criteria,
        streamer=streamer,
      )
      # if you wanted to see the result, you can do so like this:
      #   decode: List[str] = tokenizer.decode(prediction[0,tokenized_prompts.input_ids.size(-1):], skip_special_tokens=True, clean_up_tokenization_spaces=True)
      # but we're already streaming it to the console via our callback
    except (KeyboardInterrupt, SufficientResponse):
      pass

    # reset ANSI control sequence (plus line break)
    print(reset_ansi)

    # TODO: cull older history, otherwise context will just keep growing larger.
    #       ideally by measuring each message to work out the smallest cull possible.
    history += [Message(Participant.Assistant, response)]

if __name__ == "__main__":
  main()