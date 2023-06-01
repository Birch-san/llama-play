# LLaMA-Play

<img width="420" alt="Type a message to begin the conversation…/$ hey I'm mahouko/Hello, Mahouko, it's good to meet you./$ what's my name again?/What do you mean, Mahouko?/$ oh phew it worked/That's great! What do you need help with?/$ what's my name?/Hey Mahouko, what can I help you with?/$ you've already done enough/That's okay. You can always ask me for help if you need it." src="https://github.com/Birch-san/llama-play/assets/6141784/ac07e9f9-e344-4075-9972-a8613a20e58b">

Python script to demonstrate how to invoke models such as LLaMA from the command-line, with LoRA adapters.

_Uses the [`huggyllama/llama-7b`](https://huggingface.co/huggyllama/llama-7b) LLaMA distribution by default, but if you have the official LLaMA weights and would prefer to convert them to Huggingface format yourself: I provide [instructions for doing so](https://gist.github.com/Birch-san/0b2d2f9bd997801005c1b5acbbc1dc0f)._  
_Uses the [`alpaca-lora-7b`](https://huggingface.co/tloen/alpaca-lora-7b) LoRA by default, to adapt LLaMA for instruction-following._

## Setup

All instructions are written assuming your command-line shell is bash.

Clone repository:

```bash
git clone https://github.com/Birch-san/llama-play.git
cd llama-play
```

### Create + activate a new virtual environment

This is to avoid interfering with your current Python environment (other Python scripts on your computer might not appreciate it if you update a bunch of packages they were relying on).

Follow the instructions for virtualenv, or conda, or neither (if you don't care what happens to other Python scripts on your computer).

#### Using `venv`

**Create environment**:

```bash
. ./venv/bin/activate
pip install --upgrade pip
```

**Activate environment**:

```bash
. ./venv/bin/activate
```

**(First-time) update environment's `pip`**:

```bash
pip install --upgrade pip
```

#### Using `conda`

**Download [conda](https://www.anaconda.com/products/distribution).**

_Skip this step if you already have conda._

**Install conda**:

_Skip this step if you already have conda._

Assuming you're using a `bash` shell:

```bash
# Linux installs Anaconda via this shell script. Mac installs by running a .pkg installer.
bash Anaconda-latest-Linux-x86_64.sh
# this step probably works on both Linux and Mac.
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda config --set auto_activate_base false
conda init
```

**Create environment**:

```bash
conda create -n p311-llama python=3.11
```

**Activate environment**:

```bash
conda activate p311-llama
```

### Install package dependencies

**Ensure you have activated the environment you created above.**

(Optional) treat yourself to latest nightly of PyTorch, with support for Python 3.11 and CUDA 12.1:

```bash
# CUDA
pip install --upgrade --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu121
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run:

From root of repository:

```bash
python -m scripts.chat_play --bf16
```

## License

This repository is itself MIT-licensed.

Includes MIT-licensed code copied from Artidoro Pagnoni's [qlora](https://github.com/artidoro/qlora), and [Apache-licensed](licenses/MosaicML-mpt-7b-chat-hf-space.Apache.LICENSE.txt) code copied from MosaicML's [mpt-7b-chat](https://huggingface.co/spaces/mosaicml/mpt-7b-chat/blob/main/app.py) Huggingface Space.