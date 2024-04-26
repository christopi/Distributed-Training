# TRANSVERSE: An any-to-any MultiModal subnet

TransVerse is an any-to-any multimodal subnet that aims to build the AGI(Artificial General Intelligence) through decentralized distributed ecosystem in Bittensor
Building AGI is the holy grail of computer science but it contains potential threats against humankind
AGI built on truely decentralized distributed AI platform like Bittensor is the only solution to guide the AGI serve for mankind.
By leveraging the unlimited power of decentralized distributed compute resource, TransVerse will be able to build a truely decentralized democratized AGI competing centralized AIs.

# Running miners and validators
## Preparation

### Install requirements
```
pip install -e .
```

### Load pretrained model checkpoint from HuggingFace
Create a ckpt dir
```
mkdir ckpt
mkdir ckpt/pretrained_ckpt
mkdir ckpt/pretrained_ckpt/imagebind_ckpt
mkdir ckpt/pretrained_ckpt/imagebind_ckpt/huge
mkdir ckpt/pretrained_ckpt/7b_tiva_v0
mkdir ckpt/pretrained_ckpt/vicuna_ckpt/
mkdir ckpt/pretrained_ckpt/vicuna_ckpt/7b_v0
```

Download ImageBind checkpoint
```
wget -P ./ckpt/pretrained_ckpt/imagebind_ckpt/huge https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
```


Download checkpoints from huggingface repository
```
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir ckpt/pretrained_ckpt/vicuna_ckpt/7b_v0
huggingface-cli download 3it/TransVerse-v1 --local-dir ckpt/pretrained_ckpt/7b_tiva_v0
```


## Running a miner
deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459 neurons/miner.py --subtensor.network test --wallet.name tw --wallet.hotkey tw-h3 --netuid 74 --axon.port 2012 --logging.debug --logging.trace


## Running a validator

Follow the instructions to prepare for dataset
### Image Data preparation
mkdir data
cd data
huggingface-cli download 3it/TransVerse-Image-Zip --local-dir ./ --repo-type dataset
unzip cc3m.zip
rm cc3m.zip

### Video Data preparation
huggingface-cli download 3it/TransVerse-Video-Zip --local-dir ./ --repo-type dataset
unzip webvid.zip
rm webvid.zip

### Audio Data preparation
huggingface-cli download 3it/TransVerse-Audio-Zip --local-dir ./ --repo-type dataset
unzip audiocap.zip
rm audiocap-full.zip

# Roadmap
- Build decentralized distributed training network to train the any-to-any multimodal model
- Train the state-of-the-art multimodal models by leveraging the unlimited compute resources and open-source datasets
We are planning the collaboration with the omega-labs in building the true AGI by leveraging the unlimited power of Bittensor ecosystem
- Host state-of-the-art any-to-any multimodal models