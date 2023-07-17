.PHONY: setup setup-ni setup-vit setup-mani

setup: setup-vit setup-mani

setup-vit:
	echo "Installing dependencies for Vision Transformer..."
	conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
	conda install tqdm
	conda install -c conda-forge faiss
	conda install -c conda-forge timm
	conda install matplotlib
	pip install opencv-python
	pip install git+https://github.com/lucasb-eyer/pydensecrf.git
	conda install -c anaconda scikit-learn

setup-mani:
	echo "Installing maniskill2..."
	pip install mani-skill2
	echo "Downloading maniskill assets..."
	python -m mani_skill2.utils.download_asset all -o ./maniskill/data

setup-ni:
	echo "Installing dependencies for Vision Transformer..."
	conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
	conda install -y tqdm
	conda install -y -c conda-forge faiss
	conda install -y -c conda-forge timm
	conda install -y matplotlib
	pip install opencv-python
	pip install git+https://github.com/lucasb-eyer/pydensecrf.git
	conda install -y -c anaconda scikit-learn
	echo "Installing maniskill2..."
	pip install mani-skill2
	echo "Downloading maniskill assets..."
	python -m mani_skill2.utils.download_asset all -y -o ./maniskill/data
