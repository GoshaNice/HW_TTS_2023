CODE = src

switch_to_macos:
	rm poetry.lock
	cat utils/pyproject_macos.txt > pyproject.toml

switch_to_linux:
	rm poetry.lock
	cat utils/pyproject_linux.txt > pyproject.toml

install:
	python3.10 -m pip install poetry
	poetry install

lint:
	poetry run pflake8 $(CODE)

format:
	#format code
	poetry run black $(CODE)

download_checkpoint:
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1v0Gmhw32BTy1l8yjdmxswEYXJdL2UQ4b" -O default_test_model/model_best.pth && rm -rf /tmp/cookies.txt

synthesize:
	poetry run python synthesize.py -r default_test_model/model_best.pth -i test_data_folder/input.txt

train:
	poetry run python train.py -c src/config.json

download_waveglow:
	poetry run gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
	mkdir -p waveglow/pretrained_model/
	mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt