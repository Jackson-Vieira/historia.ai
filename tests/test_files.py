import os
import logging

from app.constants import BASE_DIR

import pytest

import math


# FIXTURES

@pytest.fixture
def audio_dir_path():
    return create_safe_audio_dir(aula_prefix_example, aula_num_example, aula_content_example, aula_date_example)

# CONSTANTS

MEDIAS_DIR = os.path.join(BASE_DIR, "tests", "medias")
MEDIAS_TRANSCRIPTIONS_DIR = os.path.join(MEDIAS_DIR, 'transcriptions')
MEDIA_AUDIO_DIR = os.path.join(MEDIAS_DIR, 'audios')

# VARIABLES
aula_prefix_example = "Aula"
aula_num_example = 1
aula_content_example = "CosmovisaoIndigena"
aula_date_example = "2023-05-03"

questao_prefix_example = "Questao"
questao_num_example = 1
questao_generate_date_example = "2023-05-03"

# --- FUNCTIONS ---
def log_directory_creation(dir_path, exist=False):
    action = "created" if not exist else "already exists"
    logging.info("Directory {} {}: {}".format(dir_path, action, exist))

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        logging.info(f"Creating {dir_path}")
        os.mkdir(dir_path)
        return True
    return False

def create_file(file_path):
    if not os.path.exists(file_path):
        open(file_path, 'w')
        return True
    return False

def list_dir_content(dir_path):
    return os.listdir(dir_path)

def auto_increment_subversion(dir_path, current_version=0):
    """
    Auto-increment subversion for resolving conflicts with directories of the same name.
    The maximum limit for subversion is 0.9 (9 subversions and 1 original version).

    Prove:
    1.9 + 0.1 = 2.0 (2.0 maybe has conflicts with another directory)
    """
    INCREMENT = 0.1
    MAX_SUBVERSION = 0.9
    ROUND_PRECISION = 1

    dir_names = list_dir_content(dir_path)
    logging.info("Directories found: {}".format(dir_names))

    versions = [float(dir_name.split('_')[1]) for dir_name in dir_names]
    filtered_versions = [version for version in versions if version >= current_version and version < math.ceil(current_version) + 1]

    logging.info("Versions found: {}".format(filtered_versions))
    last_version = max(filtered_versions, default=0)
    logging.info("Last version found: {}".format(last_version))

    result = round(last_version + INCREMENT, ROUND_PRECISION) if last_version > 0 else 1

    if result > (math.ceil(current_version) + MAX_SUBVERSION):
        # TODO: make a better exception
        raise Exception("Max subversion reached")

    return result

def create_safe_media_directory(base_dir, prefix, num, content, date):
    if not os.path.exists(MEDIAS_DIR):
        logging.info(f"Creating {MEDIAS_DIR}")
        os.mkdir(MEDIAS_DIR)

    if not os.path.exists(base_dir):
        logging.info(f"Creating {base_dir}")
        os.mkdir(base_dir)

    dir_path = os.path.join(base_dir, "{}_{}_{}_{}".format(prefix, str(num), content, date))
    exist = create_dir(dir_path)

    log_directory_creation(dir_path, exist)

    if exist:
        return dir_path

    logging.info("Checking for subversions...")

    subversion = auto_increment_subversion(base_dir, num)

    logging.info("Subversion found: {}".format(subversion))

    dir_path = os.path.join(base_dir, "{}_{}_{}_{}".format(prefix, str(subversion), content, date))

    logging.info("Creating safe directory {}".format(dir_path))
    create_dir(dir_path)
    return dir_path

def create_safe_audio_file(audio_dir_path, audio_prefix, audio_num, audio_extension, audio_date):
    file_path = os.path.join(audio_dir_path, "{}_{}_{}.{}".format(audio_prefix, str(audio_num), audio_date, audio_extension))
    exist = create_file(file_path)

    if exist:
        return file_path

    logging.info("Checking for subversions...")
    subversion = auto_increment_subversion(audio_dir_path, audio_num)

    file_path = os.path.join(audio_dir_path, "{}_{}_{}.{}".format(audio_prefix, str(subversion), audio_date, audio_extension))
    logging.info("Creating safe file {}".format(file_path))
    create_file(file_path)
    return file_path

def create_safe_audio_dir(aula_prefix, aula_num, aula_content, aula_date):
    return create_safe_media_directory(MEDIA_AUDIO_DIR, aula_prefix, aula_num, aula_content, aula_date)

def create_safe_transcription_dir(aula_prefix, aula_num, aula_content, aula_date):
    return create_safe_media_directory(MEDIAS_TRANSCRIPTIONS_DIR, aula_prefix, aula_num, aula_content, aula_date)

# --- TESTS ---

def test_create_safe_audio_dir():
    dir_path = create_safe_audio_dir(aula_prefix_example, aula_num_example, aula_content_example, aula_date_example)
    assert os.path.exists(dir_path)
    os.rmdir(dir_path)

def test_create_safe_audio_file(audio_dir_path):
    file_path = create_safe_audio_file(audio_dir_path, aula_prefix_example, aula_num_example, 'mp3', aula_date_example)
    file_path_subversion = create_safe_audio_file(audio_dir_path, aula_prefix_example, aula_num_example, 'mp3', aula_date_example)

    dir_content = list_dir_content(audio_dir_path)

    assert len(dir_content) == 2
    # assert list_dir_content(audio_dir_path) == ['Aula_1_1_CosmovisaoIndigena_2023-05-03.mp3', 'Aula_1_2_CosmovisaoIndigena_2023-05-03.mp3']
    assert os.path.exists(file_path)
    assert os.path.exists(file_path_subversion)

    os.remove(file_path)
    os.remove(file_path_subversion)
    os.rmdir(audio_dir_path)