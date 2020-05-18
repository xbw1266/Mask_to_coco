import json
import os
import glob


def readJson(json_file):
    assert json_file.endswith(".json"), "Wrong json file input!"
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    data = readJson("good_img.json")
    for img_name in data['good']:
        pass