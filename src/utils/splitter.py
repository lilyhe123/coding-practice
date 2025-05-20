import os
from pathlib import Path

input_name = "first50.py"
question = "question"
file_suffix = ".py"
src_dir = "tt/"


def clean_src():
    for file in Path(src_dir).iterdir():
        if file.is_file():
            file.unlink()


def parse_output(line):
    idx = line.find(question)
    tokens = [token.strip() for token in line[idx:].split(" ")]
    for token in tokens:
        if token != question:
            print(token)
            while len(token) < 3:
                token = "0" + token
            print(token)
            return src_dir + token + file_suffix


def create_one_file(file_name, arr):
    content = "".join(arr) + "\n"
    print(file_name)
    if os.path.isfile(file_name):
        print(file_name, "already exists.")
    else:
        with open(file_name, "w") as output:
            output.write(content)


def read_and_split():
    outputs = []
    with open(input_name) as input:
        lines = input.readlines()
        phase = 0
        arr = []
        output_name = ""
        for line in lines:
            if line.strip() == '"""':
                phase += 1
                if phase == 3:
                    create_one_file(output_name, arr)
                    outputs.append(output_name)
                    phase = 1
                    arr = []
            elif line.find(question) != -1:
                output_name = parse_output(line)
            arr.append(line)

    if len(arr) > 0:
        create_one_file(output_name, arr)
        outputs.append(output_name)
    print(outputs)
    print("Create", len(outputs), "files.")


if __name__ == "__main__":
    # clean_src()
    read_and_split()
