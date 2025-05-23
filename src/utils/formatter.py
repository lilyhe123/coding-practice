#
# Utility to generate code template for a question.
#
# Read from a file.
# The first line of the file is the question number.
# Remaining lines are the problem's description.

# Generate an output file with the description included in a block comment.
# with each line less than 80.
# and create question function and test function.
#
import os

from_file = "src/utils/question.txt"
to_dir = "src/"


def format_question(text: str, num: int, limit: int) -> str:
    output = '\n"""\nquestion ' + str(num) + "\n"
    cur_line = ""
    for line in text.splitlines():
        for token in line.split(" "):
            token = token.strip()
            if len(token) == 0:
                continue
            if len(cur_line) + 1 + len(token) > limit:
                # need to create a new line, save current line first
                output += cur_line + "\n"
                cur_line = ""
            if len(cur_line) > 0:
                cur_line += " "
            cur_line += token
        # add left_over
        if len(cur_line) > 0:
            output += cur_line + "\n\n"
            cur_line = ""

    # add left_over
    if len(cur_line) > 0:
        output += cur_line + "\n\n"
    output += '-------------------\n\n\n"""\n'
    output += "def question" + str(num) + "(): pass \n"
    output += "def test_" + str(num) + "(): pass \n"
    output += "test_" + str(num) + "() \n"
    return output


def read_and_format(input_file, output_file):
    with open(input_file, "r") as input_obj:
        lines = input_obj.readlines()
        text = ""
        suffix = ""
        for i, line in enumerate(lines):
            if i == 0:
                suffix = line.strip()
                while len(suffix) < 3:
                    suffix = "0" + suffix
                num = int(line.strip())
            else:
                text += line
    output_file += suffix + ".py"
    if os.path.isfile(output_file):
        print(output_file, "already exists. Do nothing.")
        return
    output = format_question(text, num, 80)
    with open(output_file, "w") as output_obj:
        output_obj.write(output)


def generate():
    read_and_format(from_file, to_dir)


if __name__ == "__main__":
    generate()
