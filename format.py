# Read problem description from a file
# the first line of the file is the quesiton number.
# Generate a output file with the description included in a block comment
# with each line less than 80.
# and create question and test functions for it.
def format_question(text: str, num: int, limit: int) -> str:
    output = '"""\nquestion ' + str(num) + "\n"
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
    with open(input_file) as file_object:
        lines = file_object.readlines()
        text = ""
        for i, line in enumerate(lines):
            if i == 0:
                num = int(line.strip())
            else:
                text += line

    output = format_question(text, num, 80)
    with open(output_file, "w") as file_object:
        file_object.write(output)


read_and_format("question.txt", "output.txt")
