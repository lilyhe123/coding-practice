#
# Utility to generate code template for a question.
#
# Read problem description from a file.
# The first line of the file is the question number.
# Generate an output file with the description included in a block comment.
# with each line less than 80.
# and create question function and test function.
#
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
        for i, line in enumerate(lines):
            if i == 0:
                num = int(line.strip())
            else:
                text += line

    output = format_question(text, num, 80)
    test_method = "test_" + str(num) + "()"
    with open(output_file, "a+") as output_obj:
        output_obj.seek(0)
        lines = output_obj.readlines()
        for line in lines:
            if test_method in line:
                print(
                    "The question is already included in ",
                    output_file + ".",
                    "Do Nothing.",
                )
                return
        output_obj.write(output)


def generate():
    read_and_format("src/question.txt", "src/daily.py")


if __name__ == "__main__":
    generate()
