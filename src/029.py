"""
question 29
Run-length encoding is a fast and simple method of encoding strings. The basic
idea is to represent repeated successive characters as a single count and
character. For example, the string "AAAABBBCCDAA" would be encoded as
"4A3B2C1D2A".

Implement run-length encoding and decoding. You can assume the string to be
encoded have no digits and consists solely of alphabetic characters. You can
assume the string to be decoded is valid.

-------------------
"AAAABBBCCDAA"
     ^
       ^
42A3B2C
^

num = num * 10 + int(c)
"""


def run_length_encoding(text):
    i1 = 0
    encoded = ""
    c1 = text[0]
    for i2 in range(len(text)):
        c2 = text[i2]
        if c1 != c2:
            encoded += str(i2 - i1)
            encoded += c1
            i1 = i2
            c1 = c2
    # process left_over
    encoded += str(len(text) - i1)
    encoded += text[i1]
    return encoded


def run_length_decoding(encoded):
    num = 0
    text = ""
    for i2 in range(len(encoded)):
        c = encoded[i2]
        if "0" <= c <= "9":
            num = num * 10 + int(c)
        else:
            # decode sequence of one character
            text += c * num
            num = 0
    return text


def test_29():
    text = "AAAABBBCCDAA"
    assert run_length_decoding(run_length_encoding(text)) == text
    text = "AAAAAAAAAAAAAAAABBBCCDDDDDDDDDDDAA"
    assert run_length_decoding(run_length_encoding(text)) == text
