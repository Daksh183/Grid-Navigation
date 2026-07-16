"""Small utility helpers used by the Utilities tab of the GUI."""

import os

NO_OF_CHARS = 256


def get_char_count_array(string):
    """Return a 256-entry array counting occurrences of each character."""
    count = [0] * NO_OF_CHARS
    for ch in string:
        count[ord(ch)] += 1
    return count


def remove_dirty_chars(string, chars_to_remove):
    """Return ``string`` with every character present in ``chars_to_remove`` removed."""
    count = get_char_count_array(chars_to_remove)
    str_list = list(string)

    res_ind = 0
    for ch in str_list:
        if count[ord(ch)] == 0:
            str_list[res_ind] = ch
            res_ind += 1
    return "".join(str_list[:res_ind])


def get_working_directory():
    """Return the current working directory."""
    return os.getcwd()


def simulate_file_upload():
    """Return a dummy file mapping to demonstrate an upload flow."""
    return {"dummy_file.txt": b"This is dummy file content."}
